from __future__ import annotations
from dataclasses import dataclass, field

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.encoder import Encoder
from models.decoder import Decoder
from utils.payload import make_random_digit_batch
from utils.metrics import psnr, ber_from_logits
from utils.bits import logits_to_bits, bits_to_digits
from utils.channel import ChannelCfg, apply_channel


@dataclass
class Cfg:
    n_digits: int = 8
    L: int = 32

    batch_size: int = 256
    epochs: int = 10
    lr: float = 2e-4

    # start from your good baseline
    init_ckpt: str = "checkpoints/stego_ed_epoch18.pt"

    # training regime
    freeze_encoder_epochs: int = 2  # first epochs: train decoder only

    # loss weights
    alpha: float = 1.0   # image MSE
    beta: float = 9.0    # message BCE
    lam_delta: float = 0.001

    # eps for encoder during channel training (slightly conservative)
    eps: float = 0.20

    # channel config
    ch: ChannelCfg = ChannelCfg(
        p_noise=0.50, noise_std=0.03,
        p_resize=0.30, resize_min=0.60, resize_max=1.00,
        p_crop=0.30, crop_min=0.70, crop_max=1.00,
    )

    num_workers: int = 2
    pin_memory: bool = True
    print_every: int = 200
    ckpt_dir: str = "checkpoints"


@torch.no_grad()
def demo(enc: Encoder, dec: Decoder, x: torch.Tensor, cfg: Cfg, device: torch.device):
    bits, texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)
    x_stego, _delta = enc(x, bits, return_delta=True)
    x_noisy = apply_channel(x_stego, cfg.ch)
    logits = dec(x_noisy)
    pred_bits = logits_to_bits(logits)
    pred_text = bits_to_digits(pred_bits[0], n_digits=cfg.n_digits)
    return texts[0], pred_text


def set_requires_grad(model: torch.nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def main():
    cfg = Cfg()
    assert cfg.L == cfg.n_digits * 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    # Models
    enc = Encoder(L=cfg.L, hidden=96, eps=cfg.eps).to(device)
    dec = Decoder(L=cfg.L, hidden=160).to(device)

    # Load baseline weights
    ckpt = torch.load(cfg.init_ckpt, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    print("Loaded baseline ckpt:", cfg.init_ckpt)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # Optimizer (will update params that require_grad=True)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # freeze encoder for first N epochs
        if epoch <= cfg.freeze_encoder_epochs:
            set_requires_grad(enc, False)
            set_requires_grad(dec, True)
        else:
            set_requires_grad(enc, True)
            set_requires_grad(dec, True)

        enc.train()
        dec.train()

        run_loss = run_psnr = run_ber = run_acc = 0.0

        for step, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            m_bits, _texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_stego, delta = enc(x, m_bits, return_delta=True)
                x_noisy = apply_channel(x_stego, cfg.ch)

                logits = dec(x_noisy)

                loss_img = mse(x_stego, x)          # незаметность считаем по x_stego vs x
                loss_msg = bce(logits, m_bits)      # сообщение после канала
                loss_delta = delta.abs().mean()

                loss = cfg.alpha * loss_img + cfg.beta * loss_msg + cfg.lam_delta * loss_delta

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                _psnr = psnr(x_stego, x)
                _ber, _acc = ber_from_logits(logits, m_bits)

            run_loss += float(loss.item())
            run_psnr += float(_psnr)
            run_ber += float(_ber)
            run_acc += float(_acc)

            if step % cfg.print_every == 0:
                avg_loss = run_loss / step
                avg_psnr = run_psnr / step
                avg_ber = run_ber / step
                avg_acc = run_acc / step
                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"loss={avg_loss:.4f} PSNR={avg_psnr:.2f} BER={avg_ber:.4f} acc={avg_acc:.4f} "
                    f"(freeze_enc={epoch <= cfg.freeze_encoder_epochs})"
                )

        steps = len(train_loader)
        ep_loss = run_loss / steps
        ep_psnr = run_psnr / steps
        ep_ber = run_ber / steps
        ep_acc = run_acc / steps

        enc.eval()
        dec.eval()
        x_demo, _ = next(iter(train_loader))
        x_demo = x_demo.to(device)
        target, pred = demo(enc, dec, x_demo[:cfg.batch_size], cfg, device)

        print(f"Epoch {epoch}: loss={ep_loss:.4f} PSNR={ep_psnr:.2f} BER={ep_ber:.4f} acc={ep_acc:.4f}")
        print(f"  demo target: '{target}'  pred: '{pred}'")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"stego_ed_channel_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "cfg": {
                    **cfg.__dict__,
                    "ch": cfg.ch.__dict__,
                },
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "opt": opt.state_dict(),
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)

    final_path = os.path.join(cfg.ckpt_dir, "stego_ed_channel_final.pt")
    torch.save(
        {
            "epoch": cfg.epochs,
            "cfg": {
                **cfg.__dict__,
                "ch": cfg.ch.__dict__,
            },
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "opt": opt.state_dict(),
        },
        final_path,
    )
    print("Saved:", final_path)
    print(f"Done. Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
