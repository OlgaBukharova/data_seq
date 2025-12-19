from __future__ import annotations

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


@dataclass
class Cfg:
    # Payload: 8 digits -> 8 * 4 bits (BCD) = 32 bits
    n_digits: int = 8
    L: int = 32

    batch_size: int = 256
    epochs: int = 12
    lr: float = 2e-4

    # Curriculum (быстро учим декодировать, потом улучшаем незаметность)
    warm_epochs: int = 3

    eps_warm: float = 0.55     # больше сигнал в начале
    eps_main: float = 0.25     # потом ужимаем

    alpha_warm: float = 0.05   # вес MSE (малый)
    beta_warm: float = 30.0    # вес сообщения (большой)
    lam_warm: float = 0.0      # штраф на |delta| в начале выключен

    alpha_main: float = 0.70
    beta_main: float = 10.0
    lam_main: float = 0.001

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True

    print_every: int = 200
    ckpt_dir: str = "checkpoints"


def stage_params(epoch0: int, cfg: Cfg):
    """epoch0: 0-based"""
    if epoch0 < cfg.warm_epochs:
        return cfg.eps_warm, cfg.alpha_warm, cfg.beta_warm, cfg.lam_warm
    return cfg.eps_main, cfg.alpha_main, cfg.beta_main, cfg.lam_main


@torch.no_grad()
def demo(enc: Encoder, dec: Decoder, x: torch.Tensor, cfg: Cfg, device: torch.device):
    bits, texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)
    x_stego, _delta = enc(x, bits, return_delta=True)
    logits = dec(x_stego)
    pred_bits = logits_to_bits(logits)
    pred_text = bits_to_digits(pred_bits[0], n_digits=cfg.n_digits)
    return texts[0], pred_text


def main():
    cfg = Cfg()
    assert cfg.L == cfg.n_digits * 4, "BCD: L должно равняться n_digits*4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Data: CIFAR10 -> [-1,1]
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
    enc = Encoder(L=cfg.L, hidden=96, eps=cfg.eps_warm).to(device)
    dec = Decoder(L=cfg.L, hidden=160).to(device)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        eps, alpha, beta, lam = stage_params(epoch - 1, cfg)
        enc.set_eps(eps)

        enc.train()
        dec.train()

        run_loss = 0.0
        run_psnr = 0.0
        run_ber = 0.0
        run_acc = 0.0

        for step, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            m_bits, _texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_stego, delta = enc(x, m_bits, return_delta=True)
                logits = dec(x_stego)

                loss_img = mse(x_stego, x)
                loss_msg = bce(logits, m_bits)
                loss_delta = delta.abs().mean()

                loss = alpha * loss_img + beta * loss_msg + lam * loss_delta

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

                logits_mean = float(logits.mean().item())
                logits_std = float(logits.std().item())
                delta_mean = float(delta.abs().mean().item())
                m_mean = float(m_bits.mean().item())

                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"loss={avg_loss:.4f} PSNR={avg_psnr:.2f} BER={avg_ber:.4f} acc={avg_acc:.4f} "
                    f"logits(mean/std)={logits_mean:.3f}/{logits_std:.3f} "
                    f"(eps={eps:.2f}, alpha={alpha}, beta={beta}, lambda_delta={lam})"
                )
                print(f"  m_bits.mean: {m_mean:.3f} |delta|.mean: {delta_mean:.6f}")

        steps = len(train_loader)
        ep_loss = run_loss / steps
        ep_psnr = run_psnr / steps
        ep_ber = run_ber / steps
        ep_acc = run_acc / steps

        # demo
        enc.eval()
        dec.eval()
        x_demo, _ = next(iter(train_loader))
        x_demo = x_demo.to(device)
        target, pred = demo(enc, dec, x_demo[:cfg.batch_size], cfg, device)

        print(f"Epoch {epoch}: loss={ep_loss:.4f} PSNR={ep_psnr:.2f} BER={ep_ber:.4f} acc={ep_acc:.4f}")
        print(f"  demo target: '{target}'  pred: '{pred}'")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"stego_ed_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "cfg": cfg.__dict__,
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "opt": opt.state_dict(),
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)

    final_path = os.path.join(cfg.ckpt_dir, "stego_ed_final.pt")
    torch.save(
        {
            "epoch": cfg.epochs,
            "cfg": cfg.__dict__,
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
