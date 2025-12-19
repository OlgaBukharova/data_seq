from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator

from utils.payload import make_random_digit_batch
from utils.metrics import psnr, ber_from_logits
from utils.channel import ChannelCfg, apply_channel


@dataclass
class Cfg:
    # payload
    n_digits: int = 8
    L: int = 32

    # data / train
    batch_size: int = 256
    epochs: int = 10
    lr_g: float = 2e-4
    lr_d: float = 2e-4

    # init ckpts
    init_stego_ckpt: str = "checkpoints/stego_ed_channel_epoch10.pt"
    init_detector_ckpt: str = "checkpoints/detector_final.pt"

    # encoder eps
    eps: float = 0.20

    # losses for G = Enc+Dec
    alpha: float = 1.0     # image MSE on (x_stego vs x)
    beta: float = 9.0      # message BCE on (Dec(channel(x_stego)) vs bits)
    gamma_start: float = 0.0   # adv weight ramp
    gamma_end: float = 0.30
    gamma_warm_epochs: int = 2  # epochs where gamma ramps up

    lam_delta: float = 0.001

    # how often to train D relative to G
    d_steps: int = 1
    g_steps: int = 1

    # channel (same as step 3)
    ch: ChannelCfg = field(default_factory=lambda: ChannelCfg(
        p_noise=0.50, noise_std=0.03,
        p_resize=0.30, resize_min=0.60, resize_max=1.00,
        p_crop=0.30, crop_min=0.70, crop_max=1.00,
    ))

    # loader / logging
    num_workers: int = 2
    pin_memory: bool = True
    print_every: int = 200
    ckpt_dir: str = "checkpoints"


def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def detector_acc(D: Discriminator, x_clean: torch.Tensor, x_stego: torch.Tensor) -> float:
    """
    Returns accuracy on a balanced batch: clean=0, stego=1
    """
    B0 = x_clean.size(0)
    B1 = x_stego.size(0)
    x = torch.cat([x_clean, x_stego], dim=0)
    y = torch.cat([
        torch.zeros(B0, device=x.device),
        torch.ones(B1, device=x.device),
    ], dim=0)
    logits = D(x)
    pred = (logits >= 0).float()
    return float((pred == y).float().mean().item())


def gamma_for_epoch(epoch0: int, cfg: Cfg) -> float:
    """
    epoch0 is 0-based. Linear ramp gamma from start to end during warm epochs.
    """
    if cfg.gamma_warm_epochs <= 0:
        return cfg.gamma_end
    if epoch0 >= cfg.gamma_warm_epochs:
        return cfg.gamma_end
    t = (epoch0 + 1) / cfg.gamma_warm_epochs  # 0..1
    return cfg.gamma_start + t * (cfg.gamma_end - cfg.gamma_start)


def main():
    cfg = Cfg()
    assert cfg.L == cfg.n_digits * 4

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
    enc = Encoder(L=cfg.L, hidden=96, eps=cfg.eps).to(device)
    dec = Decoder(L=cfg.L, hidden=160).to(device)
    D = Discriminator(hidden=64).to(device)

    # Load init checkpoints
    ckpt_s = torch.load(cfg.init_stego_ckpt, map_location=device)
    enc.load_state_dict(ckpt_s["encoder"])
    dec.load_state_dict(ckpt_s["decoder"])
    print("Loaded stego ckpt:", cfg.init_stego_ckpt)

    ckpt_d = torch.load(cfg.init_detector_ckpt, map_location=device)
    D.load_state_dict(ckpt_d["detector"])
    print("Loaded detector ckpt:", cfg.init_detector_ckpt)

    # Opts
    opt_g = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=cfg.lr_g,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    opt_d = torch.optim.AdamW(
        D.parameters(),
        lr=cfg.lr_d,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch0 = epoch - 1
        gamma = gamma_for_epoch(epoch0, cfg)

        enc.train()
        dec.train()
        D.train()

        run_g_loss = 0.0
        run_d_loss = 0.0
        run_psnr = 0.0
        run_ber = 0.0
        run_dacc = 0.0

        for step, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            B = x.size(0)
            m_bits, _txt = make_random_digit_batch(B, n_digits=cfg.n_digits, device=device)

            # ------------------------
            # (1) Train Discriminator
            # ------------------------
            set_requires_grad(D, True)
            set_requires_grad(enc, False)
            set_requires_grad(dec, False)

            for _ in range(cfg.d_steps):
                opt_d.zero_grad(set_to_none=True)

                with torch.no_grad():
                    x_stego = enc(x, m_bits)  # stego generation (no grad)
                # D sees clean vs stego (no channel here обычно сильнее; можно добавить channel позже)
                x_mix = torch.cat([x, x_stego], dim=0)
                y_mix = torch.cat([
                    torch.zeros(B, device=device),
                    torch.ones(B, device=device),
                ], dim=0)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = D(x_mix)
                    d_loss = bce_logits(logits, y_mix)

                scaler.scale(d_loss).backward()
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                scaler.step(opt_d)
                scaler.update()

            # ------------------------
            # (2) Train Generator (Enc+Dec)
            # ------------------------
            set_requires_grad(D, False)
            set_requires_grad(enc, True)
            set_requires_grad(dec, True)

            for _ in range(cfg.g_steps):
                opt_g.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    x_stego, delta = enc(x, m_bits, return_delta=True)

                    # message after channel distortions
                    x_noisy = apply_channel(x_stego, cfg.ch)
                    logits_msg = dec(x_noisy)

                    loss_img = mse(x_stego, x)
                    loss_msg = bce_logits(logits_msg, m_bits)
                    loss_delta = delta.abs().mean()

                    # adversarial: want D to classify stego as CLEAN (0)
                    logits_d = D(x_stego)
                    y_fool = torch.zeros_like(logits_d)
                    loss_adv = bce_logits(logits_d, y_fool)

                    g_loss = (
                        cfg.alpha * loss_img
                        + cfg.beta * loss_msg
                        + cfg.lam_delta * loss_delta
                        + gamma * loss_adv
                    )

                scaler.scale(g_loss).backward()
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
                scaler.step(opt_g)
                scaler.update()

            # metrics (no grad)
            with torch.no_grad():
                _psnr = psnr(x_stego, x)
                _ber, _acc = ber_from_logits(logits_msg, m_bits)
                _dacc = detector_acc(D, x_clean=x[:B//2], x_stego=x_stego[B//2:])

            run_g_loss += float(g_loss.item())
            run_d_loss += float(d_loss.item())
            run_psnr += float(_psnr)
            run_ber += float(_ber)
            run_dacc += float(_dacc)

            if step % cfg.print_every == 0:
                avg_g = run_g_loss / step
                avg_d = run_d_loss / step
                avg_p = run_psnr / step
                avg_b = run_ber / step
                avg_da = run_dacc / step
                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"G={avg_g:.4f} D={avg_d:.4f} PSNR={avg_p:.2f} BER={avg_b:.4f} D_acc={avg_da:.3f} "
                    f"(gamma={gamma:.3f})"
                )

        steps = len(train_loader)
        ep_g = run_g_loss / steps
        ep_d = run_d_loss / steps
        ep_psnr = run_psnr / steps
        ep_ber = run_ber / steps
        ep_dacc = run_dacc / steps

        print(
            f"Epoch {epoch}: G={ep_g:.4f} D={ep_d:.4f} PSNR={ep_psnr:.2f} BER={ep_ber:.4f} "
            f"D_acc={ep_dacc:.3f} (gamma={gamma:.3f})"
        )

        ckpt_path = os.path.join(cfg.ckpt_dir, f"stego_adv_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "cfg": {
                    **cfg.__dict__,
                    "ch": cfg.ch.__dict__,
                    "gamma": gamma,
                },
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "detector": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)

    final_path = os.path.join(cfg.ckpt_dir, "stego_adv_final.pt")
    torch.save(
        {
            "epoch": cfg.epochs,
            "cfg": {
                **cfg.__dict__,
                "ch": cfg.ch.__dict__,
                "gamma": gamma_for_epoch(cfg.epochs - 1, cfg),
            },
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "detector": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        final_path,
    )
    print("Saved:", final_path)
    print(f"Done. Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
