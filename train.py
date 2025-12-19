# train.py
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
from utils.metrics import psnr, ber_from_logits


@dataclass
class TrainConfig:
    # Message / data
    L: int = 128
    batch_size: int = 128
    num_workers: int = 2

    # Optimization
    epochs: int = 15
    lr: float = 3e-4

    # Two-phase training inside each batch
    # 1) update decoder only (encoder frozen) for k steps
    dec_only_steps: int = 1

    # Curriculum
    warmup_epochs: int = 4
    alpha_img_warmup: float = 0.05
    beta_msg_warmup: float = 15.0
    lambda_delta_warmup: float = 0.0

    alpha_img_main: float = 0.2
    beta_msg_main: float = 15.0
    lambda_delta_main: float = 0.0005

    # Optional later boost
    beta_boost_epoch: int = 8
    beta_msg_boost: float = 20.0

    # Channel noise (enable after decoder starts learning)
    noise_after_epoch: int = 6
    noise_std: float = 0.02

    # Logging / saving
    log_every_steps: int = 200
    save_dir: str = "checkpoints"
    save_every_epochs: int = 1

    # Repro
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    *,
    enc: nn.Module,
    dec: nn.Module,
    opt_e: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch_idx: int,
    cfg: TrainConfig,
) -> None:
    torch.save(
        {
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "opt_e": opt_e.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch_idx": epoch_idx,
            "cfg": cfg.__dict__,
        },
        path,
    )


def make_balanced_random_bits(batch_size: int, L: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, L), device=device).float()


def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def main() -> None:
    cfg = TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    # Data: CIFAR10 normalized to [-1,1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    enc = Encoder(L=cfg.L, hidden=64).to(device)
    dec = Decoder(L=cfg.L, hidden=256).to(device)

    # Separate optimizers: decoder learns faster
    opt_e = torch.optim.Adam(enc.parameters(), lr=cfg.lr)
    opt_d = torch.optim.Adam(dec.parameters(), lr=cfg.lr * 5.0)  # strong decoder push

    loss_img_fn = nn.MSELoss()
    loss_msg_fn = nn.BCEWithLogitsLoss()

    enc.train()
    dec.train()

    start_time = time.time()

    try:
        for epoch_idx in range(cfg.epochs):
            # curriculum
            if epoch_idx < cfg.warmup_epochs:
                alpha = cfg.alpha_img_warmup
                beta = cfg.beta_msg_warmup
                lambda_delta = cfg.lambda_delta_warmup
            else:
                alpha = cfg.alpha_img_main
                beta = cfg.beta_msg_main
                lambda_delta = cfg.lambda_delta_main

            if epoch_idx >= cfg.beta_boost_epoch:
                beta = cfg.beta_msg_boost

            running_psnr = 0.0
            running_ber = 0.0
            running_acc = 0.0
            running_loss = 0.0
            running_lmsg = 0.0
            running_limg = 0.0
            running_ldelta = 0.0
            n_steps = 0

            for x, _ in train_loader:
                n_steps += 1
                x = x.to(device, non_blocking=True)
                m_bits = make_balanced_random_bits(x.size(0), cfg.L, device=device)

                # -------------------------
                # (A) Decoder-only step(s)
                # -------------------------
                set_requires_grad(enc, False)
                set_requires_grad(dec, True)

                for _ in range(cfg.dec_only_steps):
                    with torch.no_grad():
                        x_stego, _ = enc(x, m_bits, return_delta=True)

                    # noise later
                    if epoch_idx >= cfg.noise_after_epoch and cfg.noise_std > 0:
                        x_stego_in = torch.clamp(x_stego + cfg.noise_std * torch.randn_like(x_stego), -1, 1)
                    else:
                        x_stego_in = x_stego

                    logits = dec(x_stego_in)
                    Lmsg_d = loss_msg_fn(logits, m_bits)

                    opt_d.zero_grad(set_to_none=True)
                    Lmsg_d.backward()
                    opt_d.step()

                # -------------------------
                # (B) Joint step (E + D)
                # -------------------------
                set_requires_grad(enc, True)
                set_requires_grad(dec, True)

                x_stego, delta = enc(x, m_bits, return_delta=True)

                if epoch_idx >= cfg.noise_after_epoch and cfg.noise_std > 0:
                    x_stego_in = torch.clamp(x_stego + cfg.noise_std * torch.randn_like(x_stego), -1, 1)
                else:
                    x_stego_in = x_stego

                logits = dec(x_stego_in)

                Limg = loss_img_fn(x_stego, x)
                Lmsg = loss_msg_fn(logits, m_bits)
                Ldelta = delta.abs().mean()

                loss = alpha * Limg + beta * Lmsg + lambda_delta * Ldelta

                opt_e.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                loss.backward()
                opt_e.step()
                opt_d.step()

                # metrics
                with torch.no_grad():
                    batch_psnr = psnr(x, x_stego)
                    batch_ber = ber_from_logits(logits, m_bits)
                    bit_acc = ((logits > 0).float() == m_bits).float().mean().item()

                    running_psnr += batch_psnr
                    running_ber += batch_ber
                    running_acc += bit_acc
                    running_loss += loss.item()
                    running_lmsg += Lmsg.item()
                    running_limg += Limg.item()
                    running_ldelta += Ldelta.item()

                if n_steps % cfg.log_every_steps == 0:
                    avg_psnr = running_psnr / n_steps
                    avg_ber = running_ber / n_steps
                    avg_acc = running_acc / n_steps
                    avg_loss = running_loss / n_steps
                    avg_lmsg = running_lmsg / n_steps
                    avg_limg = running_limg / n_steps
                    avg_ldelta = running_ldelta / n_steps

                    logits_mean = logits.mean().item()
                    logits_std = logits.std().item()
                    mb_mean = m_bits.mean().item()
                    delta_mean = delta.abs().mean().item()
                    noise_val = cfg.noise_std if epoch_idx >= cfg.noise_after_epoch else 0.0

                    print(
                        f"[epoch {epoch_idx+1}/{cfg.epochs} step {n_steps}] "
                        f"loss={avg_loss:.4f} (Lmsg={avg_lmsg:.4f} Limg={avg_limg:.6f} Ld={avg_ldelta:.6f}) "
                        f"PSNR={avg_psnr:.2f} BER={avg_ber:.4f} acc={avg_acc:.4f} "
                        f"logits(mean/std)={logits_mean:.3f}/{logits_std:.3f} "
                        f"(alpha={alpha:g}, beta={beta:g}, lambda_delta={lambda_delta:g}, noise={noise_val:g})"
                    )
                    print(f"  m_bits.mean  : {mb_mean:.3f}    |delta|.mean : {delta_mean:.6f}")

            # epoch summary
            avg_psnr = running_psnr / max(1, n_steps)
            avg_ber = running_ber / max(1, n_steps)
            avg_acc = running_acc / max(1, n_steps)
            avg_loss = running_loss / max(1, n_steps)
            avg_lmsg = running_lmsg / max(1, n_steps)
            avg_limg = running_limg / max(1, n_steps)
            avg_ldelta = running_ldelta / max(1, n_steps)

            print(
                f"Epoch {epoch_idx+1}: loss={avg_loss:.4f} (Lmsg={avg_lmsg:.4f} Limg={avg_limg:.6f} Ld={avg_ldelta:.6f}) "
                f"PSNR={avg_psnr:.2f} BER={avg_ber:.4f} acc={avg_acc:.4f}"
            )

            if (epoch_idx + 1) % cfg.save_every_epochs == 0:
                ckpt_path = os.path.join(cfg.save_dir, f"stego_ed_epoch{epoch_idx+1}.pt")
                save_checkpoint(ckpt_path, enc=enc, dec=dec, opt_e=opt_e, opt_d=opt_d, epoch_idx=epoch_idx, cfg=cfg)
                print("Saved:", ckpt_path)

        final_path = os.path.join(cfg.save_dir, "stego_ed_final.pt")
        save_checkpoint(final_path, enc=enc, dec=dec, opt_e=opt_e, opt_d=opt_d, epoch_idx=cfg.epochs - 1, cfg=cfg)
        print("Saved:", final_path)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: saving emergency checkpoint...")
        emergency_path = os.path.join(cfg.save_dir, "stego_ed_interrupt.pt")
        save_checkpoint(
            emergency_path,
            enc=enc,
            dec=dec,
            opt_e=opt_e,
            opt_d=opt_d,
            epoch_idx=epoch_idx if "epoch_idx" in locals() else -1,
            cfg=cfg,
        )
        print("Saved:", emergency_path)

    finally:
        elapsed = time.time() - start_time
        print(f"Done. Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
