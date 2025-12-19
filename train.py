# train.py
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.encoder import Encoder
from models.decoder import Decoder
from utils.payload import make_random_string_batch
from utils.metrics import psnr, ber_from_logits
from utils.bits import bits_to_string, threshold_logits_to_bits


@dataclass
class TrainConfig:
    # Message / data
    L: int = 256                 # message bits
    n_chars: int = 8             # for demo strings (fits into L if encoding is truncating/padding)
    batch_size: int = 128
    num_workers: int = 2

    # Optimization
    epochs: int = 15
    lr: float = 3e-4

    # Loss weights (curriculum)
    alpha_img_warmup: float = 0.1
    beta_msg_warmup: float = 10.0
    warmup_epochs: int = 3

    alpha_img_main: float = 1.0
    beta_msg_main: float = 2.0

    lambda_delta: float = 0.02   # regularize magnitude of delta (encoder change)

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
    opt: torch.optim.Optimizer,
    epoch_idx: int,
    cfg: TrainConfig,
) -> None:
    torch.save(
        {
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch_idx": epoch_idx,
            "cfg": cfg.__dict__,
        },
        path,
    )


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
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # Models
    enc = Encoder(L=cfg.L, hidden=96).to(device)
    dec = Decoder(L=cfg.L, hidden=128).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr)

    loss_img_fn = nn.MSELoss()
    loss_msg_fn = nn.BCEWithLogitsLoss()

    enc.train()
    dec.train()

    global_step = 0
    start_time = time.time()

    try:
        # epoch_idx is 0-based here (important for curriculum)
        for epoch_idx in range(cfg.epochs):
            # ---- curriculum weights (MUST be inside epoch loop) ----
            if epoch_idx < cfg.warmup_epochs:
                alpha = cfg.alpha_img_warmup
                beta = cfg.beta_msg_warmup
            else:
                alpha = cfg.alpha_img_main
                beta = cfg.beta_msg_main

            running_psnr = 0.0
            running_ber = 0.0
            running_acc = 0.0
            running_loss = 0.0
            n_steps = 0

            for x, _ in train_loader:
                global_step += 1
                n_steps += 1

                x = x.to(device, non_blocking=True)  # [B,3,32,32] in [-1,1]

                # Message bits: either random bits OR random strings -> bits.
                # Strings are useful for demo printing.
                m_bits, texts = make_random_string_batch(
                    batch_size=x.size(0),
                    n_chars=cfg.n_chars,
                    L=cfg.L,
                    device=device,
                )  # m_bits: [B,L] float {0,1}

                # Forward
                x_stego, delta = enc(x, m_bits, return_delta=True)
                logits = dec(x_stego)  # logits [B,L] (NO sigmoid inside decoder)

                # Losses
                Limg = loss_img_fn(x_stego, x)
                Lmsg = loss_msg_fn(logits, m_bits)
                Ldelta = delta.abs().mean()

                loss = alpha * Limg + beta * Lmsg + cfg.lambda_delta * Ldelta

                # Backprop
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                # Metrics
                with torch.no_grad():
                    batch_psnr = psnr(x, x_stego)
                    batch_ber = ber_from_logits(logits, m_bits)

                    # bit accuracy using logits threshold at 0 (matches BCEWithLogitsLoss)
                    bit_acc = ((logits > 0).float() == m_bits).float().mean().item()

                    running_psnr += batch_psnr
                    running_ber += batch_ber
                    running_acc += bit_acc
                    running_loss += loss.item()

                # Logging
                if n_steps % cfg.log_every_steps == 0:
                    avg_psnr = running_psnr / n_steps
                    avg_ber = running_ber / n_steps
                    avg_acc = running_acc / n_steps
                    avg_loss = running_loss / n_steps

                    with torch.no_grad():
                        demo_bits = threshold_logits_to_bits(logits[0]).float()
                        demo_ber = (demo_bits != m_bits[0]).float().mean().item()
                        decoded = bits_to_string(demo_bits)

                        logits_mean = logits.mean().item()
                        logits_std = logits.std().item()

                    print(
                        f"[epoch {epoch_idx+1}/{cfg.epochs} step {n_steps}] "
                        f"loss={avg_loss:.4f} PSNR={avg_psnr:.2f} "
                        f"BER={avg_ber:.4f} acc={avg_acc:.4f} "
                        f"logits(mean/std)={logits_mean:.3f}/{logits_std:.3f} "
                        f"(alpha={alpha:g}, beta={beta:g})"
                    )
                    print(f"  demo BER     : {demo_ber:.4f}")
                    print(f"  demo decoded : {decoded!r}")
                    if texts is not None:
                        print(f"  demo target  : {texts[0]!r}")

            # End of epoch
            avg_psnr = running_psnr / max(1, n_steps)
            avg_ber = running_ber / max(1, n_steps)
            avg_acc = running_acc / max(1, n_steps)
            avg_loss = running_loss / max(1, n_steps)

            print(
                f"Epoch {epoch_idx+1}: loss={avg_loss:.4f} PSNR={avg_psnr:.2f} "
                f"BER={avg_ber:.4f} acc={avg_acc:.4f}"
            )

            # Save checkpoint
            if (epoch_idx + 1) % cfg.save_every_epochs == 0:
                ckpt_path = os.path.join(cfg.save_dir, f"stego_ed_epoch{epoch_idx+1}.pt")
                save_checkpoint(
                    ckpt_path,
                    enc=enc,
                    dec=dec,
                    opt=opt,
                    epoch_idx=epoch_idx,
                    cfg=cfg,
                )
                print("Saved:", ckpt_path)

        # Final save (convenience)
        final_path = os.path.join(cfg.save_dir, "stego_ed_final.pt")
        save_checkpoint(
            final_path,
            enc=enc,
            dec=dec,
            opt=opt,
            epoch_idx=cfg.epochs - 1,
            cfg=cfg,
        )
        print("Saved:", final_path)

    except KeyboardInterrupt:
        # Save emergency checkpoint
        print("\nKeyboardInterrupt: saving emergency checkpoint...")
        emergency_path = os.path.join(cfg.save_dir, "stego_ed_interrupt.pt")
        save_checkpoint(
            emergency_path,
            enc=enc,
            dec=dec,
            opt=opt,
            epoch_idx=epoch_idx if "epoch_idx" in locals() else -1,
            cfg=cfg,
        )
        print("Saved:", emergency_path)

    finally:
        elapsed = time.time() - start_time
        print(f"Done. Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
