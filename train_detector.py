# train_detector.py
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
from models.discriminator import Discriminator
from utils.payload import make_random_digit_batch


@dataclass
class Cfg:
    # data / payload
    n_digits: int = 8
    L: int = 32
    batch_size: int = 256
    epochs: int = 8
    lr: float = 2e-4

    # which encoder checkpoint to generate stego
    # рекомендую: checkpoints/stego_ed_channel_epoch10.pt или stego_ed_channel_final.pt
    encoder_ckpt: str = "checkpoints/stego_ed_channel_epoch10.pt"

    # how strong encoder perturbation should be during stego generation
    # (обычно лучше фиксировать eps таким же, как при обучении)
    eps: float = 0.20

    # train
    num_workers: int = 2
    pin_memory: bool = True
    print_every: int = 200
    ckpt_dir: str = "checkpoints"


@torch.no_grad()
def make_stego(enc: Encoder, x: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    m_bits, _txt = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=x.device)
    x_stego = enc(x, m_bits)
    return x_stego


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor):
    """
    logits: [B], y: [B] float {0,1}
    """
    pred = (logits >= 0).float()
    acc = (pred == y).float().mean().item()

    # simple TPR/TNR for sanity
    y1 = (y == 1)
    y0 = (y == 0)
    tpr = (pred[y1] == 1).float().mean().item() if y1.any() else 0.0
    tnr = (pred[y0] == 0).float().mean().item() if y0.any() else 0.0
    return float(acc), float(tpr), float(tnr)


def set_requires_grad(model: nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


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

    # Encoder (frozen) for stego generation
    enc = Encoder(L=cfg.L, hidden=96, eps=cfg.eps).to(device)
    ckpt = torch.load(cfg.encoder_ckpt, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    set_requires_grad(enc, False)
    print("Loaded encoder ckpt:", cfg.encoder_ckpt)

    # Detector
    D = Discriminator(hidden=64).to(device)
    opt = torch.optim.AdamW(D.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        D.train()

        run_loss = 0.0
        run_acc = 0.0
        run_tpr = 0.0
        run_tnr = 0.0

        for step, (x, _yclass) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)

            # split batch in half: clean vs stego
            B = x.size(0)
            half = B // 2
            x_clean = x[:half]
            x_for_stego = x[half:]

            with torch.no_grad():
                x_stego = make_stego(enc, x_for_stego, cfg)

            x_mix = torch.cat([x_clean, x_stego], dim=0)

            y = torch.cat([
                torch.zeros(half, device=device),
                torch.ones(B - half, device=device),
            ], dim=0)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = D(x_mix)  # [B]
                loss = bce(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                acc, tpr, tnr = metrics_from_logits(logits, y)

            run_loss += float(loss.item())
            run_acc += acc
            run_tpr += tpr
            run_tnr += tnr

            if step % cfg.print_every == 0:
                avg_loss = run_loss / step
                avg_acc = run_acc / step
                avg_tpr = run_tpr / step
                avg_tnr = run_tnr / step
                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f} TPR={avg_tpr:.4f} TNR={avg_tnr:.4f}"
                )

        steps = len(train_loader)
        ep_loss = run_loss / steps
        ep_acc = run_acc / steps
        ep_tpr = run_tpr / steps
        ep_tnr = run_tnr / steps

        print(f"Epoch {epoch}: loss={ep_loss:.4f} acc={ep_acc:.4f} TPR={ep_tpr:.4f} TNR={ep_tnr:.4f}")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"detector_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "cfg": cfg.__dict__,
                "detector": D.state_dict(),
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)

    final_path = os.path.join(cfg.ckpt_dir, "detector_final.pt")
    torch.save(
        {
            "epoch": cfg.epochs,
            "cfg": cfg.__dict__,
            "detector": D.state_dict(),
        },
        final_path,
    )
    print("Saved:", final_path)
    print(f"Done. Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
