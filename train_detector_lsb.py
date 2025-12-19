# train_detector_lsb.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.discriminator import Discriminator
from utils.payload import make_random_digit_batch
from utils.lsb import embed_bits_lsb


@dataclass
class Cfg:
    n_digits: int = 8
    L: int = 32

    batch_size: int = 256
    epochs: int = 5
    lr: float = 2e-4

    num_workers: int = 2
    pin_memory: bool = True
    print_every: int = 200
    ckpt_dir: str = "checkpoints"


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor):
    pred = (logits >= 0).float()
    acc = (pred == y).float().mean().item()
    y1 = (y == 1)
    y0 = (y == 0)
    tpr = (pred[y1] == 1).float().mean().item() if y1.any() else 0.0
    tnr = (pred[y0] == 0).float().mean().item() if y0.any() else 0.0
    return float(acc), float(tpr), float(tnr)


def main():
    cfg = Cfg()
    assert cfg.L == cfg.n_digits * 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

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

    D = Discriminator(hidden=64).to(device)
    opt = torch.optim.AdamW(D.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        D.train()
        run_loss = run_acc = run_tpr = run_tnr = 0.0

        for step, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)

            B = x.size(0)
            half = B // 2
            x_clean = x[:half]
            x_for_stego = x[half:]

            bits, _txt = make_random_digit_batch(x_for_stego.size(0), n_digits=cfg.n_digits, device=device)
            x_stego = embed_bits_lsb(x_for_stego, bits)

            x_mix = torch.cat([x_clean, x_stego], dim=0)
            y = torch.cat([
                torch.zeros(half, device=device),
                torch.ones(B - half, device=device),
            ], dim=0)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = D(x_mix)
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
                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"loss={run_loss/step:.4f} acc={run_acc/step:.4f} TPR={run_tpr/step:.4f} TNR={run_tnr/step:.4f}"
                )

        steps = len(train_loader)
        print(
            f"Epoch {epoch}: loss={run_loss/steps:.4f} acc={run_acc/steps:.4f} "
            f"TPR={run_tpr/steps:.4f} TNR={run_tnr/steps:.4f}"
        )

        ckpt_path = os.path.join(cfg.ckpt_dir, f"detector_lsb_epoch{epoch}.pt")
        torch.save({"epoch": epoch, "cfg": cfg.__dict__, "detector": D.state_dict()}, ckpt_path)
        print("Saved:", ckpt_path)

    final_path = os.path.join(cfg.ckpt_dir, "detector_lsb_final.pt")
    torch.save({"epoch": cfg.epochs, "cfg": cfg.__dict__, "detector": D.state_dict()}, final_path)
    print("Saved:", final_path)
    print(f"Done. Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
