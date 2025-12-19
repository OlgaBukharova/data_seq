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
from utils.payload import make_random_digit_batch
from utils.metrics import psnr, ber_from_logits
from utils.bits import logits_to_bits, bits_to_digits


@dataclass
class Cfg:
    # Payload: 8 digits -> 8 * 4 bits = 32 bits (BCD)
    n_digits: int = 8
    L: int = 32

    batch_size: int = 256
    epochs: int = 12
    lr: float = 2e-4

    # Curriculum:
    # сначала "учим читать сообщение" (beta высокое, eps выше, alpha ниже),
    # потом "делаем незаметнее" (alpha выше, eps ниже, beta умеренное)
    eps_warm: float = 0.55
    eps_main: float = 0.25

    alpha_warm: float = 0.05   # вес MSE
    beta_warm: float = 30.0    # вес сообщения
    lam_warm: float = 0.0      # штраф на |delta| (выключен на разогреве)

    alpha_main: float = 0.70
    beta_main: float = 10.0
    lam_main: float = 0.001

    warm_epochs: int = 3

    # data
    num_workers: int = 2
    pin_memory: bool = True

    # logging
    print_every: int = 200
    ckpt_dir: str = "checkpoints"


def get_stage_params(epoch: int, cfg: Cfg):
    if epoch < cfg.warm_epochs:
        return cfg.eps_warm, cfg.alpha_warm, cfg.beta_warm, cfg.lam_warm
    return cfg.eps_main, cfg.alpha_main, cfg.beta_main, cfg.lam_main


@torch.no_grad()
def demo_batch(enc: Encoder, dec: Decoder, x: torch.Tensor, cfg: Cfg, device: torch.device):
    bits, texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)  # [B,L], list[str]
    x_stego, _delta = enc(x, bits, return_delta=True)
    logits = dec(x_stego)
    pred_bits = logits_to_bits(logits)
    pred_text = bits_to_digits(pred_bits[0].detach().cpu(), n_digits=cfg.n_digits)
    return texts[0], pred_text, x_stego


def main():
    cfg = Cfg()
    assert cfg.L == cfg.n_digits * 4, "Для BCD: L должно быть n_digits * 4"

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

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        eps, alpha, beta, lam = get_stage_params(epoch - 1, cfg)
        enc.set_eps(eps)

        enc.train()
        dec.train()

        running_loss = 0.0
        running_psnr = 0.0
        running_ber = 0.0
        running_acc = 0.0

        for step, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)

            m_bits, _texts = make_random_digit_batch(x.size(0), n_digits=cfg.n_digits, device=device)  # [B,L] float {0,1}

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_stego, delta = enc(x, m_bits, return_delta=True)
                logits = dec(x_stego)

                loss_img = mse(x_stego, x)
                loss_msg = bce(logits, m_bits)

                loss_delta = delta.abs().mean()  # L1 по возмущению
                loss = alpha * loss_img + beta * loss_msg + lam * loss_delta

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                _psnr = psnr(x_stego, x)
                _ber, _acc = ber_from_logits(logits, m_bits)

            running_loss += float(loss.item())
            running_psnr += float(_psnr)
            running_ber += float(_ber)
            running_acc += float(_acc)

            if step % cfg.print_every == 0:
                avg_loss = running_loss / step
                avg_psnr = running_psnr / step
                avg_ber = running_ber / step
                avg_acc = running_acc / step

                logits_mean = float(logits.mean().item())
                logits_std = float(logits.std().item())
                delta_mean = float(delta.abs().mean().item())
                m_mean = float(m_bits.mean().item())

                print(
                    f"[epoch {epoch}/{cfg.epochs} step {step}] "
                    f"l
