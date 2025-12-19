# compare_baselines.py
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator

from utils.bits import digits_to_bits, logits_to_bits, bits_to_digits
from utils.metrics import psnr, ber_from_logits
from utils.channel import ChannelCfg, apply_channel
from utils.lsb import embed_bits_lsb, extract_bits_lsb


@dataclass
class Cfg:
    n_digits: int = 8
    L: int = 32

    ckpt_pre: str = "checkpoints/stego_ed_channel_epoch10.pt"
    ckpt_post: str = "checkpoints/stego_adv_epoch10.pt"
    ckpt_det_neuro: str = "checkpoints/detector_final.pt"
    ckpt_det_lsb: str = "checkpoints/detector_lsb_final.pt"

    eps: float = 0.20

    ch: ChannelCfg = field(default_factory=lambda: ChannelCfg(
        p_noise=0.35, noise_std=0.02,
        p_resize=0.20, resize_min=0.80, resize_max=1.00,
        p_crop=0.15, crop_min=0.85, crop_max=1.00,
    ))


def load_image(path: str, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


def load_ed(ckpt_path: str, device: torch.device, eps: float) -> tuple[Encoder, Decoder]:
    ckpt = torch.load(ckpt_path, map_location=device)
    enc = Encoder(L=32, hidden=96, eps=eps).to(device)
    dec = Decoder(L=32, hidden=160).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    return enc, dec


def load_det(ckpt_path: str, device: torch.device) -> Discriminator:
    ckpt = torch.load(ckpt_path, map_location=device)
    D = Discriminator(hidden=64).to(device)
    D.load_state_dict(ckpt["detector"])
    D.eval()
    return D


@torch.no_grad()
def prob(D: Discriminator, x: torch.Tensor) -> float:
    logit = D(x).float().item()
    return float(torch.sigmoid(torch.tensor(logit)).item())


@torch.no_grad()
def eval_neuro(tag: str, x: torch.Tensor, digits: str, enc: Encoder, dec: Decoder, D: Discriminator, cfg: Cfg):
    bits = digits_to_bits([digits], n_digits=cfg.n_digits).to(x.device)
    x_stego = enc(x, bits)
    x_noisy = apply_channel(x_stego, cfg.ch)
    logits = dec(x_noisy)
    ber, acc = ber_from_logits(logits, bits)
    pred = bits_to_digits(logits_to_bits(logits)[0], n_digits=cfg.n_digits)
    p_psnr = psnr(x_stego, x)

    print(f"\n[{tag}]")
    print(f"target={digits} pred={pred}  BER={ber:.4f} acc_bits={acc:.4f}  PSNR={p_psnr:.2f} dB")
    print(f"P_det(clean)={prob(D,x):.4f}  P_det(stego)={prob(D,x_stego):.4f}  P_det(noisy)={prob(D,x_noisy):.4f}")


@torch.no_grad()
def eval_lsb(tag: str, x: torch.Tensor, digits: str, D: Discriminator, cfg: Cfg):
    bits = digits_to_bits([digits], n_digits=cfg.n_digits).to(x.device)
    x_stego = embed_bits_lsb(x, bits)
    x_noisy = apply_channel(x_stego, cfg.ch)

    # extract (LSB decoding) from noisy will usually break -> that's part of baseline story
    bits_hat = extract_bits_lsb(x_noisy, L=cfg.L)
    ber = float((bits_hat != bits).float().mean().item())
    acc = float(1.0 - ber)

    pred = bits_to_digits((bits_hat[0] > 0.5).to(torch.int64), n_digits=cfg.n_digits)
    p_psnr = psnr(x_stego, x)

    print(f"\n[{tag}]")
    print(f"target={digits} pred={pred}  BER={ber:.4f} acc_bits={acc:.4f}  PSNR={p_psnr:.2f} dB")
    print(f"P_det(clean)={prob(D,x):.4f}  P_det(stego)={prob(D,x_stego):.4f}  P_det(noisy)={prob(D,x_noisy):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--digits", default="", type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--ckpt_pre", default="", type=str)
    ap.add_argument("--ckpt_post", default="", type=str)
    ap.add_argument("--ckpt_det_neuro", default="", type=str)
    ap.add_argument("--ckpt_det_lsb", default="", type=str)
    args = ap.parse_args()

    if args.seed != 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg = Cfg()
    if args.ckpt_pre: cfg.ckpt_pre = args.ckpt_pre
    if args.ckpt_post: cfg.ckpt_post = args.ckpt_post
    if args.ckpt_det_neuro: cfg.ckpt_det_neuro = args.ckpt_det_neuro
    if args.ckpt_det_lsb: cfg.ckpt_det_lsb = args.ckpt_det_lsb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load_image(args.image, device)

    if args.digits.strip():
        digits = args.digits.strip()
    else:
        digits = "".join(str(random.randint(0, 9)) for _ in range(cfg.n_digits))

    print("Device:", device)
    print("Image:", os.path.abspath(args.image))
    print("Digits:", digits)

    enc_pre, dec_pre = load_ed(cfg.ckpt_pre, device, eps=cfg.eps)
    enc_post, dec_post = load_ed(cfg.ckpt_post, device, eps=cfg.eps)

    D_neuro = load_det(cfg.ckpt_det_neuro, device)
    D_lsb = load_det(cfg.ckpt_det_lsb, device)

    eval_neuro("NEURO_PRE_ADV", x, digits, enc_pre, dec_pre, D_neuro, cfg)
    eval_neuro("NEURO_POST_ADV", x, digits, enc_post, dec_post, D_neuro, cfg)
    eval_lsb("LSB_BASELINE", x, digits, D_lsb, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
