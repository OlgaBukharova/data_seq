from __future__ import annotations

import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.encoder import Encoder
from utils.bits import digits_to_bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--digits", type=str, required=True, help="Digits string, len<=8 (e.g. 12345678)")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--out", type=str, default="stego.png", help="Output stego image")
    p.add_argument("--eps", type=float, default=0.25, help="Override eps for encoding")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_digits = 8
    bits = digits_to_bits([args.digits], n_digits=n_digits).to(device)  # [1,32]

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    enc = Encoder(L=32, hidden=96, eps=args.eps).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    with torch.no_grad():
        x_stego = enc(x, bits)

    # save back to [0,1]
    x_out = (x_stego.clamp(-1, 1) + 1) / 2.0
    out_img = transforms.ToPILImage()(x_out.squeeze(0).cpu())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_img.save(args.out)

    print("Saved stego image:", args.out)


if __name__ == "__main__":
    main()
