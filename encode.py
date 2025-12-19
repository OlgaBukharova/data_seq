# encode.py
from __future__ import annotations

import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.encoder import Encoder
from utils.bits import digits_to_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--digits", type=str, required=True, help="String of digits (len <= 8)")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--out", type=str, default="stego.png")
    parser.add_argument("--eps", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- payload ---
    n_digits = 8
    bits = digits_to_bits([args.digits], n_digits=n_digits).to(device)  # [1,32]

    # --- image ---
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,32,32]

    # --- model ---
    ckpt = torch.load(args.ckpt, map_location=device)
    enc = Encoder(L=32, hidden=96, eps=args.eps).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    with torch.no_grad():
        x_stego = enc(x, bits)

    # --- save ---
    x_out = (x_stego.clamp(-1, 1) + 1) / 2.0
    out_img = transforms.ToPILImage()(x_out.squeeze(0).cpu())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_img.save(args.out)

    print("Saved stego image to:", args.out)


if __name__ == "__main__":
    main()
