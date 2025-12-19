from __future__ import annotations

import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.decoder import Decoder
from utils.bits import logits_to_bits, bits_to_digits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to stego image")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    dec = Decoder(L=32, hidden=160).to(device)
    dec.load_state_dict(ckpt["decoder"])
    dec.eval()

    with torch.no_grad():
        logits = dec(x)

    bits = logits_to_bits(logits[0])
    text = bits_to_digits(bits, n_digits=8)

    print("Decoded digits:", text)


if __name__ == "__main__":
    main()
