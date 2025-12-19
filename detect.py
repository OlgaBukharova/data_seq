# detect.py
from __future__ import annotations

import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.discriminator import Discriminator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to image (clean or stego)")
    p.add_argument("--ckpt", type=str, required=True, help="Detector checkpoint (e.g. checkpoints/detector_final.pt)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    D = Discriminator(hidden=64).to(device)
    D.load_state_dict(ckpt["detector"])
    D.eval()

    with torch.no_grad():
        logit = D(x).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

    print(f"stego probability: {prob:.4f} (logit={logit:.3f})")


if __name__ == "__main__":
    main()
