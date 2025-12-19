from __future__ import annotations

from dataclasses import dataclass, field

import argparse
import os
import random
from dataclasses import dataclass

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator

from utils.bits import digits_to_bits, logits_to_bits, bits_to_digits
from utils.metrics import psnr, ber_from_logits
from utils.channel import ChannelCfg, apply_channel
from utils.vis import to_01, diff_amplified


@dataclass
class DemoCfg:
    # Payload
    n_digits: int = 8
    L: int = 32

    # Default checkpoints (можешь поменять аргументами CLI)
    ckpt_pre_adv: str = "checkpoints/stego_ed_channel_epoch10.pt"   # до adversarial (устойчивый к каналу)
    ckpt_post_adv: str = "checkpoints/stego_adv_epoch10.pt"         # после adversarial (лучший)
    ckpt_detector: str = "checkpoints/detector_final.pt"

    # Encoder eps at inference (лучше совпадать с обучением)
    eps: float = 0.20

    # Channel used in demo
    ch: ChannelCfg = field(default_factory=lambda: ChannelCfg(
        p_noise=0.35, noise_std=0.02,
        p_resize=0.20, resize_min=0.80, resize_max=1.00,
        p_crop=0.15, crop_min=0.85, crop_max=1.00,
    ))

    # Visual
    diff_amp: float = 25.0


def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x


def save_tensor_image(x_01: torch.Tensor, path: str) -> None:
    """
    x_01: [1,3,H,W] in [0,1]
    """
    x = x_01.squeeze(0).detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(x)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


@torch.no_grad()
def detector_prob(D: Discriminator, x: torch.Tensor) -> float:
    logit = D(x).float().item()
    p = float(torch.sigmoid(torch.tensor(logit)).item())
    return p


@torch.no_grad()
def run_one(
    tag: str,
    x_cover: torch.Tensor,
    digits: str,
    enc: Encoder,
    dec: Decoder,
    D: Discriminator,
    cfg: DemoCfg,
    outdir: str,
):
    device = x_cover.device
    bits = digits_to_bits([digits], n_digits=cfg.n_digits).to(device)  # [1,32]

    # Encode
    x_stego = enc(x_cover, bits)  # [-1,1]

    # Channel distortions (for robustness demo)
    x_noisy = apply_channel(x_stego, cfg.ch)

    # Decode
    logits = dec(x_noisy)
    ber, acc = ber_from_logits(logits, bits)

    pred_bits = logits_to_bits(logits)[0]
    pred_digits = bits_to_digits(pred_bits, n_digits=cfg.n_digits)

    # PSNR relative to cover (use stego, not noisy)
    p_psnr = psnr(x_stego, x_cover)

    # Detector probabilities
    p_clean = detector_prob(D, x_cover)
    p_stego = detector_prob(D, x_stego)
    p_noisy = detector_prob(D, x_noisy)

    # Save images
    cover_01 = to_01(x_cover)
    stego_01 = to_01(x_stego)
    noisy_01 = to_01(x_noisy)
    diff_01 = diff_amplified(x_stego, x_cover, amp=cfg.diff_amp)

    save_tensor_image(cover_01, os.path.join(outdir, f"{tag}_cover.png"))
    save_tensor_image(stego_01, os.path.join(outdir, f"{tag}_stego.png"))
    save_tensor_image(noisy_01, os.path.join(outdir, f"{tag}_noisy.png"))
    save_tensor_image(diff_01, os.path.join(outdir, f"{tag}_diff_x{int(cfg.diff_amp)}.png"))

    # Print report
    print(f"\n=== {tag.upper()} ===")
    print(f"target digits : {digits}")
    print(f"pred digits   : {pred_digits}")
    print(f"BER={ber:.4f}  acc_bits={acc:.4f}  PSNR(stego,cover)={p_psnr:.2f} dB")
    print(f"Detector P(stego): clean={p_clean:.4f} stego={p_stego:.4f} noisy={p_noisy:.4f}")
    print(f"Saved to: {outdir}/{tag}_*.png")


def load_ed_from_ckpt(ckpt_path: str, device: torch.device, eps: float) -> tuple[Encoder, Decoder]:
    ckpt = torch.load(ckpt_path, map_location=device)
    enc = Encoder(L=32, hidden=96, eps=eps).to(device)
    dec = Decoder(L=32, hidden=160).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval()
    dec.eval()
    return enc, dec


def load_detector(ckpt_path: str, device: torch.device) -> Discriminator:
    ckpt = torch.load(ckpt_path, map_location=device)
    D = Discriminator(hidden=64).to(device)
    D.load_state_dict(ckpt["detector"])
    D.eval()
    return D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to cover image")
    p.add_argument("--digits", type=str, default="", help="8 digits (0-9). If empty -> random")
    p.add_argument("--outdir", type=str, default="demo_out", help="Output directory")

    p.add_argument("--ckpt_pre", type=str, default="", help="Checkpoint BEFORE adversarial (channel-trained)")
    p.add_argument("--ckpt_post", type=str, default="", help="Checkpoint AFTER adversarial")
    p.add_argument("--ckpt_det", type=str, default="", help="Detector checkpoint")

    p.add_argument("--eps", type=float, default=0.20, help="Encoder eps during demo")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 means no fixed seed)")
    p.add_argument("--diff_amp", type=float, default=25.0, help="Amplification for diff visualization")
    args = p.parse_args()

    if args.seed != 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg = DemoCfg()
    cfg.eps = float(args.eps)
    cfg.diff_amp = float(args.diff_amp)

    if args.ckpt_pre:
        cfg.ckpt_pre_adv = args.ckpt_pre
    if args.ckpt_post:
        cfg.ckpt_post_adv = args.ckpt_post
    if args.ckpt_det:
        cfg.ckpt_detector = args.ckpt_det

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.outdir, exist_ok=True)

    # digits
    if args.digits.strip():
        digits = args.digits.strip()
    else:
        digits = "".join(str(random.randint(0, 9)) for _ in range(cfg.n_digits))

    # Load cover
    x_cover = load_image_as_tensor(args.image, device)

    # Load models
    enc_pre, dec_pre = load_ed_from_ckpt(cfg.ckpt_pre_adv, device, eps=cfg.eps)
    enc_post, dec_post = load_ed_from_ckpt(cfg.ckpt_post_adv, device, eps=cfg.eps)
    D = load_detector(cfg.ckpt_detector, device)

    print("Using ckpt_pre :", cfg.ckpt_pre_adv)
    print("Using ckpt_post:", cfg.ckpt_post_adv)
    print("Using detector :", cfg.ckpt_detector)

    # Run both pipelines
    run_one("pre_adv", x_cover, digits, enc_pre, dec_pre, D, cfg, args.outdir)
    run_one("post_adv", x_cover, digits, enc_post, dec_post, D, cfg, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
