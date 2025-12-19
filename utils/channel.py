# utils/channel.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ChannelCfg:
    # probabilities
    p_noise: float = 0.50
    p_resize: float = 0.30
    p_crop: float = 0.30

    # noise
    noise_std: float = 0.03  # for x in [-1,1]

    # resize
    resize_min: float = 0.60
    resize_max: float = 1.00

    # crop (fraction of image side kept)
    crop_min: float = 0.70
    crop_max: float = 1.00


def _rand_bool(batch: int, p: float, device: torch.device) -> torch.Tensor:
    return (torch.rand(batch, device=device) < p)


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return torch.clamp(x + torch.randn_like(x) * std, -1, 1)


def random_resize(x: torch.Tensor, scale_min: float, scale_max: float) -> torch.Tensor:
    """
    Down/up-sample then back to original size. Differentiable via interpolate.
    """
    B, C, H, W = x.shape
    device = x.device
    scale = (scale_min + (scale_max - scale_min) * torch.rand(B, device=device))  # [B]
    out = x

    # do per-sample scaling in a loop (B=256 ok)
    outs = []
    for i in range(B):
        s = float(scale[i].item())
        h2 = max(2, int(round(H * s)))
        w2 = max(2, int(round(W * s)))
        xi = out[i:i+1]
        xi2 = F.interpolate(xi, size=(h2, w2), mode="bilinear", align_corners=False)
        xi3 = F.interpolate(xi2, size=(H, W), mode="bilinear", align_corners=False)
        outs.append(xi3)
    return torch.cat(outs, dim=0)


def random_crop_pad(x: torch.Tensor, crop_min: float, crop_max: float) -> torch.Tensor:
    """
    Random crop then pad back to original size (zeros in [-1,1] is 0 -> mid-gray).
    """
    B, C, H, W = x.shape
    device = x.device
    frac = crop_min + (crop_max - crop_min) * torch.rand(B, device=device)

    outs = []
    for i in range(B):
        f = float(frac[i].item())
        ch = max(2, int(round(H * f)))
        cw = max(2, int(round(W * f)))

        top = int(torch.randint(0, H - ch + 1, (1,), device=device).item()) if ch < H else 0
        left = int(torch.randint(0, W - cw + 1, (1,), device=device).item()) if cw < W else 0

        xi = x[i:i+1, :, top:top+ch, left:left+cw]  # [1,C,ch,cw]

        # pad to H,W with symmetric-ish random placement
        pad_top = int(torch.randint(0, H - ch + 1, (1,), device=device).item()) if ch < H else 0
        pad_left = int(torch.randint(0, W - cw + 1, (1,), device=device).item()) if cw < W else 0
        pad_bottom = H - ch - pad_top
        pad_right = W - cw - pad_left

        xi = F.pad(xi, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
        outs.append(xi)

    return torch.cat(outs, dim=0)


def apply_channel(x: torch.Tensor, cfg: ChannelCfg) -> torch.Tensor:
    """
    x: [-1,1]
    Applies stochastic distortions per-sample (but implemented as batch ops where possible).
    """
    B = x.size(0)
    device = x.device
    out = x

    # noise (batch)
    mask = _rand_bool(B, cfg.p_noise, device)
    if mask.any():
        out_n = gaussian_noise(out[mask], cfg.noise_std)
        out = out.clone()
        out[mask] = out_n

    # resize (loop per sample inside)
    mask = _rand_bool(B, cfg.p_resize, device)
    if mask.any():
        out_r = random_resize(out[mask], cfg.resize_min, cfg.resize_max)
        out = out.clone()
        out[mask] = out_r

    # crop+pad (loop per sample)
    mask = _rand_bool(B, cfg.p_crop, device)
    if mask.any():
        out_c = random_crop_pad(out[mask], cfg.crop_min, cfg.crop_max)
        out = out.clone()
        out[mask] = out_c

    return torch.clamp(out, -1, 1)
