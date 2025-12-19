# utils/lsb.py
from __future__ import annotations

import torch


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    x: float tensor in [-1,1], shape [B,3,H,W]
    -> uint8 tensor in [0,255]
    """
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    return (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)


def _to_float_minus1_1(xu8: torch.Tensor) -> torch.Tensor:
    """
    xu8: uint8 tensor [B,3,H,W] in [0,255]
    -> float tensor [-1,1]
    """
    x01 = xu8.to(torch.float32) / 255.0
    return x01 * 2.0 - 1.0


def embed_bits_lsb(x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """
    Embed bits into LSB of the BLUE channel (channel index 2) raster order.

    x: float [-1,1], [B,3,H,W]
    bits: float/bool/int {0,1}, [B, L]
    returns: stego float [-1,1], [B,3,H,W]
    """
    assert x.dim() == 4 and x.size(1) == 3
    B, _, H, W = x.shape
    L = bits.size(1)
    assert bits.size(0) == B
    assert L <= H * W, "LSB baseline uses 1 bit per pixel (blue channel). Increase capacity if needed."

    xu8 = _to_uint8(x).clone()
    b = xu8[:, 2, :, :].reshape(B, H * W)  # blue channel flat

    bits_i = (bits > 0.5).to(torch.uint8)  # [B,L] 0/1

    # clear LSB and set
    b[:, :L] = (b[:, :L] & 0xFE) | bits_i

    xu8[:, 2, :, :] = b.reshape(B, H, W)
    return _to_float_minus1_1(xu8)


def extract_bits_lsb(x_stego: torch.Tensor, L: int) -> torch.Tensor:
    """
    Extract L bits from LSB of the BLUE channel (channel index 2) raster order.

    x_stego: float [-1,1], [B,3,H,W]
    returns: float bits {0,1}, [B,L]
    """
    assert x_stego.dim() == 4 and x_stego.size(1) == 3
    B, _, H, W = x_stego.shape
    assert L <= H * W, "Not enough pixels to extract L bits."

    xu8 = _to_uint8(x_stego)
    b = xu8[:, 2, :, :].reshape(B, H * W)
    bits = (b[:, :L] & 0x01).to(torch.float32)
    return bits
