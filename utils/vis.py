# utils/vis.py
from __future__ import annotations

import torch


def to_01(x: torch.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,1]
    """
    return (x.clamp(-1, 1) + 1.0) / 2.0


def diff_amplified(x_stego: torch.Tensor, x_cover: torch.Tensor, amp: float = 20.0) -> torch.Tensor:
    """
    Make an amplified visualization of |stego-cover|.

    Returns: [B,3,H,W] in [0,1]
    """
    d = (x_stego - x_cover).abs() * float(amp)  # amplify
    d = d.clamp(0, 1)
    return d
