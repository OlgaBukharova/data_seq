# utils/metrics.py
from __future__ import annotations
import math
import torch


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
    """
    PSNR for tensors in [-1,1] => data_range=2.
    Returns scalar float.
    """
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ber_from_logits(logits: torch.Tensor, target_bits: torch.Tensor) -> float:
    """
    Bit Error Rate for decoder logits [B,L] vs target bits {0,1}.
    """
    pred = (logits > 0).float()
    return (pred != target_bits).float().mean().item()
