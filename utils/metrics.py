from __future__ import annotations
import torch


@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> float:
    """
    PSNR для тензоров в диапазоне [-1,1].
    """
    mse = torch.mean((x_hat - x) ** 2).clamp_min(eps)
    peak = 2.0  # -1..1
    val = 10.0 * torch.log10((peak * peak) / mse)
    return float(val.item())


@torch.no_grad()
def ber_from_logits(logits: torch.Tensor, target_bits: torch.Tensor):
    pred = (logits >= 0).float()
    total = target_bits.numel()
    wrong = (pred != target_bits).sum().item()
    ber = wrong / total
    acc = 1.0 - ber
    return float(ber), float(acc)
