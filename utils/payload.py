# utils/payload.py
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from utils.bits import digits_to_bits


def random_digit_string(n_digits: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n_digits))


def make_random_digit_batch(
    batch_size: int,
    *,
    n_digits: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns:
      bits: float32 tensor [B, L] with {0,1}, where L = n_digits*4 (BCD)
      texts: list[str] of length B
    """
    texts: List[str] = [random_digit_string(n_digits) for _ in range(batch_size)]
    bits = digits_to_bits(texts, n_digits=n_digits)  # [B, L] float32 on CPU
    if device is not None:
        bits = bits.to(device)
    return bits, texts
