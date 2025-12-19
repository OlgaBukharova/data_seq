# Генерация сообщений для обучения:
# случайные строки
# или сразу случайные биты

# utils/payload.py
from __future__ import annotations

import random
import string
from typing import List, Optional

import torch

from utils.bits import make_batch_from_strings


def random_ascii_string(n_chars: int) -> str:
    alphabet = string.ascii_letters + string.digits + " _-.,;:!?@#$%&*()[]{}"
    return "".join(random.choice(alphabet) for _ in range(n_chars))


def make_random_string_batch(
    batch_size: int,
    n_chars: int,
    *,
    L: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns:
      bits: float32 tensor [B, L] with {0,1} values
    """
    texts: List[str] = [random_ascii_string(n_chars) for _ in range(batch_size)]
    bits = make_batch_from_strings(texts, L)  # [B, L] float32 on CPU
    if device is not None:
        bits = bits.to(device)
    return bits, texts
