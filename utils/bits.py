from __future__ import annotations

from typing import List, Sequence, Union
import torch


def _digit_to_bcd4(d: int) -> List[int]:
    return [(d >> 3) & 1, (d >> 2) & 1, (d >> 1) & 1, d & 1]


def _bcd4_to_digit(bits4: Sequence[int]) -> int:
    v = (int(bits4[0]) << 3) | (int(bits4[1]) << 2) | (int(bits4[2]) << 1) | int(bits4[3])
    # если вышло 10..15 — зажмём для демо
    if v > 9:
        v = v % 10
    return v


def digits_to_bits(texts: List[str], *, n_digits: int) -> torch.Tensor:
    """
    BCD: каждая цифра -> 4 бита. L = n_digits*4.
    Возвращает float32 {0,1} [B,L]
    """
    L = n_digits * 4
    out = torch.zeros((len(texts), L), dtype=torch.float32)
    for i, s in enumerate(texts):
        s = s.strip()
        # нормализуем длину
        if len(s) < n_digits:
            s = s.zfill(n_digits)
        else:
            s = s[:n_digits]

        bits: List[int] = []
        for ch in s:
            d = ord(ch) - ord("0")
            if d < 0 or d > 9:
                d = 0
            bits.extend(_digit_to_bcd4(d))

        out[i] = torch.tensor(bits, dtype=torch.float32)
    return out


def logits_to_bits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits [B,L] -> bits {0,1} float32 [B,L]
    """
    return (logits >= 0).float()


def bits_to_digits(bits: Union[torch.Tensor, List[int]], *, n_digits: int) -> str:
    """
    bits length L=n_digits*4 -> digit string length n_digits
    """
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().view(-1).tolist()

    bits = [1 if float(b) >= 0.5 else 0 for b in bits]
    chars = []
    for i in range(n_digits):
        chunk = bits[i * 4: (i + 1) * 4]
        chars.append(str(_bcd4_to_digit(chunk)))
    return "".join(chars)
