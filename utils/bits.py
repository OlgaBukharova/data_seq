# Отвечает за:
# преобразование строки → биты
# биты → строка

# utils/bits.py
from __future__ import annotations

from typing import Iterable, List, Tuple, Union
import torch


# ----------------------------
# Helpers: bytes <-> bits
# ----------------------------

def _bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to a list of bits (0/1), big-endian per byte."""
    bits: List[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Convert an iterable of bits (0/1) length multiple of 8 to bytes."""
    bits_list = list(int(x) for x in bits)
    if len(bits_list) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8 to convert to bytes.")

    out = bytearray()
    for i in range(0, len(bits_list), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits_list[i + j] & 1)
        out.append(byte)
    return bytes(out)


def _u16_to_bits(n: int) -> List[int]:
    """Encode unsigned 16-bit int to 16 bits."""
    if not (0 <= n <= 0xFFFF):
        raise ValueError("u16 out of range.")
    return [(n >> i) & 1 for i in range(15, -1, -1)]


def _bits_to_u16(bits16: Iterable[int]) -> int:
    b = list(int(x) for x in bits16)
    if len(b) != 16:
        raise ValueError("Need exactly 16 bits for u16.")
    n = 0
    for bit in b:
        n = (n << 1) | (bit & 1)
    return n


# ----------------------------
# Public API: string <-> bits
# ----------------------------

def string_to_bits(text: str, L: int, *, encoding: str = "utf-8") -> torch.Tensor:
    """
    Encode text into a fixed-length bit vector (torch.float32) of shape [L].

    Format inside L bits:
      [0:16)   - 16-bit length (in BYTES) of payload
      [16:16+8*len] - payload bytes (utf-8)
      remaining bits - zero padding

    If payload is too long, it is truncated to fit.
    """
    if L < 16:
        raise ValueError("L must be >= 16 to store length header.")

    payload_bytes = text.encode(encoding, errors="strict")

    max_payload_bits = L - 16
    max_payload_bytes = max_payload_bits // 8  # floor
    payload_bytes = payload_bytes[:max_payload_bytes]

    length_bits = _u16_to_bits(len(payload_bytes))
    payload_bits = _bytes_to_bits(payload_bytes)

    bits = length_bits + payload_bits
    if len(bits) < L:
        bits.extend([0] * (L - len(bits)))
    else:
        bits = bits[:L]

    return torch.tensor(bits, dtype=torch.float32)


def bits_to_string(bits: Union[torch.Tensor, Iterable[int]], *, encoding: str = "utf-8") -> str:
    """
    Decode a bit vector produced by string_to_bits back into a string.

    Accepts:
      - torch.Tensor shape [L] or [B, L] (in the latter case decodes first row)
      - any iterable of ints

    NOTE: For model outputs you should threshold logits:
        bits = (torch.sigmoid(logits) > 0.5).int()
    """
    if isinstance(bits, torch.Tensor):
        if bits.dim() == 2:
            bits = bits[0]
        bits_list = bits.detach().cpu().flatten().tolist()
        # Convert floats to 0/1 safely:
        bits_list = [1 if float(x) >= 0.5 else 0 for x in bits_list]
    else:
        bits_list = [int(x) & 1 for x in bits]

    if len(bits_list) < 16:
        return ""

    n_bytes = _bits_to_u16(bits_list[:16])
    payload_bits = bits_list[16:16 + 8 * n_bytes]

    # If not enough bits, decode what we have (best-effort).
    if len(payload_bits) < 8 * n_bytes:
        # pad with zeros to full bytes
        missing = (8 * n_bytes) - len(payload_bits)
        payload_bits = payload_bits + [0] * missing

    payload_bytes = _bits_to_bytes(payload_bits)
    return payload_bytes.decode(encoding, errors="replace")


def make_batch_from_strings(texts: List[str], L: int, *, encoding: str = "utf-8") -> torch.Tensor:
    """
    Encode a list of strings into a tensor of shape [B, L] float32.
    """
    return torch.stack([string_to_bits(t, L, encoding=encoding) for t in texts], dim=0)


def threshold_logits_to_bits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert decoder logits (shape [B, L] or [L]) to {0,1} bits.
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).to(torch.int32)
