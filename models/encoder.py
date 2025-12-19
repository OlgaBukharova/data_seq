from __future__ import annotations
import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    """
    Encoder: X [B,3,32,32] + bits [B,L] -> X' in [-1,1]
    Делает delta = tanh(head(...))*eps, затем x' = clamp(x+delta).
    """

    def __init__(self, L: int, hidden: int = 96, eps: float = 0.25):
        super().__init__()
        self.L = int(L)
        self.eps = float(eps)

        in_ch = 3 + self.L

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            _ResBlock(hidden),
            _ResBlock(hidden),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            _ResBlock(hidden),
        )
        self.head = nn.Conv2d(hidden, 3, 1)

        # стабильный старт: начинаем с near-zero delta
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, m_bits: torch.Tensor, return_delta: bool = False):
        B, _, H, W = x.shape

        # {0,1} -> {-1,+1}
        m = m_bits * 2.0 - 1.0
        m_map = m.view(B, self.L, 1, 1).expand(B, self.L, H, W)

        h = self.stem(torch.cat([x, m_map], dim=1))
        h = self.mid(h)
        raw = self.head(h)

        delta = torch.tanh(raw) * self.eps
        x_stego = torch.clamp(x + delta, -1, 1)

        if return_delta:
            return x_stego, delta
        return x_stego
