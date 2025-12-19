# models/encoder.py
from __future__ import annotations
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, L: int, hidden: int = 64, eps: float = 0.20):
        super().__init__()
        self.L = L
        self.eps = float(eps)
        in_ch = 3 + L

        # ВАЖНО: без nn.Tanh() здесь
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )

    def forward(self, x: torch.Tensor, m_bits: torch.Tensor, return_delta: bool = False):
        B, _, H, W = x.shape

        # лучше: {0,1} -> {-1,+1}
        m = m_bits * 2.0 - 1.0
        m_map = m.view(B, self.L, 1, 1).expand(B, self.L, H, W)

        inp = torch.cat([x, m_map], dim=1)

        raw_delta = self.net(inp)
        delta = torch.tanh(raw_delta) * self.eps

        x_stego = torch.clamp(x + delta, -1, 1)

        if return_delta:
            return x_stego, delta
        return x_stego
