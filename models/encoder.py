# models/encoder.py
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, L: int = 256, hidden: int = 96, eps: float = 0.20, gain: float = 10.0):
        """
        L: message bits length
        eps: max embedding strength (after tanh scaling)
        gain: amplifies raw_delta before tanh to avoid tiny deltas
        """
        super().__init__()
        self.L = L
        self.eps = float(eps)
        self.gain = float(gain)

        in_ch = 3 + L

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, m_bits: torch.Tensor, return_delta: bool = False):
        """
        x:      [B,3,H,W] in [-1,1]
        m_bits: [B,L] float {0,1}
        """
        B, _, H, W = x.shape

        # IMPORTANT: map bits {0,1} -> {-1,+1} so message has zero mean
        m = m_bits * 2.0 - 1.0  # [B,L] in {-1,+1}
        m_map = m.view(B, self.L, 1, 1).expand(B, self.L, H, W)

        inp = torch.cat([x, m_map], dim=1)

        raw_delta = self.net(inp)                       # [B,3,H,W]
        delta = torch.tanh(raw_delta * self.gain) * self.eps

        x_stego = torch.clamp(x + delta, -1, 1)

        if return_delta:
            return x_stego, delta
        return x_stego
