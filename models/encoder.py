# models/encoder.py
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, L: int = 256, hidden: int = 96, eps: float = 0.10):
        """
        L: message bits length
        hidden: base hidden channels
        eps: embedding strength (max abs delta after tanh scaling)
        """
        super().__init__()
        self.L = L
        self.eps = float(eps)

        in_ch = 3 + L

        # Simple CNN that maps [B, 3+L, H, W] -> [B, 3, H, W]
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

        # Expand bits into a spatial map and concatenate
        m_map = m_bits.view(B, self.L, 1, 1).expand(B, self.L, H, W)
        inp = torch.cat([x, m_map], dim=1)

        # Predict raw delta, then constrain and scale
        raw_delta = self.net(inp)                 # [B,3,H,W]
        delta = torch.tanh(raw_delta) * self.eps  # bounded, scaled

        x_stego = torch.clamp(x + delta, -1, 1)

        if return_delta:
            return x_stego, delta
        return x_stego
