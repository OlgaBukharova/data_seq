# models/encoder.py
from __future__ import annotations
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder(A): takes image X [B,3,32,32] and message bits M [B,L],
    outputs stego image X' [B,3,32,32] in [-1, 1].
    """
    def __init__(self, L: int, hidden: int = 64):
        super().__init__()
        self.L = L
        in_ch = 3 + L

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),  # predict delta
            nn.Tanh(),
        )

    def forward(self, x, m_bits, return_delta: bool = False):
        B, _, H, W = x.shape
        m_map = m_bits.view(B, self.L, 1, 1).repeat(1, 1, H, W)
        inp = torch.cat([x, m_map], dim=1)
        delta = 0.20 * self.net(inp)
        x_stego = torch.clamp(x + delta, -1.0, 1.0)
        if return_delta:
            return x_stego, delta
        return x_stego
