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
    """
    x:      [B,3,H,W] in [-1,1]
    m_bits: [B,L] float {0,1}
    """
    B, _, H, W = x.shape

    # 1) expand message bits to spatial map and concatenate with image
    m_map = m_bits.view(B, self.L, 1, 1).expand(B, self.L, H, W)
    inp = torch.cat([x, m_map], dim=1)

    # 2) predict raw delta (unbounded)
    raw_delta = self.net(inp)  # [B,3,H,W]

    # 3) constrain and scale delta (THIS is what prevents encoder collapse)
    eps = 0.10  # попробуй 0.10 как старт (0.05 у тебя уже давал слишком слабый сигнал)
    delta = torch.tanh(raw_delta) * eps

    # 4) apply stego and clamp back to valid range
    x_stego = torch.clamp(x + delta, -1, 1)

    if return_delta:
        return x_stego, delta
    return x_stego

