# models/decoder.py
from __future__ import annotations
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder: X' [B,3,32,32] -> logits [B,L]
    Возвращает ЛОГИТЫ (без sigmoid) для BCEWithLogitsLoss.
    """
    def __init__(self, L: int, hidden: int = 160):
        super().__init__()
        self.L = int(L)

        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 2x2
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.L),
        )

    def forward(self, x_stego: torch.Tensor) -> torch.Tensor:
        f = self.features(x_stego)
        return self.head(f)
