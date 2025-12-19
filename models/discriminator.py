# models/discriminator.py
from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Binary classifier: image [B,3,32,32] -> logit [B]
    """
    def __init__(self, hidden: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden * 4, hidden * 4, 3, stride=2, padding=1),  # 2x2
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4 * 2 * 2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        logit = self.head(f).squeeze(1)  # [B]
        return logit
