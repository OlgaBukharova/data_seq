# models/decoder.py
from __future__ import annotations
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder(R): takes stego image X' [B,3,32,32],
    outputs logits for message bits [B,L].
    """
    def __init__(self, L: int, hidden: int = 64):
        super().__init__()
        self.L = L
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, L),  # logits
        )

    def forward(self, x_stego: torch.Tensor) -> torch.Tensor:
        f = self.features(x_stego)
        logits = self.head(f)
        return logits
