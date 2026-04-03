from __future__ import annotations

import torch
from torch import nn


class PaperCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(27, 27), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(9, 9), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(15, 15), stride=1, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
