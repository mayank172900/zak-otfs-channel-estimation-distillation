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


class ResidualAdapter(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1, stride=1),
        )
        self._zero_init_last_layer()

    def _zero_init_last_layer(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())


class GenericResidualAdapter(ResidualAdapter):
    def __init__(self) -> None:
        super().__init__(input_channels=6)


class LatticeAliasAdapter(ResidualAdapter):
    def __init__(self) -> None:
        super().__init__(input_channels=8)
