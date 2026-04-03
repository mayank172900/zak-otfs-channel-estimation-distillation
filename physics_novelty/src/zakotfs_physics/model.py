from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from zakotfs.params import SystemConfig


def _phase2_model_cfg(config: SystemConfig) -> dict:
    if "phase2_model" not in config.raw:
        raise ValueError("Phase 2 model config missing")
    return config.raw["phase2_model"]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class Phase2ResidualUncertaintyModel(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_channels: int = 32, num_blocks: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualConvBlock(hidden_channels, kernel_size) for _ in range(num_blocks)))
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(x)
        features = self.blocks(features)
        out = self.head(features)
        delta = out[:, 0:2]
        # Keep S non-negative so confidence=exp(-S) stays bounded in (0, 1].
        uncertainty = F.softplus(out[:, 2:3])
        return {"delta": delta, "uncertainty": uncertainty}

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def instantiate_phase2_model(config: SystemConfig) -> Phase2ResidualUncertaintyModel:
    cfg = _phase2_model_cfg(config)
    return Phase2ResidualUncertaintyModel(
        in_channels=int(cfg.get("in_channels", 6)),
        hidden_channels=int(cfg.get("hidden_channels", 32)),
        num_blocks=int(cfg.get("num_blocks", 3)),
        kernel_size=int(cfg.get("kernel_size", 3)),
    )


def predict_phase2(model: nn.Module, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    single = inputs.ndim == 3
    if single:
        inputs = inputs.unsqueeze(0)
    outputs = model(inputs)
    h_base = inputs[:, 4:6]
    h_hat = h_base + outputs["delta"]
    confidence = torch.exp(-outputs["uncertainty"])
    result = {
        "delta": outputs["delta"],
        "uncertainty": outputs["uncertainty"],
        "confidence": confidence,
        "h_base": h_base,
        "h_hat": h_hat,
    }
    if single:
        return {key: value[0] for key, value in result.items()}
    return result
