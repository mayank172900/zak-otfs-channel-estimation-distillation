from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from zakotfs.params import SystemConfig


@dataclass(frozen=True)
class StudentSpec:
    channels: tuple[int, int, int]
    kernels: tuple[int, int, int, int]


STUDENT_SPECS: dict[str, StudentSpec] = {
    "lite_l": StudentSpec(channels=(32, 16, 16), kernels=(15, 7, 5, 9)),
    "lite_m": StudentSpec(channels=(24, 12, 12), kernels=(13, 7, 5, 9)),
    "lite_s": StudentSpec(channels=(16, 8, 8), kernels=(11, 5, 3, 7)),
    "lite_xs": StudentSpec(channels=(12, 6, 6), kernels=(9, 5, 3, 5)),
}


class LiteBranchCNN(nn.Module):
    def __init__(self, channels: tuple[int, int, int], kernels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3, k4 = kernels
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(k1, k1), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=(k2, k2), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=(k3, k3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(c3, 1, kernel_size=(k4, k4), stride=1, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistilledStudentCNN(nn.Module):
    def __init__(self, spec: StudentSpec) -> None:
        super().__init__()
        self.spec = spec
        self.branch = LiteBranchCNN(spec.channels, spec.kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())


def _distill_model_cfg(config: SystemConfig) -> dict:
    if "distill_model" not in config.raw:
        raise ValueError("distill_model config missing")
    return config.raw["distill_model"]


def instantiate_student_model(config: SystemConfig) -> DistilledStudentCNN:
    cfg = _distill_model_cfg(config)
    variant = str(cfg.get("variant", "lite_l")).lower()
    if variant in STUDENT_SPECS:
        spec = STUDENT_SPECS[variant]
    else:
        channels = tuple(int(x) for x in cfg["channels"])
        kernels = tuple(int(x) for x in cfg["kernels"])
        if len(channels) != 3 or len(kernels) != 4:
            raise ValueError("Custom student spec expects 3 channels and 4 kernels")
        spec = StudentSpec(channels=channels, kernels=kernels)
    return DistilledStudentCNN(spec)
