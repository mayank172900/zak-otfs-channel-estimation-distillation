from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from zakotfs.evaluation import cnn_enhance_support
from zakotfs.params import SystemConfig
from zakotfs.training import load_cnn_checkpoint


def resolve_baseline_checkpoint(config: SystemConfig) -> Path:
    phase1_cfg = config.raw["phase1_dataset"]
    checkpoint_path = Path(str(phase1_cfg.get("baseline_checkpoint_path", "../logs/checkpoints/full_cnn_best.pt")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config.root / checkpoint_path).resolve()
    return checkpoint_path


def load_frozen_baseline(config: SystemConfig) -> tuple[torch.nn.Module, torch.device, Path]:
    checkpoint_path = resolve_baseline_checkpoint(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    model = load_cnn_checkpoint(config, checkpoint_path=checkpoint_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    return model, device, checkpoint_path


def baseline_estimate_support(model: torch.nn.Module, support_input: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    return cnn_enhance_support(model, np.asarray(support_input, dtype=np.complex64), device)
