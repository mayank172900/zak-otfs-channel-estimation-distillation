from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from statistics import NormalDist

from .params import SystemConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repo_root(config: SystemConfig) -> Path:
    return config.root


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(config: SystemConfig) -> Path:
    return ensure_dir(repo_root(config) / config.raw["paths"]["results_dir"])


def logs_dir(config: SystemConfig) -> Path:
    return ensure_dir(repo_root(config) / config.raw["paths"]["logs_dir"])


def report_dir(config: SystemConfig) -> Path:
    return ensure_dir(repo_root(config) / config.raw["paths"]["report_dir"])


def centered_coord_to_index(coord: int, size: int) -> int:
    return coord % size


def index_to_centered_coord(index: int, size: int) -> int:
    half = size // 2
    return index if index <= half else index - size


def centered_mesh(config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([index_to_centered_coord(i, config.M) for i in range(config.M)], dtype=int)
    l = np.array([index_to_centered_coord(i, config.N) for i in range(config.N)], dtype=int)
    return np.meshgrid(k, l, indexing="ij")


def complex_noise(shape: tuple[int, ...], variance: float, rng: np.random.Generator) -> np.ndarray:
    scale = math.sqrt(variance / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_mean_ci(values: np.ndarray, seed: int, num_bootstrap: int = 400, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 1:
        scalar = float(values[0])
        return scalar, scalar
    means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    means = np.sort(np.array(means))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def snr_to_noise_variance(E_signal: float, snr_db: float, num_samples: int) -> float:
    snr_lin = 10.0 ** (snr_db / 10.0)
    n0 = E_signal / (num_samples * snr_lin)
    return float(n0)


def wilson_ci(num_errors: int, num_bits: int, confidence: float = 0.95) -> tuple[float, float]:
    if num_bits <= 0:
        return 0.0, 0.0
    if num_errors < 0 or num_errors > num_bits:
        raise ValueError("num_errors must satisfy 0 <= num_errors <= num_bits")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = num_errors / num_bits
    denom = 1.0 + (z**2) / num_bits
    center = (phat + (z**2) / (2.0 * num_bits)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / num_bits) + (z**2) / (4.0 * num_bits**2))
    return max(0.0, center - radius), min(1.0, center + radius)
