from __future__ import annotations

import numpy as np


def nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.sum(np.abs(truth) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(estimate - truth) ** 2) / denom)


def ber_from_bits(bits_hat: np.ndarray, bits_true: np.ndarray) -> float:
    return float(np.mean(bits_hat.reshape(-1) != bits_true.reshape(-1)))
