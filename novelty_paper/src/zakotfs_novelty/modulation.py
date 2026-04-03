from __future__ import annotations

import numpy as np


def constellation(name: str) -> tuple[np.ndarray, np.ndarray]:
    name = name.lower().replace("-", "").replace("_", "")
    if name == "bpsk":
        points = np.array([-1.0, 1.0], dtype=np.complex64)
        bits = np.array([[0], [1]], dtype=np.int8)
        return points, bits
    if name in {"8qam", "8qamcross"}:
        points = np.array(
            [
                -3 + 0j,
                -1 - 1j,
                -1 + 1j,
                0 - 3j,
                0 + 3j,
                1 - 1j,
                1 + 1j,
                3 + 0j,
            ],
            dtype=np.complex64,
        )
        points = points / np.sqrt(np.mean(np.abs(points) ** 2))
        bits = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 0],
            ],
            dtype=np.int8,
        )
        return points, bits
    if name in {"8qamstar"}:
        angles = np.deg2rad(np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float))
        radii = np.array([1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)], dtype=float)
        points = (radii * np.exp(1j * angles)).astype(np.complex64)
        points = points / np.sqrt(np.mean(np.abs(points) ** 2))
        bits = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.int8)
        return points, bits
    raise ValueError(f"Unsupported modulation: {name}")


def sample_symbols(modulation: str, shape: tuple[int, ...], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points, bits = constellation(modulation)
    indices = rng.integers(0, len(points), size=shape)
    return points[indices], bits[indices]


def hard_demodulate(symbols: np.ndarray, modulation: str) -> tuple[np.ndarray, np.ndarray]:
    points, bits = constellation(modulation)
    distances = np.abs(symbols[..., None] - points[None, None, :]) ** 2
    idx = np.argmin(distances, axis=-1)
    return points[idx], bits[idx]
