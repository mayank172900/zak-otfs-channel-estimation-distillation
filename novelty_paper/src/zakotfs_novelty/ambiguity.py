from __future__ import annotations

import numpy as np

from .params import SystemConfig
from .utils import centered_coord_to_index


def quasi_periodic_get(arr: np.ndarray, k: int, l: int) -> np.complex64:
    M, N = arr.shape
    base_k = k % M
    base_l = l % N
    wrap_k = (k - base_k) // M
    phase = np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
    return np.complex64(arr[base_k, base_l] * phase)


def cross_ambiguity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M, N = a.shape
    Q = M * N
    out = np.zeros((M, N), dtype=np.complex64)
    kp = np.arange(M, dtype=int)[:, None]
    lp = np.arange(N, dtype=int)[None, :]
    for k in range(M):
        shifted_k = kp - k
        base_k = np.mod(shifted_k, M)
        wrap_k = np.floor_divide(shifted_k - base_k, M)
        for l in range(N):
            shifted_l = lp - l
            base_l = np.mod(shifted_l, N)
            phase_qp = np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
            b_shifted = b[base_k, base_l] * phase_qp
            phase = np.exp(-1j * 2 * np.pi * l * shifted_k / Q)
            out[k, l] = np.sum(a * np.conjugate(b_shifted) * phase)
    return out


def cross_ambiguity_window(a: np.ndarray, b: np.ndarray, k_values: list[int] | range | np.ndarray, l_values: list[int] | range | np.ndarray) -> np.ndarray:
    M, N = a.shape
    Q = M * N
    ks = list(k_values)
    ls = list(l_values)
    out = np.zeros((len(ks), len(ls)), dtype=np.complex64)
    kp = np.arange(M, dtype=int)[:, None]
    lp = np.arange(N, dtype=int)[None, :]
    for i, k in enumerate(ks):
        shifted_k = kp - k
        base_k = np.mod(shifted_k, M)
        wrap_k = np.floor_divide(shifted_k - base_k, M)
        for j, l in enumerate(ls):
            shifted_l = lp - l
            base_l = np.mod(shifted_l, N)
            phase_qp = np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
            b_shifted = b[base_k, base_l] * phase_qp
            phase = np.exp(-1j * 2 * np.pi * l * shifted_k / Q)
            out[i, j] = np.sum(a * np.conjugate(b_shifted) * phase)
    return out


def centered_cross_ambiguity(a: np.ndarray, b: np.ndarray, config: SystemConfig) -> np.ndarray:
    raw = cross_ambiguity(a, b)
    out = np.zeros_like(raw)
    for k in range(config.M):
        ck = k if k <= config.M // 2 else k - config.M
        for l in range(config.N):
            cl = l if l <= config.N // 2 else l - config.N
            out[centered_coord_to_index(ck, config.M), centered_coord_to_index(cl, config.N)] = raw[k, l]
    return out


def self_ambiguity_support(config: SystemConfig, spread_pilot: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    amb = cross_ambiguity(spread_pilot, spread_pilot)
    return (np.abs(amb) > tol).astype(np.uint8)
