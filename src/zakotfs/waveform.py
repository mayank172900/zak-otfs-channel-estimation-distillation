from __future__ import annotations

import numpy as np
from functools import lru_cache

from .modulation import sample_symbols
from .params import SystemConfig
from .pulses import chirp_spreading_filter, mn_periodic_chirp_filter


def data_symbols(modulation: str, config: SystemConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return sample_symbols(modulation, (config.M, config.N), rng)


def data_waveform(symbols: np.ndarray) -> np.ndarray:
    return (symbols / np.sqrt(symbols.size)).astype(np.complex64)


def point_pilot(config: SystemConfig) -> np.ndarray:
    pilot = np.zeros((config.M, config.N), dtype=np.complex64)
    pilot[int(config.frame["pilot_delay_bin"]), int(config.frame["pilot_doppler_bin"])] = 1.0 + 0.0j
    return pilot


def periodic_twisted_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M, N = a.shape
    out = np.zeros((M, N), dtype=np.complex64)
    Q = M * N
    for k in range(M):
        for l in range(N):
            acc = 0.0j
            for kp in range(M):
                for lp in range(N):
                    acc += a[(k - kp) % M, (l - lp) % N] * b[kp, lp] * np.exp(1j * 2 * np.pi * kp * (l - lp) / Q)
            out[k, l] = acc
    return out


def spread_pilot(config: SystemConfig) -> np.ndarray:
    kp = int(config.frame["pilot_delay_bin"])
    lp = int(config.frame["pilot_doppler_bin"])
    return _spread_pilot_exact(config.M, config.N, config.q, kp, lp).copy()


@lru_cache(maxsize=8)
def _spread_pilot_exact(M: int, N: int, q: int, kp: int, lp: int) -> np.ndarray:
    Q = M * N
    # Eq. (2) uses the MN-periodic extension of the chirp filter, not the MxN crop.
    w = np.exp(1j * 2 * np.pi * q * ((np.arange(Q)[:, None] ** 2) + (np.arange(Q)[None, :] ** 2)) / Q) / Q
    xs = np.zeros((M, N), dtype=np.complex64)
    if kp == 0 and lp == 0:
        phase_n = np.exp(1j * 2 * np.pi * (np.arange(N)[:, None] * np.arange(N)[None, :]) / N)
        for k in range(M):
            for l in range(N):
                acc = 0.0j
                for n in range(N):
                    inner = 0.0j
                    for m in range(M):
                        inner += w[(k - n * M) % Q, (l - m * N) % Q]
                    acc += phase_n[n, l] * inner
                xs[k, l] = acc
        return xs
    for k in range(M):
        for l in range(N):
            acc = 0.0j
            for n in range(N):
                for m in range(M):
                    phase = np.exp(1j * 2 * np.pi * lp * n / N) * np.exp(
                        1j * 2 * np.pi * (l - lp - m * N) * (kp + n * M) / Q
                    )
                    acc += w[(k - kp - n * M) % Q, (l - lp - m * N) % Q] * phase
            xs[k, l] = acc
    return xs


def superimposed_frame(data_dd: np.ndarray, spread_dd: np.ndarray, E_d: float, E_p: float) -> np.ndarray:
    return (np.sqrt(E_d) * data_dd + np.sqrt(E_p) * spread_dd).astype(np.complex64)
