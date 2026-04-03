from __future__ import annotations

from functools import lru_cache

import numpy as np

from .params import SystemConfig
from .utils import centered_mesh


def gs_delay_pulse(delay_s: np.ndarray, config: SystemConfig) -> np.ndarray:
    p = config.pulse
    B = config.B
    value = (
        p["Omega_tau"]
        * np.sqrt(B)
        * np.sinc(B * delay_s)
        * np.exp(-p["alpha_tau"] * (B * delay_s) ** 2)
    )
    return np.asarray(value, dtype=np.complex64)


def gs_doppler_pulse(doppler_hz: np.ndarray, config: SystemConfig) -> np.ndarray:
    p = config.pulse
    T = config.T
    value = (
        p["Omega_nu"]
        * np.sqrt(T)
        * np.sinc(T * doppler_hz)
        * np.exp(-p["alpha_nu"] * (T * doppler_hz) ** 2)
    )
    return np.asarray(value, dtype=np.complex64)


def gs_transmit_pulse(delay_s: np.ndarray, doppler_hz: np.ndarray, config: SystemConfig) -> np.ndarray:
    return (gs_delay_pulse(delay_s, config) * gs_doppler_pulse(doppler_hz, config)).astype(np.complex64)


def gs_matched_filter(delay_s: np.ndarray, doppler_hz: np.ndarray, config: SystemConfig) -> np.ndarray:
    return np.conjugate(gs_transmit_pulse(-delay_s, -doppler_hz, config)) * np.exp(1j * 2 * np.pi * delay_s * doppler_hz)


def _quad_points(config: SystemConfig, axis: str) -> int:
    pulse_cfg = config.raw.get("numerics", {}).get("pulse_quadrature", {})
    return int(pulse_cfg.get(f"{axis}_points", 2049))


def _quad_span(config: SystemConfig, axis: str) -> float:
    pulse_cfg = config.raw.get("numerics", {}).get("pulse_quadrature", {})
    return float(pulse_cfg.get(f"{axis}_span", 10.0))


@lru_cache(maxsize=8)
def _delay_quadrature_grid(B: float, points: int, span: float) -> np.ndarray:
    return np.linspace(-span / B, span / B, points, dtype=float)


@lru_cache(maxsize=8)
def _doppler_quadrature_grid(T: float, points: int, span: float) -> np.ndarray:
    return np.linspace(-span / T, span / T, points, dtype=float)


def delay_quadrature_grid(config: SystemConfig) -> np.ndarray:
    return _delay_quadrature_grid(config.B, _quad_points(config, "delay"), _quad_span(config, "delay"))


def doppler_quadrature_grid(config: SystemConfig) -> np.ndarray:
    return _doppler_quadrature_grid(config.T, _quad_points(config, "doppler"), _quad_span(config, "doppler"))


def gs_delay_autocorrelation(delay_offset_s: np.ndarray, config: SystemConfig) -> np.ndarray:
    tau = delay_quadrature_grid(config)
    w = gs_delay_pulse(tau, config).astype(np.complex128)
    shifted = gs_delay_pulse(np.asarray(delay_offset_s)[..., None] - tau[None, :], config).astype(np.complex128)
    return np.trapezoid(w[None, :] * shifted, tau, axis=-1).astype(np.complex64)


def gs_doppler_autocorrelation(doppler_offset_hz: np.ndarray, config: SystemConfig) -> np.ndarray:
    nu = doppler_quadrature_grid(config)
    w = gs_doppler_pulse(nu, config).astype(np.complex128)
    shifted = gs_doppler_pulse(np.asarray(doppler_offset_hz)[..., None] - nu[None, :], config).astype(np.complex128)
    return np.trapezoid(w[None, :] * shifted, nu, axis=-1).astype(np.complex64)


def gs_delay_overlap_exact(delay_s: np.ndarray, path_delay_s: float, path_doppler_hz: float, config: SystemConfig) -> np.ndarray:
    tau = delay_quadrature_grid(config)
    w = gs_delay_pulse(tau, config).astype(np.complex128)
    shifted = gs_delay_pulse(np.asarray(delay_s)[..., None] - path_delay_s - tau[None, :], config).astype(np.complex128)
    phase = np.exp(-1j * 2 * np.pi * path_doppler_hz * tau)[None, :]
    return np.trapezoid(w[None, :] * shifted * phase, tau, axis=-1).astype(np.complex64)


def gs_doppler_overlap_exact(delay_s: np.ndarray, doppler_hz: np.ndarray, path_doppler_hz: float, config: SystemConfig) -> np.ndarray:
    nu = doppler_quadrature_grid(config)
    w = gs_doppler_pulse(nu, config).astype(np.complex128)
    phase = np.exp(1j * 2 * np.pi * np.asarray(delay_s)[:, None] * nu[None, :]).astype(np.complex128)
    shifted = gs_doppler_pulse(np.asarray(doppler_hz)[:, None] - path_doppler_hz - nu[None, :], config).astype(np.complex128)
    integrand = phase[:, None, :] * shifted[None, :, :] * w[None, None, :]
    return np.trapezoid(integrand, nu, axis=-1).astype(np.complex64)


def effective_pulse_kernel(delay_s: np.ndarray, doppler_hz: np.ndarray, config: SystemConfig) -> np.ndarray:
    # Kept as a compact localized envelope helper. The channel code now builds the
    # paper-consistent phase outside this helper.
    p = config.pulse
    B = config.B
    T = config.T
    envelope = (
        p["Omega_tau"]
        * p["Omega_nu"]
        * np.sinc(B * delay_s)
        * np.sinc(T * doppler_hz)
        * np.exp(-p["alpha_tau"] * (B * delay_s) ** 2 - p["alpha_nu"] * (T * doppler_hz) ** 2)
    )
    return np.asarray(envelope, dtype=np.complex64)


def chirp_spreading_filter(config: SystemConfig) -> np.ndarray:
    M, N, q = config.M, config.N, config.q
    k = np.arange(M, dtype=float)[:, None]
    l = np.arange(N, dtype=float)[None, :]
    return (np.exp(1j * 2 * np.pi * q * (k**2 + l**2) / (M * N)) / np.sqrt(M * N)).astype(np.complex64)


def mn_periodic_chirp_filter(config: SystemConfig) -> np.ndarray:
    Q = config.Q
    q = config.q
    k = np.arange(Q, dtype=float)[:, None]
    l = np.arange(Q, dtype=float)[None, :]
    return (np.exp(1j * 2 * np.pi * q * (k**2 + l**2) / Q) / Q).astype(np.complex64)


def support_kernel_grid(config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    k, l = centered_mesh(config)
    delay_s = k / config.B
    doppler_hz = l / config.T
    return delay_s, doppler_hz
