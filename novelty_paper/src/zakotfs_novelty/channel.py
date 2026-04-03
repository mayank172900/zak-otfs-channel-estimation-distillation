from __future__ import annotations

import numpy as np

from .compat import dataclass_slots, strict_zip
from .lattice import derive_support_geometry, embed_support_image
from .params import SystemConfig
from .pulses import (
    effective_pulse_kernel,
    gs_delay_autocorrelation,
    gs_delay_overlap_exact,
    gs_doppler_autocorrelation,
    gs_doppler_overlap_exact,
)
from .utils import complex_noise


@dataclass_slots()
class PhysicalChannel:
    gains: np.ndarray
    delays_s: np.ndarray
    dopplers_hz: np.ndarray
    thetas_rad: np.ndarray


def sample_vehicular_a_channel(config: SystemConfig, rng: np.random.Generator) -> PhysicalChannel:
    c = config.channel
    powers_lin = 10.0 ** (np.array(c["relative_powers_db"], dtype=float) / 10.0)
    powers_lin = powers_lin / np.sum(powers_lin)
    gains = np.sqrt(powers_lin / 2.0) * (
        rng.standard_normal(len(powers_lin)) + 1j * rng.standard_normal(len(powers_lin))
    )
    delays_s = np.array(c["delays_s"], dtype=float)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=len(powers_lin))
    dopplers_hz = c["max_doppler_hz"] * np.cos(thetas)
    return PhysicalChannel(gains=gains.astype(np.complex64), delays_s=delays_s, dopplers_hz=dopplers_hz.astype(float), thetas_rad=thetas)


def _support_axes(config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    g = derive_support_geometry(config)
    delay_s = np.arange(g.k_min, g.k_max + 1, dtype=float) / config.B
    doppler_hz = np.arange(g.l_min, g.l_max + 1, dtype=float) / config.T
    return delay_s, doppler_hz


def effective_channel_method(config: SystemConfig) -> str:
    return str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")).lower()


def effective_channel_support_reference(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    # Reference implementation of Eq. (26) in the GS-filter paper. It is slower but
    # captures the path-dependent phase and exact matched-filter overlaps.
    delay_s, doppler_hz = _support_axes(config)
    heff = np.zeros((delay_s.size, doppler_hz.size), dtype=np.complex64)
    for gain, tau_i, nu_i in strict_zip(channel.gains, channel.delays_s, channel.dopplers_hz):
        i1 = gs_delay_overlap_exact(delay_s, tau_i, nu_i, config).astype(np.complex64)
        i2 = gs_doppler_overlap_exact(delay_s, doppler_hz, nu_i, config).astype(np.complex64)
        path_phase = np.exp(1j * 2 * np.pi * nu_i * (delay_s - tau_i)).astype(np.complex64)
        heff += gain * path_phase[:, None] * i1[:, None] * i2
    return heff


def effective_channel_support_fast(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    # Fast crystalline-regime approximation consistent with Corollary 3 in the Zak-OTFS
    # receiver paper: pathwise DD localization with phase exp(j 2 pi (nu tau - nu_i tau_i)).
    delay_s, doppler_hz = _support_axes(config)
    heff = np.zeros((delay_s.size, doppler_hz.size), dtype=np.complex64)
    for gain, tau_i, nu_i in strict_zip(channel.gains, channel.delays_s, channel.dopplers_hz):
        delay_auto = gs_delay_autocorrelation(delay_s - tau_i, config).astype(np.complex64)
        doppler_auto = gs_doppler_autocorrelation(doppler_hz - nu_i, config).astype(np.complex64)
        path_phase = np.exp(1j * 2 * np.pi * (delay_s[:, None] * doppler_hz[None, :] - nu_i * tau_i)).astype(np.complex64)
        heff += gain * path_phase * delay_auto[:, None] * doppler_auto[None, :]
    return heff


def effective_channel_support_envelope(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    delay_s, doppler_hz = _support_axes(config)
    heff = np.zeros((delay_s.size, doppler_hz.size), dtype=np.complex64)
    for gain, tau_i, nu_i in strict_zip(channel.gains, channel.delays_s, channel.dopplers_hz):
        envelope = effective_pulse_kernel(delay_s[:, None] - tau_i, doppler_hz[None, :] - nu_i, config)
        phase = np.exp(1j * 2 * np.pi * (delay_s[:, None] * doppler_hz[None, :] - nu_i * tau_i))
        heff += gain * envelope * phase.astype(np.complex64)
    return heff


def effective_channel_support(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    method = effective_channel_method(config)
    if method == "reference":
        return effective_channel_support_reference(channel, config)
    if method == "envelope":
        return effective_channel_support_envelope(channel, config)
    return effective_channel_support_fast(channel, config)


def effective_channel_taps_reference(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    return embed_support_image(effective_channel_support_reference(channel, config), config)


def effective_channel_taps_fast(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    return embed_support_image(effective_channel_support_fast(channel, config), config)


def effective_channel_taps(channel: PhysicalChannel, config: SystemConfig) -> np.ndarray:
    return embed_support_image(effective_channel_support(channel, config), config)


def add_awgn(signal: np.ndarray, noise_variance: float, rng: np.random.Generator) -> np.ndarray:
    return signal + complex_noise(signal.shape, noise_variance, rng)
