from pathlib import Path

import numpy as np

from zakotfs.channel import PhysicalChannel, effective_channel_support_fast, effective_channel_support_reference, sample_vehicular_a_channel
from zakotfs.metrics import nmse
from zakotfs.operators import apply_support_operator
from zakotfs.params import load_config


def _single_path(delay_s: float, doppler_hz: float) -> PhysicalChannel:
    return PhysicalChannel(
        gains=np.array([1.0 + 0.0j], dtype=np.complex64),
        delays_s=np.array([delay_s], dtype=float),
        dopplers_hz=np.array([doppler_hz], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )


def test_single_path_fractional_support_reference_and_fast_agree():
    cfg = load_config(Path("configs/system.yaml"))
    channel = _single_path(0.71e-6, 220.0)
    h_ref = effective_channel_support_reference(channel, cfg)
    h_fast = effective_channel_support_fast(channel, cfg)
    assert nmse(h_fast, h_ref) < 1e-4


def test_random_vehicular_a_support_reference_and_fast_stay_close():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 333)
    x = (rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))).astype(np.complex64)
    channel = sample_vehicular_a_channel(cfg, rng)
    h_ref = effective_channel_support_reference(channel, cfg)
    h_fast = effective_channel_support_fast(channel, cfg)
    y_ref = apply_support_operator(h_ref, x, cfg)
    y_fast = apply_support_operator(h_fast, x, cfg)
    assert nmse(h_fast, h_ref) < 5e-2
    assert nmse(y_fast, y_ref) < 5e-2
