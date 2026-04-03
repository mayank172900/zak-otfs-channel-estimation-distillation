from pathlib import Path

import numpy as np

from zakotfs.channel import PhysicalChannel, effective_channel_taps_fast, effective_channel_taps_reference
from zakotfs.metrics import nmse
from zakotfs.params import load_config


def _single_path(delay_s: float, doppler_hz: float) -> PhysicalChannel:
    return PhysicalChannel(
        gains=np.array([1.0 + 0.0j], dtype=np.complex64),
        delays_s=np.array([delay_s], dtype=float),
        dopplers_hz=np.array([doppler_hz], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )


def test_single_path_integer_reference_and_fast_agree():
    cfg = load_config(Path("configs/system.yaml"))
    channel = _single_path(0.0, 0.0)
    h_ref = effective_channel_taps_reference(channel, cfg)
    h_fast = effective_channel_taps_fast(channel, cfg)
    assert nmse(h_fast, h_ref) < 1e-5
    assert np.unravel_index(np.argmax(np.abs(h_ref)), h_ref.shape) == (0, 0)


def test_single_path_fractional_reference_and_fast_agree():
    cfg = load_config(Path("configs/system.yaml"))
    channel = _single_path(0.71e-6, 220.0)
    h_ref = effective_channel_taps_reference(channel, cfg)
    h_fast = effective_channel_taps_fast(channel, cfg)
    assert nmse(h_fast, h_ref) < 1e-4
