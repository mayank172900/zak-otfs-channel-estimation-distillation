from pathlib import Path

import numpy as np

from zakotfs.channel import PhysicalChannel, effective_channel_taps_fast, effective_channel_taps_reference
from zakotfs.metrics import nmse
from zakotfs.operators import apply_heff_operator
from zakotfs.params import load_config


def test_fast_and_reference_channel_give_same_dd_output():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 77)
    x = (rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))).astype(np.complex64)
    channel = PhysicalChannel(
        gains=np.array([0.8 + 0.2j], dtype=np.complex64),
        delays_s=np.array([1.09e-6], dtype=float),
        dopplers_hz=np.array([-300.0], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )
    h_ref = effective_channel_taps_reference(channel, cfg)
    h_fast = effective_channel_taps_fast(channel, cfg)
    y_ref = apply_heff_operator(h_ref, x, cfg)
    y_fast = apply_heff_operator(h_fast, x, cfg)
    assert nmse(y_fast, y_ref) < 1e-4
