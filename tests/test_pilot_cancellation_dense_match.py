from pathlib import Path

import numpy as np

from zakotfs.channel import effective_channel_support_fast, sample_vehicular_a_channel
from zakotfs.estimators import pilot_cancellation_with_config
from zakotfs.operators import build_dense_support_matrix
from zakotfs.params import load_config
from zakotfs.waveform import spread_pilot


def test_pilot_cancellation_matches_dense_subtraction():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 91)
    xs = spread_pilot(cfg)
    channel = sample_vehicular_a_channel(cfg, rng)
    h_eff = effective_channel_support_fast(channel, cfg)
    y = (rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))).astype(np.complex64)
    dense_pilot = (build_dense_support_matrix(h_eff, cfg) @ xs.reshape(-1)).reshape(cfg.M, cfg.N)
    expected = y - dense_pilot
    actual = pilot_cancellation_with_config(y, h_eff, xs, cfg)
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
