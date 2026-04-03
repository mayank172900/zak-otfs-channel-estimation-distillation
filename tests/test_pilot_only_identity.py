from pathlib import Path

import numpy as np

from zakotfs.ambiguity import cross_ambiguity_window
from zakotfs.channel import effective_channel_support_fast, sample_vehicular_a_channel
from zakotfs.operators import apply_support_operator
from zakotfs.params import load_config
from zakotfs.waveform import spread_pilot


def test_pilot_only_read_off_matches_true_channel_inside_support():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 100)
    xs = spread_pilot(cfg)
    channel = sample_vehicular_a_channel(cfg, rng)
    heff = effective_channel_support_fast(channel, cfg)
    y = apply_support_operator(heff, xs, cfg)
    support = cross_ambiguity_window(y, xs, range(-13, 14), range(-21, 22))
    rel_nmse = np.sum(np.abs(support - heff) ** 2) / np.sum(np.abs(heff) ** 2)
    assert rel_nmse < 1e-8
