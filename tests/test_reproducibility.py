from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config


def test_fixed_seed_reproducibility():
    cfg = load_config(Path("configs/system.yaml"))
    frame1 = simulate_frame(cfg, "bpsk", 15.0, 5.0, np.random.default_rng(cfg.seed + 2))
    frame2 = simulate_frame(cfg, "bpsk", 15.0, 5.0, np.random.default_rng(cfg.seed + 2))
    assert np.allclose(frame1.h_eff, frame2.h_eff)
    assert np.allclose(frame1.y_dd, frame2.y_dd)
    assert np.allclose(frame1.support_input, frame2.support_input)
