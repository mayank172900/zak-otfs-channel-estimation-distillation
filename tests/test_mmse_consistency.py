from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.mmse import mmse_dense, mmse_iterative
from zakotfs.params import load_config


def test_dense_and_iterative_mmse_match():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 1)
    frame = simulate_frame(cfg, "bpsk", 15.0, 5.0, rng)
    x_dense = mmse_dense(frame.y_dd, frame.h_eff_support, frame.noise_variance, cfg, E_d=frame.E_d)
    x_iter = mmse_iterative(frame.y_dd, frame.h_eff_support, frame.noise_variance, cfg, E_d=frame.E_d)
    rel_error = np.linalg.norm(x_dense - x_iter) / np.linalg.norm(x_dense)
    assert rel_error < 5e-2
