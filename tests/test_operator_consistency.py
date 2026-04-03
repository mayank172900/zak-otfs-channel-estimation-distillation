from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.operators import apply_support_operator, build_dense_support_matrix
from zakotfs.params import load_config


def test_dense_and_matrix_free_operator_consistency():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed)
    frame = simulate_frame(cfg, "bpsk", 15.0, 5.0, rng)
    y_free = apply_support_operator(frame.h_eff_support, frame.x_dd, cfg)
    H = build_dense_support_matrix(frame.h_eff_support, cfg)
    y_dense = (H @ frame.x_dd.reshape(-1)).reshape(cfg.M, cfg.N)
    assert np.allclose(y_free, y_dense, atol=1e-5, rtol=1e-5)
    assert H.shape == (cfg.Q, cfg.Q)
