from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.estimators import pilot_cancellation_with_config
from zakotfs.mmse import mmse_dense, mmse_iterative, mmse_ridge_lambda
from zakotfs.operators import build_dense_support_matrix
from zakotfs.params import load_config


def test_mmse_ridge_matches_repo_normalization():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 123)
    frame = simulate_frame(cfg, "bpsk", 18.0, 5.0, rng)
    y_data = pilot_cancellation_with_config(frame.y_dd, frame.h_eff_support, np.sqrt(frame.E_p) * frame.spread_dd, cfg)
    ridge = mmse_ridge_lambda(frame.noise_variance, frame.E_d, cfg)
    assert np.isclose(ridge, cfg.Q * frame.noise_variance / frame.E_d)
    assert np.isclose(ridge, 1.0 / frame.rho_d)

    H = build_dense_support_matrix(frame.h_eff_support, cfg)
    normal = H.conj().T @ H + ridge * np.eye(cfg.Q, dtype=np.complex64)
    rhs = H.conj().T @ y_data.reshape(-1)
    expected = np.linalg.solve(normal, rhs).reshape(cfg.M, cfg.N)

    x_dense = mmse_dense(y_data, frame.h_eff_support, frame.noise_variance, cfg, E_d=frame.E_d)
    x_iter = mmse_iterative(y_data, frame.h_eff_support, frame.noise_variance, cfg, E_d=frame.E_d)
    assert np.allclose(x_dense, expected, atol=1e-6, rtol=1e-6)
    rel_error = np.linalg.norm(x_iter - expected) / np.linalg.norm(expected)
    assert rel_error < 5e-2
