from pathlib import Path

import numpy as np
import torch

from zakotfs.dataset import simulate_frame
from zakotfs.mmse import mmse_dense, mmse_dense_torch
from zakotfs.params import load_config


def test_torch_dense_mmse_matches_numpy_dense():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 515)
    frame = simulate_frame(cfg, "bpsk", 15.0, 5.0, rng)
    x_np = mmse_dense(frame.y_dd, frame.h_eff_support, frame.noise_variance, cfg, E_d=frame.E_d)
    x_torch = mmse_dense_torch(
        frame.y_dd,
        frame.h_eff_support,
        frame.noise_variance,
        cfg,
        E_d=frame.E_d,
        device=torch.device("cpu"),
    )
    assert np.allclose(x_torch, x_np, atol=1e-5, rtol=1e-5)
