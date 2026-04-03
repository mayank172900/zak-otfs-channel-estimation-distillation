from pathlib import Path

import numpy as np

from zakotfs.channel import PhysicalChannel, effective_channel_support_fast
from zakotfs.metrics import nmse
from zakotfs.mmse import mmse_dense
from zakotfs.params import load_config
from zakotfs.waveform import data_symbols, data_waveform
from zakotfs.operators import apply_support_operator


def test_perfect_csi_recovers_easy_single_path_case():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 88)
    symbols, _ = data_symbols("bpsk", cfg, rng)
    x = data_waveform(symbols)
    channel = PhysicalChannel(
        gains=np.array([1.0 + 0.0j], dtype=np.complex64),
        delays_s=np.array([0.0], dtype=float),
        dopplers_hz=np.array([0.0], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )
    h_eff = effective_channel_support_fast(channel, cfg)
    y = apply_support_operator(h_eff, x, cfg)
    x_hat = mmse_dense(y, h_eff, 1e-9, cfg)
    assert nmse(x_hat, x) < 1e-8
