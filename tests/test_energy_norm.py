from pathlib import Path

import numpy as np

from zakotfs.params import load_config
from zakotfs.waveform import data_waveform, spread_pilot


def test_energy_normalization_data_and_spread_pilot():
    cfg = load_config(Path("configs/system.yaml"))
    data = np.ones((cfg.M, cfg.N), dtype=np.complex64)
    data_dd = data_waveform(data)
    pilot = spread_pilot(cfg)
    assert np.isclose(np.sum(np.abs(data_dd) ** 2), 1.0, atol=1e-6)
    assert np.isclose(np.sum(np.abs(pilot) ** 2), 1.0, atol=1e-4)
