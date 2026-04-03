from pathlib import Path

import numpy as np

from zakotfs.ambiguity import cross_ambiguity
from zakotfs.params import load_config
from zakotfs.waveform import spread_pilot


def test_spread_pilot_self_ambiguity_has_expected_lattice_peaks():
    cfg = load_config(Path("configs/system.yaml"))
    xs = spread_pilot(cfg)
    A = cross_ambiguity(xs, xs)
    assert np.isclose(np.abs(A[0, 0]), 1.0, atol=1e-6)
    assert np.isclose(np.abs(A[27 % cfg.M, 7 % cfg.N]), 1.0, atol=1e-6)
    assert np.abs(A[1 % cfg.M, 0 % cfg.N]) < 1e-5
    assert np.abs(A[2 % cfg.M, 6 % cfg.N]) < 1e-5
