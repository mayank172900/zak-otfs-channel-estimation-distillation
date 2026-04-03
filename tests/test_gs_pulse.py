from pathlib import Path

import numpy as np

from zakotfs.params import load_config
from zakotfs.pulses import gs_transmit_pulse


def test_gs_pulse_energy_is_close_to_unit_energy():
    cfg = load_config(Path("configs/system.yaml"))
    tau = np.linspace(-8 / cfg.B, 8 / cfg.B, 1201)
    nu = np.linspace(-8 / cfg.T, 8 / cfg.T, 1201)
    dt = tau[1] - tau[0]
    dn = nu[1] - nu[0]
    pulse = gs_transmit_pulse(tau[:, None], nu[None, :], cfg)
    energy = np.sum(np.abs(pulse) ** 2) * dt * dn
    assert 0.9 < energy < 1.05
