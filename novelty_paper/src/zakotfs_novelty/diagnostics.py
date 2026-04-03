from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .ambiguity import cross_ambiguity
from .compat import dataclass_slots
from .dataset import simulate_frame
from .lattice import crop_support
from .operators import apply_heff_operator
from .params import SystemConfig


@dataclass_slots()
class ErrorDecomposition:
    effective_energy: float
    alias_energy: float
    data_interference_energy: float
    noise_energy: float
    total_energy: float


def error_decomposition(config: SystemConfig, seed_offset: int = 0) -> tuple[dict[str, np.ndarray], ErrorDecomposition]:
    rng = np.random.default_rng(config.seed + 500 + seed_offset)
    frame = simulate_frame(config, "bpsk", 15.0, 5.0, rng)
    A_xsxs = cross_ambiguity(frame.spread_dd, frame.spread_dd)
    A_xdxs = cross_ambiguity(frame.data_dd, frame.spread_dd)
    noise_dd = frame.y_dd - frame.y_clean
    A_nxs = cross_ambiguity(noise_dd, frame.spread_dd)
    effective_term = np.sqrt(frame.E_p) * frame.h_eff
    alias_full = np.sqrt(frame.E_p) * (apply_heff_operator(frame.h_eff, A_xsxs, config) - frame.h_eff)
    data_term = np.sqrt(frame.E_d) * apply_heff_operator(frame.h_eff, A_xdxs, config)
    noise_term = A_nxs
    components = {
        "effective_term": crop_support(effective_term, config),
        "alias_term": crop_support(alias_full, config),
        "data_term": crop_support(data_term, config),
        "noise_term": crop_support(noise_term, config),
        "read_off": frame.support_input,
        "truth": frame.support_true,
    }
    energies = {name: float(np.sum(np.abs(value) ** 2)) for name, value in components.items() if name.endswith("_term")}
    summary = ErrorDecomposition(
        effective_energy=energies["effective_term"],
        alias_energy=energies["alias_term"],
        data_interference_energy=energies["data_term"],
        noise_energy=energies["noise_term"],
        total_energy=float(sum(energies.values())),
    )
    return components, summary


def decomposition_dict(summary: ErrorDecomposition) -> dict[str, float]:
    return asdict(summary)
