from __future__ import annotations

import numpy as np

from .ambiguity import cross_ambiguity, cross_ambiguity_window
from .compat import dataclass_slots
from .lattice import crop_support, derive_support_geometry, embed_support_image
from .operators import apply_heff_operator, apply_support_operator
from .params import SystemConfig


@dataclass_slots()
class EstimationBundle:
    ambiguity: np.ndarray
    h_hat_raw: np.ndarray
    h_hat_thresholded: np.ndarray
    support_input: np.ndarray
    support_true: np.ndarray


def read_off_estimator(y_dd: np.ndarray, xs_dd: np.ndarray, E_p: float, config: SystemConfig) -> EstimationBundle:
    ambiguity = cross_ambiguity(y_dd, xs_dd) / np.sqrt(E_p)
    g = derive_support_geometry(config)
    support_input = cross_ambiguity_window(y_dd, xs_dd, range(g.k_min, g.k_max + 1), range(g.l_min, g.l_max + 1)) / np.sqrt(E_p)
    h_hat_raw = embed_support_image(support_input, config)
    return EstimationBundle(
        ambiguity=ambiguity,
        h_hat_raw=h_hat_raw,
        h_hat_thresholded=h_hat_raw.copy(),
        support_input=support_input,
        support_true=np.zeros_like(support_input),
    )


def threshold_readoff(h_hat_raw: np.ndarray, rho_d: float, rho_p: float, config: SystemConfig) -> np.ndarray:
    sigma = np.sqrt((1.0 / config.Q) * (1.0 + rho_d / rho_p))
    threshold = config.estimation["threshold_multiplier"] * sigma
    out = h_hat_raw.copy()
    out[np.abs(out) < threshold] = 0.0
    return out


def pilot_cancellation(y_dd: np.ndarray, h_est: np.ndarray, xs_dd: np.ndarray) -> np.ndarray:
    raise RuntimeError("pilot_cancellation now requires pilot_cancellation_with_config")


def pilot_cancellation_with_config(y_dd: np.ndarray, h_est: np.ndarray, xs_dd: np.ndarray, config: SystemConfig) -> np.ndarray:
    op = apply_heff_operator if h_est.shape == (config.M, config.N) else apply_support_operator
    return y_dd - op(h_est, xs_dd, config)


def support_images(h_eff: np.ndarray, h_hat: np.ndarray, config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    return crop_support(h_hat, config), crop_support(h_eff, config)


def embed_cnn_output(support_img: np.ndarray, config: SystemConfig) -> np.ndarray:
    return embed_support_image(support_img, config)
