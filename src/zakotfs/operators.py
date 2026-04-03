from __future__ import annotations

import numpy as np

from .compat import dataclass_slots
from .lattice import derive_support_geometry
from .params import SystemConfig
from .utils import index_to_centered_coord


@dataclass_slots()
class ShiftTerm:
    delay: int
    doppler: int
    value: np.complex64


def nonzero_shift_terms(heff: np.ndarray, config: SystemConfig, tol: float = 1e-9) -> list[ShiftTerm]:
    terms: list[ShiftTerm] = []
    for i in range(config.M):
        for j in range(config.N):
            value = heff[i, j]
            if abs(value) > tol:
                terms.append(ShiftTerm(index_to_centered_coord(i, config.M), index_to_centered_coord(j, config.N), np.complex64(value)))
    return terms


def support_shift_terms(support_img: np.ndarray, config: SystemConfig, tol: float = 1e-9) -> list[ShiftTerm]:
    g = derive_support_geometry(config)
    terms: list[ShiftTerm] = []
    for i, k in enumerate(range(g.k_min, g.k_max + 1)):
        for j, l in enumerate(range(g.l_min, g.l_max + 1)):
            value = support_img[i, j]
            if abs(value) > tol:
                terms.append(ShiftTerm(k, l, np.complex64(value)))
    return terms


def apply_shift_terms(terms: list[ShiftTerm], x: np.ndarray, config: SystemConfig) -> np.ndarray:
    M, N, Q = config.M, config.N, config.Q
    k = np.arange(M, dtype=int)[:, None]
    l = np.arange(N, dtype=int)[None, :]
    y = np.zeros((M, N), dtype=np.complex64)
    for term in terms:
        src_k = k - term.delay
        src_l = l - term.doppler
        base_k = np.mod(src_k, M)
        base_l = np.mod(src_l, N)
        wrap_k = np.floor_divide(src_k - base_k, M)
        x_qp = x[base_k, base_l] * np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
        phase = np.exp(1j * 2 * np.pi * src_k * term.doppler / Q)
        y += term.value * x_qp * phase
    return y


def apply_heff_operator(heff: np.ndarray, x: np.ndarray, config: SystemConfig) -> np.ndarray:
    return apply_shift_terms(nonzero_shift_terms(heff, config), x, config)


def apply_support_operator(support_img: np.ndarray, x: np.ndarray, config: SystemConfig) -> np.ndarray:
    return apply_shift_terms(support_shift_terms(support_img, config), x, config)


def apply_shift_terms_adjoint(terms: list[ShiftTerm], y: np.ndarray, config: SystemConfig) -> np.ndarray:
    M, N, Q = config.M, config.N, config.Q
    k = np.arange(M, dtype=int)[:, None]
    l = np.arange(N, dtype=int)[None, :]
    x = np.zeros((M, N), dtype=np.complex64)
    for term in terms:
        src_k = k - term.delay
        src_l = l - term.doppler
        base_k = np.broadcast_to(np.mod(src_k, M), (M, N))
        base_l = np.broadcast_to(np.mod(src_l, N), (M, N))
        wrap_k = np.floor_divide(src_k - base_k, M)
        phase_qp = np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
        phase = np.exp(1j * 2 * np.pi * src_k * term.doppler / Q)
        coeff = np.conjugate(term.value * phase_qp * phase)
        np.add.at(x, (base_k.ravel(), base_l.ravel()), (coeff * y).ravel())
    return x


def apply_heff_adjoint(heff: np.ndarray, y: np.ndarray, config: SystemConfig) -> np.ndarray:
    return apply_shift_terms_adjoint(nonzero_shift_terms(heff, config), y, config)


def apply_support_adjoint(support_img: np.ndarray, y: np.ndarray, config: SystemConfig) -> np.ndarray:
    return apply_shift_terms_adjoint(support_shift_terms(support_img, config), y, config)


def build_dense_from_terms(terms: list[ShiftTerm], config: SystemConfig) -> np.ndarray:
    M, N, Q = config.M, config.N, config.Q
    rows = np.arange(Q, dtype=int).reshape(M, N)
    k = np.arange(M, dtype=int)[:, None]
    l = np.arange(N, dtype=int)[None, :]
    H = np.zeros((Q, Q), dtype=np.complex64)
    for term in terms:
        src_k = k - term.delay
        src_l = l - term.doppler
        base_k = np.mod(src_k, M)
        base_l = np.mod(src_l, N)
        wrap_k = np.floor_divide(src_k - base_k, M)
        cols = base_k * N + base_l
        coeff = term.value * np.exp(1j * 2 * np.pi * wrap_k * base_l / N) * np.exp(1j * 2 * np.pi * src_k * term.doppler / Q)
        H[rows.ravel(), cols.ravel()] += coeff.ravel()
    return H


def build_dense_heff_matrix(heff: np.ndarray, config: SystemConfig) -> np.ndarray:
    return build_dense_from_terms(nonzero_shift_terms(heff, config), config)


def build_dense_support_matrix(support_img: np.ndarray, config: SystemConfig) -> np.ndarray:
    return build_dense_from_terms(support_shift_terms(support_img, config), config)


def dense_apply(heff: np.ndarray, x: np.ndarray, config: SystemConfig) -> np.ndarray:
    H = build_dense_heff_matrix(heff, config)
    y = H @ x.reshape(-1)
    return y.reshape(config.M, config.N)
