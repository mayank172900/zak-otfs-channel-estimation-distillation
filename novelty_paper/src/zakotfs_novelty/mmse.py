from __future__ import annotations

import numpy as np
import torch

from .operators import (
    apply_heff_adjoint,
    apply_heff_operator,
    apply_support_adjoint,
    apply_support_operator,
    build_dense_heff_matrix,
    build_dense_support_matrix,
)
from .params import SystemConfig


def _dense_matrix(channel_estimate: np.ndarray, config: SystemConfig) -> np.ndarray:
    return (
        build_dense_heff_matrix(channel_estimate, config)
        if channel_estimate.shape == (config.M, config.N)
        else build_dense_support_matrix(channel_estimate, config)
    )


def mmse_ridge_lambda(noise_variance: float, E_d: float, config: SystemConfig) -> float:
    data_variance = float(E_d) / float(config.Q)
    if data_variance <= 0.0:
        raise ValueError("E_d must be positive for MMSE detection")
    return float(noise_variance) / data_variance


def mmse_dense(
    y: np.ndarray,
    heff: np.ndarray,
    noise_variance: float,
    config: SystemConfig,
    E_d: float = 1.0,
) -> np.ndarray:
    H = _dense_matrix(heff, config)
    ridge = mmse_ridge_lambda(noise_variance, E_d, config)
    A = H.conj().T @ H + ridge * np.eye(config.Q, dtype=np.complex64)
    b = H.conj().T @ y.reshape(-1)
    x = np.linalg.solve(A, b)
    return x.reshape(config.M, config.N)


def mmse_dense_torch(
    y: np.ndarray,
    heff: np.ndarray,
    noise_variance: float,
    config: SystemConfig,
    E_d: float = 1.0,
    device: torch.device | str | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    H = torch.from_numpy(_dense_matrix(heff, config)).to(device=device, dtype=torch.complex64)
    y_vec = torch.from_numpy(y.reshape(-1)).to(device=device, dtype=torch.complex64)
    ridge = mmse_ridge_lambda(noise_variance, E_d, config)
    A = H.conj().transpose(0, 1) @ H + ridge * torch.eye(config.Q, device=device, dtype=torch.complex64)
    b = H.conj().transpose(0, 1) @ y_vec
    x = torch.linalg.solve(A, b)
    return x.detach().cpu().numpy().reshape(config.M, config.N)


def cg_solve(operator, rhs: np.ndarray, tol: float, maxiter: int) -> np.ndarray:
    x = np.zeros_like(rhs)
    r = rhs - operator(x)
    p = r.copy()
    rs_old = np.vdot(r, r)
    if not np.isfinite(rs_old) or np.real(rs_old) <= 0.0:
        return x
    for _ in range(maxiter):
        Ap = operator(p)
        denom = np.vdot(p, Ap)
        if not np.isfinite(denom) or abs(denom) < 1e-20:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.vdot(r, r)
        if not np.isfinite(rs_new):
            break
        if np.sqrt(np.real(rs_new)) < tol:
            break
        if abs(rs_old) < 1e-20:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def mmse_iterative(
    y: np.ndarray,
    heff: np.ndarray,
    noise_variance: float,
    config: SystemConfig,
    E_d: float = 1.0,
) -> np.ndarray:
    forward = apply_heff_operator if heff.shape == (config.M, config.N) else apply_support_operator
    adjoint = apply_heff_adjoint if heff.shape == (config.M, config.N) else apply_support_adjoint
    rhs = adjoint(heff, y, config)
    ridge = mmse_ridge_lambda(noise_variance, E_d, config)

    def normal_op(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape(config.M, config.N)
        hx = forward(heff, x, config)
        ahx = adjoint(heff, hx, config)
        return (ahx + ridge * x).reshape(-1)

    maxiter = int(config.detection["cg_maxiter"])
    x = cg_solve(normal_op, rhs.reshape(-1), float(config.detection["cg_tol"]), maxiter)
    return x.reshape(config.M, config.N)
