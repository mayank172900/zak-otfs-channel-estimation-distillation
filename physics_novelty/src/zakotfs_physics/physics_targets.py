from __future__ import annotations

from typing import Any

import numpy as np
import torch

from zakotfs.estimators import read_off_estimator
from zakotfs.lattice import derive_support_geometry
from zakotfs.operators import apply_support_operator
from zakotfs.params import SystemConfig
from zakotfs.waveform import spread_pilot


_PHYSICS_MATRIX_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_PHYSICS_TORCH_CACHE: dict[tuple[tuple[Any, ...], str], torch.Tensor] = {}


def _physics_cache_key(config: SystemConfig) -> tuple[Any, ...]:
    g = derive_support_geometry(config)
    return (
        int(config.M),
        int(config.N),
        int(config.q),
        int(config.frame["pilot_delay_bin"]),
        int(config.frame["pilot_doppler_bin"]),
        int(g.k_min),
        int(g.k_max),
        int(g.l_min),
        int(g.l_max),
    )


def _support_coords(config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    g = derive_support_geometry(config)
    return (
        np.arange(g.k_min, g.k_max + 1, dtype=int),
        np.arange(g.l_min, g.l_max + 1, dtype=int),
    )


def _complex_channels_to_tensor(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if torch.is_complex(value):
        return (value.unsqueeze(0), True) if value.ndim == 2 else (value, False)
    if value.ndim == 3:
        if value.shape[0] != 2:
            raise ValueError("Expected a 2-channel tensor with shape (2, H, W)")
        complex_value = torch.complex(value[0], value[1]).unsqueeze(0)
        return complex_value, True
    if value.ndim == 4:
        if value.shape[1] != 2:
            raise ValueError("Expected a 2-channel batched tensor with shape (B, 2, H, W)")
        complex_value = torch.complex(value[:, 0], value[:, 1])
        return complex_value, False
    raise ValueError(f"Unsupported tensor shape for physics operator: {tuple(value.shape)}")


def _complex_tensor_to_channels(value: torch.Tensor, single: bool) -> torch.Tensor:
    channels = torch.stack([value.real, value.imag], dim=1)
    return channels[0] if single else channels


def physics_operator_matrix(config: SystemConfig) -> np.ndarray:
    """Return the exact linear map for G(H) on the support window.

    The matrix is built once from the baseline Zak-OTFS equations and cached.
    It maps the flattened support-domain channel image to the flattened
    synthesized pilot/read-off support image.
    """

    key = _physics_cache_key(config)
    cached = _PHYSICS_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    M, N, Q = config.M, config.N, config.Q
    xs = spread_pilot(config).astype(np.complex64)
    delay_coords, doppler_coords = _support_coords(config)
    kp = np.arange(M, dtype=int)[:, None]
    lp = np.arange(N, dtype=int)[None, :]
    support_size = int(delay_coords.size * doppler_coords.size)
    basis_waveforms = np.empty((support_size, M, N), dtype=np.complex64)
    readout_kernels = np.empty((support_size, M, N), dtype=np.complex64)

    basis_index = 0
    for delay in delay_coords:
        src_k = kp - delay
        base_k = np.mod(src_k, M)
        wrap_k = np.floor_divide(src_k - base_k, M)
        for doppler in doppler_coords:
            src_l = lp - doppler
            base_l = np.mod(src_l, N)
            phase_qp = np.exp(1j * 2 * np.pi * wrap_k * base_l / N).astype(np.complex64)
            shifted = xs[base_k, base_l] * phase_qp
            phase = np.exp(1j * 2 * np.pi * src_k * doppler / Q).astype(np.complex64)
            basis_waveforms[basis_index] = (shifted * phase).astype(np.complex64)
            basis_index += 1

    readout_index = 0
    for delay in delay_coords:
        shifted_k = kp - delay
        base_k = np.mod(shifted_k, M)
        wrap_k = np.floor_divide(shifted_k - base_k, M)
        for doppler in doppler_coords:
            shifted_l = lp - doppler
            base_l = np.mod(shifted_l, N)
            phase_qp = np.exp(1j * 2 * np.pi * wrap_k * base_l / N).astype(np.complex64)
            b_shifted = xs[base_k, base_l] * phase_qp
            phase = np.exp(-1j * 2 * np.pi * doppler * shifted_k / Q).astype(np.complex64)
            readout_kernels[readout_index] = (np.conjugate(b_shifted) * phase).astype(np.complex64)
            readout_index += 1

    matrix = np.einsum("tmn,omn->ot", basis_waveforms, readout_kernels, optimize=True).astype(np.complex64)
    _PHYSICS_MATRIX_CACHE[key] = matrix
    return matrix


def _physics_operator_matrix_torch(config: SystemConfig, device: torch.device) -> torch.Tensor:
    key = (_physics_cache_key(config), str(device))
    cached = _PHYSICS_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    matrix = torch.from_numpy(physics_operator_matrix(config)).to(device=device, dtype=torch.complex64)
    _PHYSICS_TORCH_CACHE[key] = matrix
    return matrix


def forward_physics_target_torch(h_support: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    """Apply the training-time G(H) operator to one sample or a batch.

    This is a hybrid implementation: the exact linear map is precomputed once
    from the baseline NumPy equations and then applied in Torch as a fixed
    complex matrix. Gradients flow to `h_support` through the Torch matmul.
    """

    complex_support, single = _complex_channels_to_tensor(h_support)
    batch, H, W = complex_support.shape
    matrix = _physics_operator_matrix_torch(config, complex_support.device)
    out = complex_support.reshape(batch, H * W) @ matrix.transpose(0, 1)
    out = out.reshape(batch, H, W)
    if torch.is_complex(h_support):
        return out[0] if single else out
    return _complex_tensor_to_channels(out, single)


def forward_physics_target(h_support: np.ndarray, spread_dd: np.ndarray, E_p: float, config: SystemConfig) -> np.ndarray:
    pilot_waveform = np.asarray(np.sqrt(np.float32(E_p)) * spread_dd, dtype=np.complex64)
    y_syn = apply_support_operator(np.asarray(h_support, dtype=np.complex64), pilot_waveform, config)
    return np.asarray(read_off_estimator(y_syn, spread_dd, E_p, config).support_input, dtype=np.complex64)
