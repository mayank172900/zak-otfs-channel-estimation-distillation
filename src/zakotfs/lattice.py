from __future__ import annotations

import numpy as np

from .compat import dataclass_slots
from .params import SystemConfig
from .utils import centered_coord_to_index


@dataclass_slots()
class SupportGeometry:
    delta_k: int
    delta_l: int
    k_min: int
    k_max: int
    l_min: int
    l_max: int
    basis_delay: tuple[int, int]
    basis_doppler: tuple[int, int]


def lattice_condition(k: int, l: int, config: SystemConfig) -> bool:
    M, N, q = config.M, config.N, config.q
    mod_inv = pow(2 * q, -1, M * N)
    return ((2 * q * k - l) % M == 0) and ((k - l * (mod_inv - 2 * q)) % N == 0)


def enumerate_lattice_points(config: SystemConfig, radius_k: int = 64, radius_l: int = 96) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for k in range(-radius_k, radius_k + 1):
        for l in range(-radius_l, radius_l + 1):
            if lattice_condition(k, l, config):
                pts.append((k, l))
    return pts


def derive_support_geometry(config: SystemConfig) -> SupportGeometry:
    points = [pt for pt in enumerate_lattice_points(config) if pt != (0, 0)]
    basis_doppler = min((pt for pt in points if pt[1] > 0), key=lambda pt: (abs(pt[0]), pt[1]))
    independent = [pt for pt in points if (basis_doppler[0] * pt[1] - basis_doppler[1] * pt[0]) != 0]
    basis_delay = min((pt for pt in independent if pt[0] > 0), key=lambda pt: (abs(pt[1]), pt[0]))
    delta_k = abs(basis_delay[0])
    delta_l = abs(basis_doppler[1])
    k_max = delta_k // 2
    l_max = delta_l // 2
    return SupportGeometry(
        delta_k=delta_k,
        delta_l=delta_l,
        k_min=-k_max,
        k_max=k_max,
        l_min=-l_max,
        l_max=l_max,
        basis_delay=basis_delay,
        basis_doppler=basis_doppler,
    )


def support_shape(config: SystemConfig) -> tuple[int, int]:
    g = derive_support_geometry(config)
    return g.delta_k, g.delta_l


def support_coords(config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    g = derive_support_geometry(config)
    k = np.arange(g.k_min, g.k_max + 1, dtype=int)
    l = np.arange(g.l_min, g.l_max + 1, dtype=int)
    return np.meshgrid(k, l, indexing="ij")


def support_mask(config: SystemConfig) -> np.ndarray:
    mask = np.zeros((config.M, config.N), dtype=np.uint8)
    g = derive_support_geometry(config)
    for k in range(g.k_min, g.k_max + 1):
        for l in range(g.l_min, g.l_max + 1):
            mask[centered_coord_to_index(k, config.M), centered_coord_to_index(l, config.N)] = 1
    return mask


def crop_support(arr: np.ndarray, config: SystemConfig) -> np.ndarray:
    g = derive_support_geometry(config)
    out = np.zeros((g.delta_k, g.delta_l), dtype=arr.dtype)
    for i, k in enumerate(range(g.k_min, g.k_max + 1)):
        for j, l in enumerate(range(g.l_min, g.l_max + 1)):
            out[i, j] = arr[centered_coord_to_index(k, config.M), centered_coord_to_index(l, config.N)]
    return out


def embed_support_image(support_img: np.ndarray, config: SystemConfig) -> np.ndarray:
    full = np.zeros((config.M, config.N), dtype=support_img.dtype)
    g = derive_support_geometry(config)
    for i, k in enumerate(range(g.k_min, g.k_max + 1)):
        for j, l in enumerate(range(g.l_min, g.l_max + 1)):
            full[centered_coord_to_index(k, config.M), centered_coord_to_index(l, config.N)] = support_img[i, j]
    return full
