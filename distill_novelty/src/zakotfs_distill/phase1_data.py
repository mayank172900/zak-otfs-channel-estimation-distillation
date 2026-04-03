from __future__ import annotations

from pathlib import Path

import numpy as np

from zakotfs.utils import load_json


SCALAR_ARRAY_NAMES: tuple[str, ...] = (
    "pdr_db",
    "sample_index",
    "sample_seed",
    "data_snr_db",
    "E_p",
    "rho_d",
    "rho_p",
    "noise_variance",
)

COMPLEX_STEMS: tuple[str, ...] = (
    "h_obs",
    "h_true",
    "h_thr",
    "h_base",
    "h_phys_true",
)


def load_phase1_manifest(path: str | Path) -> dict:
    return load_json(Path(path))


def _resolve_meta_path(manifest_path: Path, value: str) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (manifest_path.parent / target).resolve()


def open_phase1_arrays(manifest_path: str | Path, mmap_mode: str = "r") -> dict[str, np.ndarray]:
    path = Path(manifest_path)
    manifest = load_phase1_manifest(path)
    arrays: dict[str, np.ndarray] = {}
    include_physics_target = bool(manifest.get("include_physics_target", True))

    for stem in COMPLEX_STEMS:
        if stem == "h_phys_true" and not include_physics_target:
            continue
        if f"{stem}_re_path" not in manifest or f"{stem}_im_path" not in manifest:
            continue
        arrays[f"{stem}_re"] = np.load(_resolve_meta_path(path, str(manifest[f"{stem}_re_path"])), mmap_mode=mmap_mode)
        arrays[f"{stem}_im"] = np.load(_resolve_meta_path(path, str(manifest[f"{stem}_im_path"])), mmap_mode=mmap_mode)

    for name in SCALAR_ARRAY_NAMES:
        if f"{name}_path" not in manifest:
            continue
        arrays[name] = np.load(_resolve_meta_path(path, str(manifest[f"{name}_path"])), mmap_mode=mmap_mode)
    return arrays
