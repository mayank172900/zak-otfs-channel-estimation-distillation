from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from zakotfs.dataset import simulate_frame
from zakotfs.params import SystemConfig
from zakotfs.utils import load_json, results_dir, save_json

from .baseline_bridge import baseline_estimate_support, load_frozen_baseline, resolve_baseline_checkpoint
from .physics_targets import forward_physics_target


COMPLEX_TENSORS: tuple[tuple[str, str], ...] = (
    ("H_obs", "h_obs"),
    ("H_true", "h_true"),
    ("H_thr", "h_thr"),
    ("H_base", "h_base"),
    ("H_phys_true", "h_phys_true"),
)

SCALAR_ARRAY_DTYPES: dict[str, np.dtype] = {
    "pdr_db": np.float32,
    "sample_index": np.int64,
    "sample_seed": np.int64,
    "data_snr_db": np.float32,
    "E_p": np.float32,
    "rho_d": np.float32,
    "rho_p": np.float32,
    "noise_variance": np.float32,
}

PHASE2_INPUT_CHANNEL_NAMES: tuple[str, ...] = (
    "H_obs_re",
    "H_obs_im",
    "H_thr_re",
    "H_thr_im",
    "H_base_re",
    "H_base_im",
)

PHASE2_TARGET_CHANNEL_NAMES: tuple[str, ...] = (
    "H_true_re",
    "H_true_im",
)


def _phase1_cfg(config: SystemConfig) -> dict:
    if "phase1_dataset" not in config.raw:
        raise ValueError("Phase 1 dataset config missing")
    return config.raw["phase1_dataset"]


def _split_seed(config: SystemConfig, split: str) -> int:
    offsets = {"train": 0, "val": 10_000, "test": 20_000, "eval": 30_000, "smoke": 40_000}
    return config.seed + offsets.get(split, 50_000)


def _storage_dtype(config: SystemConfig) -> np.dtype:
    dtype_name = str(_phase1_cfg(config).get("materialize_dtype", "float16")).lower()
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"Unsupported phase1_dataset.materialize_dtype '{dtype_name}'")


def _dataset_spec(config: SystemConfig) -> tuple[int, int, list[float], float, str, str, bool]:
    cfg = _phase1_cfg(config)
    total = int(cfg["total_size"])
    pdrs = list(map(float, cfg["training_pdr_db"]))
    if not pdrs:
        raise ValueError("phase1_dataset.training_pdr_db must not be empty")
    if total % len(pdrs) != 0:
        raise ValueError("phase1_dataset.total_size must be divisible by the number of PDR values")
    per_pdr = total // len(pdrs)
    snr = float(cfg["training_snr_db"])
    modulation = str(cfg.get("modulation", "bpsk"))
    split = str(cfg["split"])
    include_physics_target = bool(cfg.get("include_physics_target", True))
    return total, per_pdr, pdrs, snr, modulation, split, include_physics_target


def phase1_dataset_manifest(config: SystemConfig) -> dict:
    total, per_pdr, pdrs, snr, modulation, split, include_physics_target = _dataset_spec(config)
    seed = _split_seed(config, split)
    first = simulate_frame(config, modulation, snr, pdrs[0], np.random.default_rng(seed))
    shape = [int(first.support_input.shape[0]), int(first.support_input.shape[1])]
    tensor_names = [name for name, _ in COMPLEX_TENSORS if include_physics_target or name != "H_phys_true"]
    manifest = {
        "generator": "physics_phase1_memmap",
        "split": split,
        "mode": str(_phase1_cfg(config).get("mode", "full")),
        "seed": seed,
        "size": total,
        "per_pdr": per_pdr,
        "snr_db": snr,
        "pdr_db": pdrs,
        "shape": shape,
        "modulation": modulation,
        "include_physics_target": include_physics_target,
        "storage_dtype": np.dtype(_storage_dtype(config)).name,
        "tensor_names": tensor_names,
        "scalar_names": list(SCALAR_ARRAY_DTYPES.keys()),
        "baseline_checkpoint_path": str(resolve_baseline_checkpoint(config)),
        "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
    }
    return manifest


def _manifest_dir(config: SystemConfig) -> Path:
    out_dir = results_dir(config) / "phase1_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _manifest_path(config: SystemConfig) -> Path:
    split = str(_phase1_cfg(config)["split"])
    return _manifest_dir(config) / f"{split}_phase1.json"


def _resolve_meta_path(manifest_path: Path, value: str) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (manifest_path.parent / target).resolve()


def load_phase1_manifest(path: str | Path) -> dict:
    return load_json(Path(path))


def _required_artifact_keys(manifest: dict) -> list[str]:
    keys: list[str] = []
    include_physics_target = bool(manifest.get("include_physics_target", True))
    for tensor_name, stem in COMPLEX_TENSORS:
        if tensor_name == "H_phys_true" and not include_physics_target:
            continue
        keys.extend([f"{stem}_re_path", f"{stem}_im_path"])
    keys.extend(f"{name}_path" for name in SCALAR_ARRAY_DTYPES)
    return keys


def _artifacts_exist(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = load_phase1_manifest(manifest_path)
    return all(_resolve_meta_path(manifest_path, str(manifest[key])).exists() for key in _required_artifact_keys(manifest))


def _write_complex(memmaps: dict[str, np.memmap], stem: str, index: int, value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.complex64)
    memmaps[f"{stem}_re"][index] = arr.real.astype(memmaps[f"{stem}_re"].dtype)
    memmaps[f"{stem}_im"][index] = arr.imag.astype(memmaps[f"{stem}_im"].dtype)


def _write_scalar(memmaps: dict[str, np.memmap], name: str, index: int, value: float | int) -> None:
    memmaps[name][index] = np.asarray(value, dtype=memmaps[name].dtype)


def _open_writer_memmaps(manifest: dict, out_dir: Path) -> dict[str, np.memmap]:
    size = int(manifest["size"])
    H, W = map(int, manifest["shape"])
    dtype = np.dtype(str(manifest["storage_dtype"]))
    memmaps: dict[str, np.memmap] = {}
    include_physics_target = bool(manifest.get("include_physics_target", True))
    split = str(manifest["split"])
    for tensor_name, stem in COMPLEX_TENSORS:
        if tensor_name == "H_phys_true" and not include_physics_target:
            continue
        memmaps[f"{stem}_re"] = np.lib.format.open_memmap(out_dir / f"{split}_phase1_{stem}_re.npy", mode="w+", dtype=dtype, shape=(size, H, W))
        memmaps[f"{stem}_im"] = np.lib.format.open_memmap(out_dir / f"{split}_phase1_{stem}_im.npy", mode="w+", dtype=dtype, shape=(size, H, W))
    for name, dtype_value in SCALAR_ARRAY_DTYPES.items():
        memmaps[name] = np.lib.format.open_memmap(out_dir / f"{split}_phase1_{name}.npy", mode="w+", dtype=np.dtype(dtype_value), shape=(size,))
    return memmaps


def open_phase1_arrays(manifest_path: str | Path, mmap_mode: str = "r") -> dict[str, np.ndarray]:
    path = Path(manifest_path)
    manifest = load_phase1_manifest(path)
    arrays: dict[str, np.ndarray] = {}
    include_physics_target = bool(manifest.get("include_physics_target", True))
    for tensor_name, stem in COMPLEX_TENSORS:
        if tensor_name == "H_phys_true" and not include_physics_target:
            continue
        arrays[f"{stem}_re"] = np.load(_resolve_meta_path(path, str(manifest[f"{stem}_re_path"])), mmap_mode=mmap_mode)
        arrays[f"{stem}_im"] = np.load(_resolve_meta_path(path, str(manifest[f"{stem}_im_path"])), mmap_mode=mmap_mode)
    for name in SCALAR_ARRAY_DTYPES:
        arrays[name] = np.load(_resolve_meta_path(path, str(manifest[f"{name}_path"])), mmap_mode=mmap_mode)
    return arrays


class Phase1MemmapDataset:
    def __init__(self, manifest_path: str | Path, mmap_mode: str = "r") -> None:
        self.path = Path(manifest_path)
        self.meta = load_phase1_manifest(self.path)
        self.arrays = open_phase1_arrays(self.path, mmap_mode=mmap_mode)

    def __len__(self) -> int:
        return int(self.meta["size"])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | float | int]:
        sample: dict[str, np.ndarray | float | int] = {}
        include_physics_target = bool(self.meta.get("include_physics_target", True))
        for tensor_name, stem in COMPLEX_TENSORS:
            if tensor_name == "H_phys_true" and not include_physics_target:
                continue
            re = np.asarray(self.arrays[f"{stem}_re"][index], dtype=np.float32)
            im = np.asarray(self.arrays[f"{stem}_im"][index], dtype=np.float32)
            sample[tensor_name] = (re + 1j * im).astype(np.complex64)
        for name, dtype_value in SCALAR_ARRAY_DTYPES.items():
            scalar = self.arrays[name][index]
            if np.issubdtype(np.dtype(dtype_value), np.integer):
                sample[name] = int(scalar)
            else:
                sample[name] = float(scalar)
        return sample


def _stack_complex_parts(arrays: dict[str, np.ndarray], stem: str, index: int) -> np.ndarray:
    re = np.asarray(arrays[f"{stem}_re"][index], dtype=np.float32)
    im = np.asarray(arrays[f"{stem}_im"][index], dtype=np.float32)
    return np.stack([re, im], axis=0).copy()


class Phase2TrainingDataset(Dataset):
    def __init__(self, manifest_path: str | Path, mmap_mode: str = "r") -> None:
        self.path = Path(manifest_path)
        self.meta = load_phase1_manifest(self.path)
        self.arrays = open_phase1_arrays(self.path, mmap_mode=mmap_mode)
        self.input_channel_names = PHASE2_INPUT_CHANNEL_NAMES
        self.target_channel_names = PHASE2_TARGET_CHANNEL_NAMES
        self.include_physics_target = bool(self.meta.get("include_physics_target", True))

    def __len__(self) -> int:
        return int(self.meta["size"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h_obs = _stack_complex_parts(self.arrays, "h_obs", index)
        h_thr = _stack_complex_parts(self.arrays, "h_thr", index)
        h_base = _stack_complex_parts(self.arrays, "h_base", index)
        h_true = _stack_complex_parts(self.arrays, "h_true", index)
        sample = {
            "inputs": torch.from_numpy(np.concatenate([h_obs, h_thr, h_base], axis=0)),
            "h_true": torch.from_numpy(h_true),
            "h_base": torch.from_numpy(h_base),
            "pdr_db": torch.tensor(float(self.arrays["pdr_db"][index]), dtype=torch.float32),
            "sample_index": torch.tensor(int(self.arrays["sample_index"][index]), dtype=torch.int64),
            "sample_seed": torch.tensor(int(self.arrays["sample_seed"][index]), dtype=torch.int64),
        }
        if self.include_physics_target:
            sample["h_phys_true"] = torch.from_numpy(_stack_complex_parts(self.arrays, "h_phys_true", index))
        return sample


def generate_phase1_dataset(config: SystemConfig, force: bool = False) -> Path:
    manifest = phase1_dataset_manifest(config)
    out_dir = _manifest_dir(config)
    meta_path = _manifest_path(config)
    if _artifacts_exist(meta_path) and not force:
        print(f"[physics-phase1] reusing {meta_path}", flush=True)
        return meta_path

    size = int(manifest["size"])
    per_pdr = int(manifest["per_pdr"])
    pdrs = list(map(float, manifest["pdr_db"]))
    snr = float(manifest["snr_db"])
    modulation = str(manifest["modulation"])
    include_physics_target = bool(manifest["include_physics_target"])
    base_seed = int(manifest["seed"])
    backbone, device, checkpoint_path = load_frozen_baseline(config)
    memmaps = _open_writer_memmaps(manifest, out_dir)

    print(
        f"[physics-phase1] split={manifest['split']} size={size} per_pdr={per_pdr} device={device} "
        f"baseline={checkpoint_path}",
        flush=True,
    )
    offset = 0
    for pdr_db in pdrs:
        print(f"[physics-phase1] split={manifest['split']} pdr={pdr_db:.1f} samples={per_pdr}", flush=True)
        for local_idx in range(per_pdr):
            sample_seed = base_seed + offset
            frame = simulate_frame(config, modulation, snr, pdr_db, np.random.default_rng(sample_seed))
            h_base = baseline_estimate_support(backbone, frame.support_input)
            _write_complex(memmaps, "h_obs", offset, frame.support_input)
            _write_complex(memmaps, "h_true", offset, frame.support_true)
            _write_complex(memmaps, "h_thr", offset, frame.h_hat_support_thr)
            _write_complex(memmaps, "h_base", offset, h_base)
            if include_physics_target:
                h_phys_true = forward_physics_target(frame.support_true, frame.spread_dd, frame.E_p, config)
                _write_complex(memmaps, "h_phys_true", offset, h_phys_true)
            _write_scalar(memmaps, "pdr_db", offset, pdr_db)
            _write_scalar(memmaps, "sample_index", offset, offset)
            _write_scalar(memmaps, "sample_seed", offset, sample_seed)
            _write_scalar(memmaps, "data_snr_db", offset, snr)
            _write_scalar(memmaps, "E_p", offset, frame.E_p)
            _write_scalar(memmaps, "rho_d", offset, frame.rho_d)
            _write_scalar(memmaps, "rho_p", offset, frame.rho_p)
            _write_scalar(memmaps, "noise_variance", offset, frame.noise_variance)
            offset += 1
            if local_idx == 0 or (local_idx + 1) == per_pdr:
                print(
                    f"[physics-phase1] split={manifest['split']} pdr={pdr_db:.1f} "
                    f"progress={local_idx + 1}/{per_pdr} total={offset}/{size}",
                    flush=True,
                )

    for name, memmap in memmaps.items():
        memmap.flush()
        if name in SCALAR_ARRAY_DTYPES:
            manifest[f"{name}_path"] = Path(memmap.filename).name
        else:
            manifest[f"{name}_path"] = Path(memmap.filename).name
    save_json(meta_path, manifest)
    print(f"[physics-phase1] saved manifest {meta_path}", flush=True)
    return meta_path
