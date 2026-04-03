from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cnn_model import PaperCNN
from .dataset import _split_seed, simulate_frame
from .estimators import read_off_estimator
from .operators import apply_support_operator
from .params import SystemConfig, load_config
from .utils import load_json, resolve_torch_device, results_dir, save_json
from .waveform import spread_pilot


FEATURE_CHANNEL_NAMES = (
    "raw_re",
    "raw_im",
    "base_re",
    "base_im",
    "alias_re",
    "alias_im",
    "raw_minus_base_re",
    "raw_minus_base_im",
)


def load_frozen_backbone(config: SystemConfig, checkpoint_path: Path | None = None) -> tuple[PaperCNN, torch.device]:
    if checkpoint_path is None:
        checkpoint_cfg = config.raw.get("backbone", {})
        checkpoint_path = Path(str(checkpoint_cfg.get("checkpoint_path", "../logs/checkpoints/full_cnn_best.pt")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config.root / checkpoint_path).resolve()
    device = resolve_torch_device(config)
    model = PaperCNN().to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, device


def backbone_enhance_support(model: PaperCNN, support_input: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_re = torch.from_numpy(np.asarray(support_input.real[None, None, :, :], dtype=np.float32)).to(device)
        x_im = torch.from_numpy(np.asarray(support_input.imag[None, None, :, :], dtype=np.float32)).to(device)
        y_re = model(x_re).detach().cpu().numpy()[0, 0]
        y_im = model(x_im).detach().cpu().numpy()[0, 0]
    return (y_re + 1j * y_im).astype(np.complex64)


def synthesize_alias_prior(h_base: np.ndarray, spread_dd: np.ndarray, E_p: float, config: SystemConfig) -> np.ndarray:
    y_syn = apply_support_operator(h_base, np.sqrt(E_p) * spread_dd, config)
    h_syn = read_off_estimator(y_syn, spread_dd, E_p, config).support_input
    return (h_syn - h_base).astype(np.complex64)


def build_fb_lara_features(h_raw: np.ndarray, h_base: np.ndarray, h_alias: np.ndarray) -> np.ndarray:
    residual = h_raw - h_base
    return np.stack(
        [
            h_raw.real,
            h_raw.imag,
            h_base.real,
            h_base.imag,
            h_alias.real,
            h_alias.imag,
            residual.real,
            residual.imag,
        ],
        axis=0,
    ).astype(np.float32)


def residual_target(h_true: np.ndarray, h_base: np.ndarray) -> np.ndarray:
    delta = h_true - h_base
    return np.stack([delta.real, delta.imag], axis=0).astype(np.float32)


def feature_channel_indices(adapter_kind: str) -> tuple[int, ...]:
    kind = str(adapter_kind).lower()
    if kind == "fb_lara":
        return tuple(range(8))
    if kind == "generic":
        return (0, 1, 2, 3, 6, 7)
    raise ValueError(f"Unknown adapter kind '{adapter_kind}'")


def adapter_dataset_sizes(config: SystemConfig, split: str) -> tuple[int, list[float], float]:
    dataset_cfg = config.raw["adapter_dataset"]
    total = int(dataset_cfg["train_size_total"] if split == "train" else dataset_cfg["val_size_total"])
    pdrs = list(map(float, dataset_cfg["training_pdr_db"]))
    snr = float(dataset_cfg["training_snr_db"])
    return total, pdrs, snr


def adapter_dataset_manifest(config: SystemConfig, split: str) -> dict:
    total, pdrs, snr = adapter_dataset_sizes(config, split)
    seed = _split_seed(config, split)
    first = simulate_frame(config, "bpsk", snr, pdrs[0], np.random.default_rng(seed))
    return {
        "generator": "memmap_fp16",
        "split": split,
        "seed": seed,
        "size": int(total),
        "snr_db": snr,
        "pdr_db": pdrs,
        "per_pdr": int(total // len(pdrs)),
        "shape": [int(first.support_input.shape[0]), int(first.support_input.shape[1])],
        "feature_channels": list(FEATURE_CHANNEL_NAMES),
        "target_kind": "residual",
        "backbone_checkpoint_path": str(config.raw.get("backbone", {}).get("checkpoint_path", "../logs/checkpoints/full_cnn_best.pt")),
    }


def _resolve_external_path(base: Path, value: str) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (base / target).resolve()


def _baseline_manifest_path(config: SystemConfig, split: str) -> Path | None:
    baseline_paths = config.raw.get("adapter_dataset", {}).get("baseline_manifest_paths", {}) or {}
    candidate = baseline_paths.get(split)
    if not candidate:
        return None
    path = _resolve_external_path(config.root, str(candidate))
    return path if path.exists() else None


def _load_baseline_support_data(
    manifest_path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = load_json(manifest_path)
    if str(meta.get("generator", "")) != "memmap_fp16":
        raise ValueError(f"Baseline manifest {manifest_path} is not a memmap_fp16 dataset")
    inputs_re = np.load(_resolve_external_path(manifest_path.parent, str(meta["inputs_re_path"])), mmap_mode="r")
    inputs_im = np.load(_resolve_external_path(manifest_path.parent, str(meta["inputs_im_path"])), mmap_mode="r")
    targets_re = np.load(_resolve_external_path(manifest_path.parent, str(meta["targets_re_path"])), mmap_mode="r")
    targets_im = np.load(_resolve_external_path(manifest_path.parent, str(meta["targets_im_path"])), mmap_mode="r")
    label_key = "labels_path" if "labels_path" in meta else "pdr_labels_path"
    labels = np.load(_resolve_external_path(manifest_path.parent, str(meta[label_key])), mmap_mode="r")
    return meta, inputs_re, inputs_im, targets_re, targets_im, labels


class AdapterFeatureDataset(Dataset):
    def __init__(self, manifest_path: Path, adapter_kind: str) -> None:
        self.path = Path(manifest_path)
        self.meta = load_json(self.path)
        self.adapter_kind = str(adapter_kind).lower()
        self.channel_indices = np.array(feature_channel_indices(self.adapter_kind), dtype=np.int64)
        load_mode = "r"
        self.inputs = np.load(self._resolve("inputs_path"), mmap_mode=load_mode)
        self.targets = np.load(self._resolve("targets_path"), mmap_mode=load_mode)

    def _resolve(self, key: str) -> str:
        target = Path(str(self.meta[key]))
        if target.is_absolute():
            return str(target)
        return str((self.path.parent / target).resolve())

    def __len__(self) -> int:
        return int(self.meta["size"])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.inputs[index], dtype=np.float32)[self.channel_indices].copy()
        y = np.asarray(self.targets[index], dtype=np.float32).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def generate_adapter_dataset(config: SystemConfig, split: str, force: bool = False) -> Path:
    out_dir = results_dir(config) / "adapter_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{split}_adapter.json"
    if meta_path.exists() and not force:
        meta = load_json(meta_path)
        required = []
        for key in ["inputs_path", "targets_path", "pdr_labels_path"]:
            candidate = Path(str(meta[key]))
            if not candidate.is_absolute():
                candidate = (meta_path.parent / candidate).resolve()
            required.append(candidate)
        if all(path.exists() for path in required):
            print(f"[adapter-dataset] reusing {meta_path}", flush=True)
            return meta_path
    manifest = adapter_dataset_manifest(config, split)
    size = int(manifest["size"])
    H, W = map(int, manifest["shape"])
    per_pdr = int(manifest["per_pdr"])
    inputs_path = out_dir / f"{split}_adapter_inputs.npy"
    targets_path = out_dir / f"{split}_adapter_targets.npy"
    pdr_labels_path = out_dir / f"{split}_adapter_pdr.npy"
    inputs = np.lib.format.open_memmap(inputs_path, mode="w+", dtype=np.float16, shape=(size, 8, H, W))
    targets = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.float16, shape=(size, 2, H, W))
    labels = np.lib.format.open_memmap(pdr_labels_path, mode="w+", dtype=np.float32, shape=(size,))
    backbone, device = load_frozen_backbone(config)
    baseline_manifest = _baseline_manifest_path(config, split)
    if baseline_manifest is not None:
        base_meta, inputs_re, inputs_im, targets_re, targets_im, pdr_all = _load_baseline_support_data(baseline_manifest)
        if int(base_meta["size"]) < size:
            raise ValueError(f"Baseline manifest {baseline_manifest} has only {base_meta['size']} samples, expected at least {size}")
        spread_dd = spread_pilot(config)
        print(
            f"[adapter-dataset] split={split} using baseline manifest {baseline_manifest} size={size} device={device}",
            flush=True,
        )
        for offset in range(size):
            h_raw = np.asarray(inputs_re[offset], dtype=np.float32) + 1j * np.asarray(inputs_im[offset], dtype=np.float32)
            h_true = np.asarray(targets_re[offset], dtype=np.float32) + 1j * np.asarray(targets_im[offset], dtype=np.float32)
            pdr_db = float(pdr_all[offset])
            E_p = float(10.0 ** (pdr_db / 10.0))
            h_base = backbone_enhance_support(backbone, h_raw.astype(np.complex64), device)
            h_alias = synthesize_alias_prior(h_base, spread_dd, E_p, config)
            inputs[offset] = build_fb_lara_features(h_raw.astype(np.complex64), h_base, h_alias).astype(np.float16)
            targets[offset] = residual_target(h_true.astype(np.complex64), h_base).astype(np.float16)
            labels[offset] = np.float32(pdr_db)
            if offset == 0 or (offset + 1) % 1000 == 0 or (offset + 1) == size:
                print(f"[adapter-dataset] split={split} baseline progress={offset + 1}/{size}", flush=True)
    else:
        offset = 0
        total, pdrs, snr = adapter_dataset_sizes(config, split)
        if total != size:
            raise RuntimeError("Adapter dataset size mismatch")
        for pdr_db in pdrs:
            print(
                f"[adapter-dataset] split={split} pdr={pdr_db:.1f} samples={per_pdr} device={device}",
                flush=True,
            )
            for local_idx in range(per_pdr):
                sample_seed = int(manifest["seed"]) + int(offset)
                frame = simulate_frame(config, "bpsk", snr, pdr_db, np.random.default_rng(sample_seed))
                h_base = backbone_enhance_support(backbone, frame.support_input, device)
                h_alias = synthesize_alias_prior(h_base, frame.spread_dd, frame.E_p, config)
                inputs[offset] = build_fb_lara_features(frame.support_input, h_base, h_alias).astype(np.float16)
                targets[offset] = residual_target(frame.support_true, h_base).astype(np.float16)
                labels[offset] = np.float32(pdr_db)
                offset += 1
                if local_idx == 0 or (local_idx + 1) % 1000 == 0 or (local_idx + 1) == per_pdr:
                    print(
                        f"[adapter-dataset] split={split} pdr={pdr_db:.1f} progress={local_idx + 1}/{per_pdr} total={offset}/{size}",
                        flush=True,
                    )
    inputs.flush()
    targets.flush()
    labels.flush()
    materialized = dict(manifest)
    materialized["inputs_path"] = inputs_path.name
    materialized["targets_path"] = targets_path.name
    materialized["pdr_labels_path"] = pdr_labels_path.name
    materialized["baseline_manifest_path"] = str(baseline_manifest) if baseline_manifest is not None else None
    save_json(meta_path, materialized)
    print(f"[adapter-dataset] saved manifest {meta_path}", flush=True)
    return meta_path


def load_adapter_train_config(path: Path | None = None) -> SystemConfig:
    resolved = Path("novelty_paper/configs/adapter_train.yaml") if path is None else Path(path)
    return load_config(resolved)
