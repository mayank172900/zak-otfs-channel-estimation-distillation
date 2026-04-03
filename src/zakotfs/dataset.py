from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .compat import dataclass_slots
from .channel import add_awgn, effective_channel_method, effective_channel_support, sample_vehicular_a_channel
from .estimators import read_off_estimator, threshold_readoff
from .lattice import crop_support, embed_support_image
from .modulation import hard_demodulate
from .params import SystemConfig
from .utils import results_dir, save_json, set_global_seed, snr_to_noise_variance
from .waveform import data_symbols, data_waveform, spread_pilot, superimposed_frame
from .operators import apply_heff_operator, apply_support_operator


@dataclass_slots()
class FrameBundle:
    modulation: str
    symbols: np.ndarray
    bits: np.ndarray
    data_dd: np.ndarray
    spread_dd: np.ndarray
    x_dd: np.ndarray
    physical_channel: object
    h_eff_support: np.ndarray
    h_eff: np.ndarray
    y_clean: np.ndarray
    y_dd: np.ndarray
    ambiguity: np.ndarray
    h_hat_support_raw: np.ndarray
    h_hat_support_thr: np.ndarray
    h_hat_raw: np.ndarray
    h_hat_thr: np.ndarray
    support_input: np.ndarray
    support_true: np.ndarray
    E_d: float
    E_p: float
    rho_d: float
    rho_p: float
    noise_variance: float


def _split_seed(config: SystemConfig, split: str) -> int:
    offsets = {"train": 0, "val": 10_000, "test": 20_000, "eval": 30_000, "smoke": 40_000}
    return config.seed + offsets.get(split, 50_000)


def simulate_frame(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    rng: np.random.Generator,
) -> FrameBundle:
    symbols, bits = data_symbols(modulation, config, rng)
    data_dd = data_waveform(symbols)
    spread_dd = spread_pilot(config)
    E_d = 1.0
    E_p = 10.0 ** (pdr_db / 10.0) * E_d
    noise_variance = snr_to_noise_variance(E_d, data_snr_db, config.Q)
    rho_d = E_d / (config.Q * noise_variance)
    rho_p = E_p / (config.Q * noise_variance)
    x_dd = superimposed_frame(data_dd, spread_dd, E_d=E_d, E_p=E_p)
    physical_channel = sample_vehicular_a_channel(config, rng)
    h_eff_support = effective_channel_support(physical_channel, config)
    h_eff = embed_support_image(h_eff_support, config)
    y_clean = apply_support_operator(h_eff_support, x_dd, config)
    y_dd = add_awgn(y_clean, noise_variance, rng)
    est = read_off_estimator(y_dd, spread_dd, E_p, config)
    h_hat_support_raw = est.support_input
    h_hat_support_thr = threshold_readoff(h_hat_support_raw, rho_d, rho_p, config)
    h_hat_raw = embed_support_image(h_hat_support_raw, config)
    h_hat_thr = embed_support_image(h_hat_support_thr, config)
    support_input = h_hat_support_raw
    support_true = h_eff_support
    return FrameBundle(
        modulation=modulation,
        symbols=symbols,
        bits=bits,
        data_dd=data_dd,
        spread_dd=spread_dd,
        x_dd=x_dd,
        physical_channel=physical_channel,
        h_eff_support=h_eff_support,
        h_eff=h_eff,
        y_clean=y_clean,
        y_dd=y_dd,
        ambiguity=est.ambiguity,
        h_hat_support_raw=h_hat_support_raw,
        h_hat_support_thr=h_hat_support_thr,
        h_hat_raw=h_hat_raw,
        h_hat_thr=h_hat_thr,
        support_input=support_input,
        support_true=support_true,
        E_d=E_d,
        E_p=E_p,
        rho_d=rho_d,
        rho_p=rho_p,
        noise_variance=noise_variance,
    )


def dataset_sizes(config: SystemConfig, split: str, mode: str) -> tuple[int, list[float], float]:
    if mode == "smoke":
        dataset_cfg = config.raw.get("smoke", {}).get("dataset", {})
    else:
        dataset_cfg = config.raw["dataset"]
    total = int(dataset_cfg["train_size_total"] if split == "train" else dataset_cfg["val_size_total"])
    pdrs = list(map(float, config.raw["dataset"]["training_pdr_db"]))
    snr = float(config.raw["dataset"]["training_snr_db"])
    return total, pdrs, snr


def dataset_manifest(config: SystemConfig, split: str, mode: str) -> dict:
    total, pdrs, snr = dataset_sizes(config, split, mode)
    seed = _split_seed(config, split) + (0 if mode == "full" else 100_000)
    per_pdr = total // len(pdrs)
    first_rng = np.random.default_rng(seed)
    first = simulate_frame(config, "bpsk", snr, pdrs[0], first_rng)
    return {
        "generator": "on_the_fly" if mode == "full" else "npz",
        "split": split,
        "mode": mode,
        "seed": seed,
        "size": int(per_pdr * len(pdrs)),
        "snr_db": snr,
        "pdr_db": pdrs,
        "shape": [int(first.support_input.shape[0]), int(first.support_input.shape[1])],
        "per_pdr": int(per_pdr),
        "modulation": "bpsk",
        "effective_channel_method": effective_channel_method(config),
    }


class GeneratedSupportDataset(Dataset):
    def __init__(self, config: SystemConfig, manifest: dict) -> None:
        self.config = config
        self.meta = manifest

    def __len__(self) -> int:
        return int(self.meta["size"])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.float32]:
        pdrs = list(map(float, self.meta["pdr_db"]))
        per_pdr = int(self.meta["per_pdr"])
        pdr_idx = min(index // per_pdr, len(pdrs) - 1)
        pdr_db = pdrs[pdr_idx]
        sample_seed = int(self.meta["seed"]) + int(index)
        frame = simulate_frame(
            self.config,
            str(self.meta.get("modulation", "bpsk")),
            float(self.meta["snr_db"]),
            pdr_db,
            np.random.default_rng(sample_seed),
        )
        return frame.support_input.astype(np.complex64), frame.support_true.astype(np.complex64), np.float32(pdr_db)


def _materialize_full_dataset(config: SystemConfig, split: str, manifest: dict, out_dir: Path) -> Path:
    meta_path = out_dir / f"{split}_full.json"
    inputs_re_path = out_dir / f"{split}_full_inputs_re.npy"
    inputs_im_path = out_dir / f"{split}_full_inputs_im.npy"
    targets_re_path = out_dir / f"{split}_full_targets_re.npy"
    targets_im_path = out_dir / f"{split}_full_targets_im.npy"
    labels_path = out_dir / f"{split}_full_pdr.npy"
    size = int(manifest["size"])
    H, W = map(int, manifest["shape"])
    batch_size = int(config.raw["dataset"].get("materialize_batch_size", 128))
    num_workers = int(config.raw["dataset"].get("materialize_num_workers", config.raw["dataset"].get("num_workers", 0)))
    print(
        f"[dataset] materializing split={split} samples={size} shape=({H},{W}) "
        f"batch_size={batch_size} num_workers={num_workers}",
        flush=True,
    )
    ds = GeneratedSupportDataset(config, manifest)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    inputs_re = np.lib.format.open_memmap(inputs_re_path, mode="w+", dtype=np.float16, shape=(size, H, W))
    inputs_im = np.lib.format.open_memmap(inputs_im_path, mode="w+", dtype=np.float16, shape=(size, H, W))
    targets_re = np.lib.format.open_memmap(targets_re_path, mode="w+", dtype=np.float16, shape=(size, H, W))
    targets_im = np.lib.format.open_memmap(targets_im_path, mode="w+", dtype=np.float16, shape=(size, H, W))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.float32, shape=(size,))
    offset = 0
    total_batches = len(loader)
    for batch_idx, (x, y, pdr) in enumerate(loader, start=1):
        batch = int(x.shape[0])
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        labels[offset : offset + batch] = pdr.detach().cpu().numpy().astype(np.float32)
        inputs_re[offset : offset + batch] = x_np.real.astype(np.float16)
        inputs_im[offset : offset + batch] = x_np.imag.astype(np.float16)
        targets_re[offset : offset + batch] = y_np.real.astype(np.float16)
        targets_im[offset : offset + batch] = y_np.imag.astype(np.float16)
        offset += batch
        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
            print(
                f"[dataset] split={split} materialize batch {batch_idx}/{total_batches} samples_written={offset}/{size}",
                flush=True,
            )
    inputs_re.flush()
    inputs_im.flush()
    targets_re.flush()
    targets_im.flush()
    labels.flush()
    materialized = dict(manifest)
    materialized["generator"] = "memmap_fp16"
    materialized["inputs_re_path"] = inputs_re_path.name
    materialized["inputs_im_path"] = inputs_im_path.name
    materialized["targets_re_path"] = targets_re_path.name
    materialized["targets_im_path"] = targets_im_path.name
    materialized["labels_path"] = labels_path.name
    save_json(meta_path, materialized)
    print(f"[dataset] saved materialized manifest {meta_path}", flush=True)
    return meta_path


def generate_dataset(config: SystemConfig, split: str, mode: str = "full", force: bool = False) -> Path:
    if "dataset" not in config.raw:
        raise ValueError("Dataset config missing")
    out_dir = results_dir(config) / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_{mode}.npz"
    meta_path = out_dir / f"{split}_{mode}.json"
    if mode == "full":
        if meta_path.exists() and not force:
            meta = None
            try:
                from .utils import load_json

                meta = load_json(meta_path)
            except Exception:
                meta = None
            if meta and meta.get("generator") == "memmap_fp16":
                required = []
                for key in ["inputs_re_path", "inputs_im_path", "targets_re_path", "targets_im_path", "labels_path"]:
                    candidate = Path(str(meta[key]))
                    if not candidate.is_absolute():
                        candidate = (meta_path.parent / candidate).resolve()
                    required.append(candidate)
                if all(path.exists() for path in required):
                    print(f"[dataset] reusing materialized full dataset {meta_path}", flush=True)
                    return meta_path
            elif meta and meta.get("generator") == "on_the_fly" and not bool(config.raw["dataset"].get("save_npz", True)):
                print(f"[dataset] reusing full manifest {meta_path}", flush=True)
                return meta_path
        manifest = dataset_manifest(config, split, mode)
        if bool(config.raw["dataset"].get("save_npz", True)):
            return _materialize_full_dataset(config, split, manifest, out_dir)
        print(f"[dataset] writing full manifest for split={split} to {meta_path}", flush=True)
        save_json(meta_path, manifest)
        return meta_path
    if out_path.exists() and meta_path.exists() and not force:
        print(f"[dataset] reusing materialized dataset {out_path}", flush=True)
        return out_path
    manifest = dataset_manifest(config, split, mode)
    seed = int(manifest["seed"])
    pdrs = list(map(float, manifest["pdr_db"]))
    snr = float(manifest["snr_db"])
    per_pdr = int(manifest["per_pdr"])
    H, W = map(int, manifest["shape"])
    inputs = np.zeros((per_pdr * len(pdrs), H, W), dtype=np.complex64)
    targets = np.zeros_like(inputs)
    pdr_labels = np.zeros(inputs.shape[0], dtype=np.float32)
    index = 0
    rng = np.random.default_rng(seed)
    for pdr_db in pdrs:
        print(f"[dataset] split={split} mode={mode} generating pdr={pdr_db} dB samples={per_pdr}", flush=True)
        for _ in range(per_pdr):
            frame = simulate_frame(config, "bpsk", snr, pdr_db, rng)
            inputs[index] = frame.support_input
            targets[index] = frame.support_true
            pdr_labels[index] = pdr_db
            index += 1
    np.savez_compressed(out_path, inputs=inputs, targets=targets, pdr_db=pdr_labels)
    manifest["generator"] = "npz"
    save_json(meta_path, manifest)
    print(f"[dataset] saved dataset {out_path}", flush=True)
    print(f"[dataset] saved metadata {meta_path}", flush=True)
    return out_path


def detect_bits_from_data_symbols(symbols_hat: np.ndarray, modulation: str) -> np.ndarray:
    _, bits = hard_demodulate(symbols_hat, modulation)
    return bits
