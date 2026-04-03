from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from zakotfs.dataset import simulate_frame
from zakotfs.evaluation import detect_frame as baseline_detect_frame
from zakotfs.metrics import nmse
from zakotfs.params import SystemConfig
from zakotfs.plotting import curve_plot_columns, save_curve_plot
from zakotfs.utils import bootstrap_mean_ci, results_dir, save_json, wilson_ci

from .baseline_bridge import baseline_estimate_support, load_frozen_baseline
from .dataset import Phase1MemmapDataset, load_phase1_manifest
from .model import predict_phase2
from .training import load_phase2_checkpoint

PHASE4_CHANNEL_MODES: tuple[str, ...] = ("baseline", "refined", "blended", "thresholded")
PHASE5_SUPPORTED_METHODS: tuple[str, ...] = ("conventional", "baseline", "refined", "blended", "thresholded", "perfect")

DEFAULT_PHASE4_INFERENCE = {
    "channel_mode": "blended",
    "confidence_floor": 0.0,
    "confidence_ceiling": 1.0,
    "confidence_scale": 1.0,
    "confidence_threshold": 0.5,
    "solver": "iterative",
}

DEFAULT_PHASE5_EVALUATION = {
    "mode": "full",
    "checkpoint_path": None,
    "methods": {
        "nmse": ["conventional", "baseline", "refined", "blended", "perfect"],
        "ber": ["conventional", "baseline", "refined", "blended", "perfect"],
    },
    "method_phase4_overrides": {},
    "save_png": True,
    "nmse_vs_pdr": {"enabled": True},
    "nmse_vs_snr": {"enabled": True},
    "ber_vs_pdr": {"enabled": True},
    "ber_vs_snr": {"enabled": True},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(config: SystemConfig, value: str | Path) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (config.root / target).resolve()


def resolve_phase4_inference_cfg(config: SystemConfig) -> dict:
    cfg = dict(DEFAULT_PHASE4_INFERENCE)
    cfg.update(config.raw.get("phase4_inference", {}))
    mode = str(config.raw.get("phase2_training", {}).get("mode", "full"))
    if mode == "smoke":
        cfg.update(config.raw.get("smoke", {}).get("phase4_inference", {}))
    return cfg


def resolve_phase5_evaluation_cfg(config: SystemConfig) -> dict:
    cfg = _deep_update(DEFAULT_PHASE5_EVALUATION, config.raw.get("phase5_evaluation", {}))
    if str(cfg.get("mode", "full")).lower() == "smoke":
        cfg = _deep_update(cfg, config.raw.get("smoke", {}).get("phase5_evaluation", {}))
    return cfg


def _complex_support_to_channels(value: np.ndarray) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.complex64)
    return torch.from_numpy(np.stack([arr.real.astype(np.float32), arr.imag.astype(np.float32)], axis=0))


def support_channels_to_complex_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError(f"Expected support tensor with shape (2, H, W), got {tuple(tensor.shape)}")
        return (tensor[0].numpy() + 1j * tensor[1].numpy()).astype(np.complex64)
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return (arr[0].astype(np.float32) + 1j * arr[1].astype(np.float32)).astype(np.complex64)
    raise ValueError(f"Unsupported support value shape: {arr.shape}")


def build_phase2_inputs_from_supports(h_obs: np.ndarray, h_thr: np.ndarray, h_base: np.ndarray) -> torch.Tensor:
    return torch.cat(
        [
            _complex_support_to_channels(h_obs),
            _complex_support_to_channels(h_thr),
            _complex_support_to_channels(h_base),
        ],
        dim=0,
    )


def build_phase4_inputs_from_frame(frame, baseline_model: torch.nn.Module) -> dict[str, torch.Tensor | np.ndarray]:
    h_obs = np.asarray(frame.support_input, dtype=np.complex64)
    h_thr = np.asarray(frame.h_hat_support_thr, dtype=np.complex64)
    h_base = baseline_estimate_support(baseline_model, h_obs)
    return {
        "inputs": build_phase2_inputs_from_supports(h_obs, h_thr, h_base),
        "h_obs": h_obs,
        "h_thr": h_thr,
        "h_base": h_base,
    }


def build_phase4_inputs_from_sample(sample: dict[str, np.ndarray | float | int]) -> dict[str, torch.Tensor | np.ndarray]:
    h_obs = np.asarray(sample["H_obs"], dtype=np.complex64)
    h_thr = np.asarray(sample["H_thr"], dtype=np.complex64)
    h_base = np.asarray(sample["H_base"], dtype=np.complex64)
    return {
        "inputs": build_phase2_inputs_from_supports(h_obs, h_thr, h_base),
        "h_obs": h_obs,
        "h_thr": h_thr,
        "h_base": h_base,
    }


def _detach_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def fuse_support_estimate(
    h_hat: torch.Tensor,
    h_base: torch.Tensor,
    confidence: torch.Tensor,
    mode: str = "blended",
    confidence_floor: float = 0.0,
    confidence_ceiling: float = 1.0,
    confidence_scale: float = 1.0,
    confidence_threshold: float | None = 0.5,
) -> dict[str, torch.Tensor | str]:
    resolved_mode = str(mode).lower()
    if resolved_mode not in PHASE4_CHANNEL_MODES:
        raise ValueError(f"Unknown Phase 4 channel mode '{mode}'")
    floor = max(0.0, min(float(confidence_floor), 1.0))
    ceiling = max(floor, min(float(confidence_ceiling), 1.0))
    confidence_clamped = confidence.clamp(min=floor, max=ceiling)
    fusion_weight = (confidence_clamped * max(0.0, float(confidence_scale))).clamp(min=0.0, max=1.0)
    h_blended = fusion_weight * h_hat + (1.0 - fusion_weight) * h_base
    if resolved_mode == "baseline":
        h_use = h_base
    elif resolved_mode == "refined":
        h_use = h_hat
    elif resolved_mode == "blended":
        h_use = h_blended
    else:
        threshold = float(0.5 if confidence_threshold is None else confidence_threshold)
        hard_mask = (confidence_clamped >= threshold).to(dtype=h_hat.dtype)
        h_use = hard_mask * h_hat + (1.0 - hard_mask) * h_base
    return {
        "channel_mode": resolved_mode,
        "confidence": confidence_clamped,
        "fusion_weight": fusion_weight,
        "h_blended": h_blended,
        "h_use": h_use,
    }


def select_phase4_channel_outputs(
    predictions: dict[str, torch.Tensor],
    config: SystemConfig,
    channel_mode: str | None = None,
    phase4_cfg: dict | None = None,
) -> dict[str, torch.Tensor | str]:
    resolved_cfg = dict(resolve_phase4_inference_cfg(config) if phase4_cfg is None else phase4_cfg)
    if channel_mode is not None:
        resolved_cfg["channel_mode"] = str(channel_mode)
    fused = fuse_support_estimate(
        predictions["h_hat"],
        predictions["h_base"],
        predictions["confidence"],
        mode=str(resolved_cfg["channel_mode"]),
        confidence_floor=float(resolved_cfg.get("confidence_floor", 0.0)),
        confidence_ceiling=float(resolved_cfg.get("confidence_ceiling", 1.0)),
        confidence_scale=float(resolved_cfg.get("confidence_scale", 1.0)),
        confidence_threshold=float(resolved_cfg.get("confidence_threshold", 0.5)),
    )
    return {
        "delta": predictions["delta"],
        "uncertainty": predictions["uncertainty"],
        "confidence": fused["confidence"],
        "fusion_weight": fused["fusion_weight"],
        "h_base": predictions["h_base"],
        "h_hat": predictions["h_hat"],
        "h_blended": fused["h_blended"],
        "h_use": fused["h_use"],
        "channel_mode": fused["channel_mode"],
    }


def infer_phase2_support(model: torch.nn.Module, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    with torch.no_grad():
        return predict_phase2(model, inputs.to(device=device, dtype=torch.float32))


def infer_phase2_checkpoint(config: SystemConfig, inputs: torch.Tensor, checkpoint_path: Path | None = None) -> dict[str, torch.Tensor]:
    model = load_phase2_checkpoint(config, checkpoint_path=checkpoint_path)
    device = next(model.parameters()).device
    return infer_phase2_support(model, inputs.to(device=device, dtype=torch.float32))


def infer_phase4_support(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    config: SystemConfig,
    channel_mode: str | None = None,
    phase4_cfg: dict | None = None,
) -> dict[str, torch.Tensor | str]:
    predictions = infer_phase2_support(model, inputs)
    selected = select_phase4_channel_outputs(predictions, config, channel_mode=channel_mode, phase4_cfg=phase4_cfg)
    return {key: _detach_cpu(value) for key, value in selected.items()}


def infer_phase4_checkpoint(
    config: SystemConfig,
    inputs: torch.Tensor,
    checkpoint_path: Path | None = None,
    channel_mode: str | None = None,
    phase4_cfg: dict | None = None,
) -> dict[str, torch.Tensor | str]:
    model = load_phase2_checkpoint(config, checkpoint_path=checkpoint_path)
    return infer_phase4_support(model, inputs, config, channel_mode=channel_mode, phase4_cfg=phase4_cfg)


def frame_from_phase1_sample(
    config: SystemConfig,
    manifest: dict,
    sample: dict[str, np.ndarray | float | int],
):
    return simulate_frame(
        config,
        str(manifest["modulation"]),
        float(manifest["snr_db"]),
        float(sample["pdr_db"]),
        np.random.default_rng(int(sample["sample_seed"])),
    )


def detect_phase4_frame(
    frame,
    config: SystemConfig,
    model: torch.nn.Module,
    baseline_model: torch.nn.Module | None = None,
    channel_mode: str | None = None,
    phase4_cfg: dict | None = None,
) -> dict[str, torch.Tensor | np.ndarray | float | str]:
    resolved_cfg = dict(resolve_phase4_inference_cfg(config) if phase4_cfg is None else phase4_cfg)
    if channel_mode is not None:
        resolved_cfg["channel_mode"] = str(channel_mode)
    if baseline_model is None:
        baseline_model, _, _ = load_frozen_baseline(config)
    frame_inputs = build_phase4_inputs_from_frame(frame, baseline_model)
    outputs = infer_phase4_support(model, frame_inputs["inputs"], config, channel_mode=str(resolved_cfg["channel_mode"]), phase4_cfg=resolved_cfg)
    decided, bits_hat = baseline_detect_frame(
        frame,
        support_channels_to_complex_numpy(outputs["h_use"]),
        config,
        solver=str(resolved_cfg.get("solver", "iterative")),
    )
    ber = float(np.count_nonzero(bits_hat != frame.bits) / max(frame.bits.size, 1))
    return {
        "inputs": frame_inputs["inputs"],
        "h_obs": _complex_support_to_channels(frame_inputs["h_obs"]),
        "h_thr": _complex_support_to_channels(frame_inputs["h_thr"]),
        "h_base_input": _complex_support_to_channels(frame_inputs["h_base"]),
        **outputs,
        "decided": decided,
        "bits_hat": bits_hat,
        "ber": ber,
        "solver": str(resolved_cfg.get("solver", "iterative")),
    }


def detect_phase4_sample(
    manifest_path: str | Path,
    index: int,
    config: SystemConfig,
    model: torch.nn.Module,
    channel_mode: str | None = None,
    phase4_cfg: dict | None = None,
) -> dict[str, torch.Tensor | np.ndarray | float | str]:
    manifest = load_phase1_manifest(manifest_path)
    sample = Phase1MemmapDataset(manifest_path)[index]
    frame = frame_from_phase1_sample(config, manifest, sample)
    resolved_cfg = dict(resolve_phase4_inference_cfg(config) if phase4_cfg is None else phase4_cfg)
    if channel_mode is not None:
        resolved_cfg["channel_mode"] = str(channel_mode)
    sample_inputs = build_phase4_inputs_from_sample(sample)
    outputs = infer_phase4_support(model, sample_inputs["inputs"], config, channel_mode=str(resolved_cfg["channel_mode"]), phase4_cfg=resolved_cfg)
    decided, bits_hat = baseline_detect_frame(
        frame,
        support_channels_to_complex_numpy(outputs["h_use"]),
        config,
        solver=str(resolved_cfg.get("solver", "iterative")),
    )
    ber = float(np.count_nonzero(bits_hat != frame.bits) / max(frame.bits.size, 1))
    return {
        "inputs": sample_inputs["inputs"],
        "h_obs": _complex_support_to_channels(sample_inputs["h_obs"]),
        "h_thr": _complex_support_to_channels(sample_inputs["h_thr"]),
        "h_base_input": _complex_support_to_channels(sample_inputs["h_base"]),
        **outputs,
        "decided": decided,
        "bits_hat": bits_hat,
        "ber": ber,
        "solver": str(resolved_cfg.get("solver", "iterative")),
    }


def compare_phase4_frame_modes(
    frame,
    config: SystemConfig,
    model: torch.nn.Module,
    baseline_model: torch.nn.Module | None = None,
    modes: tuple[str, ...] = ("baseline", "refined", "blended"),
    phase4_cfg: dict | None = None,
) -> dict[str, dict[str, torch.Tensor | np.ndarray | float | str]]:
    if baseline_model is None:
        baseline_model, _, _ = load_frozen_baseline(config)
    frame_inputs = build_phase4_inputs_from_frame(frame, baseline_model)
    predictions = infer_phase2_support(model, frame_inputs["inputs"])
    cfg = dict(resolve_phase4_inference_cfg(config) if phase4_cfg is None else phase4_cfg)
    results: dict[str, dict[str, torch.Tensor | np.ndarray | float | str]] = {}
    for mode in modes:
        outputs = {key: _detach_cpu(value) for key, value in select_phase4_channel_outputs(predictions, config, channel_mode=mode, phase4_cfg=cfg).items()}
        decided, bits_hat = baseline_detect_frame(
            frame,
            support_channels_to_complex_numpy(outputs["h_use"]),
            config,
            solver=str(cfg.get("solver", "iterative")),
        )
        results[str(mode)] = {
            **outputs,
            "decided": decided,
            "bits_hat": bits_hat,
            "ber": float(np.count_nonzero(bits_hat != frame.bits) / max(frame.bits.size, 1)),
        }
    return results


def compare_phase4_sample_modes(
    manifest_path: str | Path,
    index: int,
    config: SystemConfig,
    model: torch.nn.Module,
    modes: tuple[str, ...] = ("baseline", "refined", "blended"),
    phase4_cfg: dict | None = None,
) -> dict[str, dict[str, torch.Tensor | np.ndarray | float | str]]:
    manifest = load_phase1_manifest(manifest_path)
    sample = Phase1MemmapDataset(manifest_path)[index]
    frame = frame_from_phase1_sample(config, manifest, sample)
    sample_inputs = build_phase4_inputs_from_sample(sample)
    predictions = infer_phase2_support(model, sample_inputs["inputs"])
    cfg = dict(resolve_phase4_inference_cfg(config) if phase4_cfg is None else phase4_cfg)
    results: dict[str, dict[str, torch.Tensor | np.ndarray | float | str]] = {}
    for mode in modes:
        outputs = {key: _detach_cpu(value) for key, value in select_phase4_channel_outputs(predictions, config, channel_mode=mode, phase4_cfg=cfg).items()}
        decided, bits_hat = baseline_detect_frame(
            frame,
            support_channels_to_complex_numpy(outputs["h_use"]),
            config,
            solver=str(cfg.get("solver", "iterative")),
        )
        results[str(mode)] = {
            **outputs,
            "decided": decided,
            "bits_hat": bits_hat,
            "ber": float(np.count_nonzero(bits_hat != frame.bits) / max(frame.bits.size, 1)),
        }
    return results


def normalize_phase5_methods(methods: str | Iterable[str]) -> list[str]:
    values = [str(methods)] if isinstance(methods, str) else [str(item) for item in methods]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        method = value.lower()
        if method not in PHASE5_SUPPORTED_METHODS:
            raise ValueError(f"Unsupported Phase 5 method '{value}'")
        if method not in seen:
            seen.add(method)
            normalized.append(method)
    if not normalized:
        raise ValueError("Phase 5 methods must not be empty")
    return normalized


def _phase5_methods_for_metric(phase5_cfg: dict, metric_name: str) -> list[str]:
    methods_cfg = phase5_cfg.get("methods", DEFAULT_PHASE5_EVALUATION["methods"])
    if isinstance(methods_cfg, dict):
        values = methods_cfg.get(metric_name, methods_cfg.get("all", []))
    else:
        values = methods_cfg
    return normalize_phase5_methods(values)


def _phase5_method_phase4_cfg(config: SystemConfig, phase5_cfg: dict, method: str) -> dict:
    cfg = dict(resolve_phase4_inference_cfg(config))
    cfg["channel_mode"] = method
    override = dict(phase5_cfg.get("method_phase4_overrides", {}).get(method, {}))
    cfg.update(override)
    return cfg


def _needs_baseline_model(methods: Iterable[str]) -> bool:
    return any(method in {"baseline", "refined", "blended", "thresholded"} for method in methods)


def _needs_novelty_model(methods: Iterable[str]) -> bool:
    return any(method in {"refined", "blended", "thresholded"} for method in methods)


def _load_phase5_models(
    config: SystemConfig,
    methods: Iterable[str],
    phase5_cfg: dict,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    methods_list = list(methods)
    if _needs_baseline_model(methods_list) and baseline_model is None:
        baseline_model, _, _ = load_frozen_baseline(config)
    if _needs_novelty_model(methods_list) and novelty_model is None:
        target_path: Path | None = None
        explicit = checkpoint_path if checkpoint_path is not None else phase5_cfg.get("checkpoint_path")
        if explicit is not None:
            target_path = _resolve_path(config, explicit)
        novelty_model = load_phase2_checkpoint(config, checkpoint_path=target_path)
    return novelty_model, baseline_model


def estimate_phase5_channels(
    frame,
    config: SystemConfig,
    methods: str | Iterable[str],
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    phase5_cfg: dict | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    normalized = normalize_phase5_methods(methods)
    novelty_model, baseline_model = _load_phase5_models(
        config,
        normalized,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    estimates: dict[str, np.ndarray] = {}
    if "conventional" in normalized:
        estimates["conventional"] = np.asarray(frame.h_hat_support_thr, dtype=np.complex64)
    if "perfect" in normalized:
        estimates["perfect"] = np.asarray(frame.h_eff_support, dtype=np.complex64)
    if _needs_baseline_model(normalized):
        assert baseline_model is not None
        frame_inputs = build_phase4_inputs_from_frame(frame, baseline_model)
        h_base = np.asarray(frame_inputs["h_base"], dtype=np.complex64)
        if "baseline" in normalized:
            estimates["baseline"] = h_base
        novelty_methods = [method for method in normalized if method in {"refined", "blended", "thresholded"}]
        if novelty_methods:
            assert novelty_model is not None
            predictions = infer_phase2_support(novelty_model, frame_inputs["inputs"])
            for method in novelty_methods:
                phase4_cfg = _phase5_method_phase4_cfg(config, resolved_phase5, method)
                outputs = select_phase4_channel_outputs(predictions, config, channel_mode=method, phase4_cfg=phase4_cfg)
                estimates[method] = support_channels_to_complex_numpy(outputs["h_use"])
    return {method: estimates[method] for method in normalized}


def _phase5_metric_metadata(config: SystemConfig, methods: list[str], sweep_name: str, point_cfg: dict, phase5_cfg: dict) -> dict[str, Any]:
    return {
        "mode": str(phase5_cfg.get("mode", "full")),
        "methods": methods,
        "sweep": sweep_name,
        "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
        "point_config": point_cfg,
    }


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _nmse_rows_for_point(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    realizations: int,
    methods: list[str],
    novelty_model: torch.nn.Module | None,
    baseline_model: torch.nn.Module | None,
    phase5_cfg: dict,
    seed: int,
    x_name: str,
    x_value: float,
) -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = {method: [] for method in methods}
    print(
        f"[phase5:nmse] {x_name}={x_value:.3f} modulation={modulation} realizations={realizations} methods={methods}",
        flush=True,
    )
    for idx in range(realizations):
        frame = simulate_frame(config, modulation, data_snr_db, pdr_db, rng)
        estimates = estimate_phase5_channels(
            frame,
            config,
            methods,
            novelty_model=novelty_model,
            baseline_model=baseline_model,
            phase5_cfg=phase5_cfg,
        )
        for method in methods:
            metric_samples[method].append(nmse(estimates[method], frame.h_eff_support))
        if (idx + 1) == 1 or (idx + 1) == realizations or (idx + 1) % 50 == 0:
            print(f"[phase5:nmse] {x_name}={x_value:.3f} progress={idx + 1}/{realizations}", flush=True)
    rows: list[dict[str, float | str | int]] = []
    for method in methods:
        values = np.asarray(metric_samples[method], dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(values, seed=seed + int(abs(hash((method, modulation, x_name, x_value))) % 1_000_000))
        rows.append(
            {
                x_name: float(x_value),
                "method": method,
                "nmse": float(np.mean(values)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "realizations": int(realizations),
                "modulation": str(modulation),
                "data_snr_db": float(data_snr_db),
                "pdr_db": float(pdr_db),
                "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
            }
        )
    return rows


def _ber_rows_for_point(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    methods: list[str],
    novelty_model: torch.nn.Module | None,
    baseline_model: torch.nn.Module | None,
    phase5_cfg: dict,
    point_cfg: dict,
    seed: int,
    x_name: str,
    x_value: float,
) -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(seed)
    solver_name = str(point_cfg.get("solver", resolve_phase4_inference_cfg(config).get("solver", "iterative"))).lower()
    target_errors = int(point_cfg.get("target_bit_errors", 200))
    max_bits = int(point_cfg.get("max_bits", 2_000_000))
    min_frames = int(point_cfg.get("min_frames", 20))
    err_counts = {method: 0 for method in methods}
    total_bits = 0
    frames = 0
    print(
        f"[phase5:ber] {x_name}={x_value:.3f} modulation={modulation} solver={solver_name} "
        f"target_errors={target_errors} max_bits={max_bits} min_frames={min_frames} methods={methods}",
        flush=True,
    )
    while total_bits < max_bits and (frames < min_frames or any(err_counts[method] < target_errors for method in methods)):
        frame = simulate_frame(config, modulation, data_snr_db, pdr_db, rng)
        estimates = estimate_phase5_channels(
            frame,
            config,
            methods,
            novelty_model=novelty_model,
            baseline_model=baseline_model,
            phase5_cfg=phase5_cfg,
        )
        for method in methods:
            _, bits_hat = baseline_detect_frame(frame, estimates[method], config, solver=solver_name)
            err_counts[method] += int(np.count_nonzero(bits_hat != frame.bits))
        total_bits += int(frame.bits.size)
        frames += 1
        if frames == 1 or frames % 10 == 0:
            summary = " ".join(f"{method}={err_counts[method] / max(total_bits, 1):.3e}" for method in methods)
            print(f"[phase5:ber] {x_name}={x_value:.3f} frames={frames} bits={total_bits} {summary}", flush=True)
    rows: list[dict[str, float | str | int]] = []
    for method in methods:
        ber = err_counts[method] / max(total_bits, 1)
        ci_low, ci_high = wilson_ci(err_counts[method], total_bits)
        rows.append(
            {
                x_name: float(x_value),
                "method": method,
                "ber": float(ber),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "frames": int(frames),
                "bits": int(total_bits),
                "errors": int(err_counts[method]),
                "target_bit_errors": int(target_errors),
                "max_bits": int(max_bits),
                "min_frames": int(min_frames),
                "modulation": str(modulation),
                "data_snr_db": float(data_snr_db),
                "pdr_db": float(pdr_db),
                "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
                "solver": solver_name,
            }
        )
    return rows


def evaluate_phase5_nmse_point(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    realizations: int,
    methods: str | Iterable[str],
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    phase5_cfg: dict | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods_list = normalize_phase5_methods(methods)
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods_list,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    rows = _nmse_rows_for_point(
        config,
        modulation,
        float(data_snr_db),
        float(pdr_db),
        int(realizations),
        methods_list,
        novelty_model,
        baseline_model,
        resolved_phase5,
        seed=config.seed + 601,
        x_name="pdr_db",
        x_value=float(pdr_db),
    )
    return pd.DataFrame(rows)


def evaluate_phase5_ber_point(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    methods: str | Iterable[str],
    target_bit_errors: int,
    max_bits: int,
    min_frames: int,
    solver: str = "iterative",
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    phase5_cfg: dict | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods_list = normalize_phase5_methods(methods)
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods_list,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    rows = _ber_rows_for_point(
        config,
        modulation,
        float(data_snr_db),
        float(pdr_db),
        methods_list,
        novelty_model,
        baseline_model,
        resolved_phase5,
        {
            "target_bit_errors": int(target_bit_errors),
            "max_bits": int(max_bits),
            "min_frames": int(min_frames),
            "solver": str(solver),
        },
        seed=config.seed + 701,
        x_name="pdr_db",
        x_value=float(pdr_db),
    )
    return pd.DataFrame(rows)


def _curve_eval_phase5_nmse(
    config: SystemConfig,
    point_cfg: dict,
    x_name: str,
    methods: list[str],
    novelty_model: torch.nn.Module | None,
    baseline_model: torch.nn.Module | None,
    phase5_cfg: dict,
    seed_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for mod_idx, modulation in enumerate(_as_list(point_cfg["modulation"])):
        x_values = _as_list(point_cfg[x_name])
        for x_idx, x_value in enumerate(x_values):
            data_snr_db = float(x_value if x_name == "data_snr_db" else point_cfg["data_snr_db"])
            pdr_db = float(x_value if x_name == "pdr_db" else point_cfg["pdr_db"])
            rows.extend(
                _nmse_rows_for_point(
                    config,
                    str(modulation),
                    data_snr_db,
                    pdr_db,
                    int(point_cfg.get("realizations", point_cfg.get("frames", 200))),
                    methods,
                    novelty_model,
                    baseline_model,
                    phase5_cfg,
                    seed=config.seed + seed_offset + 100 * mod_idx + x_idx,
                    x_name=x_name,
                    x_value=float(x_value),
                )
            )
    return pd.DataFrame(rows)


def _curve_eval_phase5_ber(
    config: SystemConfig,
    point_cfg: dict,
    x_name: str,
    methods: list[str],
    novelty_model: torch.nn.Module | None,
    baseline_model: torch.nn.Module | None,
    phase5_cfg: dict,
    seed_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for mod_idx, modulation in enumerate(_as_list(point_cfg["modulation"])):
        x_values = _as_list(point_cfg[x_name])
        for x_idx, x_value in enumerate(x_values):
            data_snr_db = float(x_value if x_name == "data_snr_db" else point_cfg["data_snr_db"])
            pdr_db = float(x_value if x_name == "pdr_db" else point_cfg["pdr_db"])
            rows.extend(
                _ber_rows_for_point(
                    config,
                    str(modulation),
                    data_snr_db,
                    pdr_db,
                    methods,
                    novelty_model,
                    baseline_model,
                    phase5_cfg,
                    point_cfg,
                    seed=config.seed + seed_offset + 100 * mod_idx + x_idx,
                    x_name=x_name,
                    x_value=float(x_value),
                )
            )
    return pd.DataFrame(rows)


def evaluate_phase5_nmse_vs_pdr(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "nmse")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    return _curve_eval_phase5_nmse(config, dict(resolved_phase5["nmse_vs_pdr"]), "pdr_db", methods, novelty_model, baseline_model, resolved_phase5, 801)


def evaluate_phase5_nmse_vs_snr(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "nmse")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    return _curve_eval_phase5_nmse(config, dict(resolved_phase5["nmse_vs_snr"]), "data_snr_db", methods, novelty_model, baseline_model, resolved_phase5, 901)


def evaluate_phase5_ber_vs_pdr(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "ber")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    return _curve_eval_phase5_ber(config, dict(resolved_phase5["ber_vs_pdr"]), "pdr_db", methods, novelty_model, baseline_model, resolved_phase5, 1001)


def evaluate_phase5_ber_vs_snr(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> pd.DataFrame:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "ber")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    return _curve_eval_phase5_ber(config, dict(resolved_phase5["ber_vs_snr"]), "data_snr_db", methods, novelty_model, baseline_model, resolved_phase5, 1101)


def save_phase5_eval_outputs(
    df: pd.DataFrame,
    metric_name: str,
    prefix: str,
    config: SystemConfig,
    metadata: dict[str, Any] | None = None,
    save_png: bool = True,
) -> dict[str, Path]:
    out_dir = results_dir(config)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    png_path = out_dir / f"{prefix}.png"
    df.to_csv(csv_path, index=False)
    payload: dict[str, Any] = {"metric": metric_name, "records": df.to_dict(orient="records")}
    if metadata is not None:
        payload["metadata"] = metadata
    save_json(json_path, payload)
    if save_png:
        x_col = "pdr_db" if "pdr_db" in df.columns and df["pdr_db"].nunique() > 1 else "data_snr_db"
        title = prefix.replace("_", " ").upper()
        hue_col, style_col = curve_plot_columns(df)
        save_curve_plot(df, x_col, metric_name, hue_col, title, png_path, logy=True, style_col=style_col)
    print(f"[phase5] saved csv={csv_path}", flush=True)
    print(f"[phase5] saved json={json_path}", flush=True)
    if save_png:
        print(f"[phase5] saved png={png_path}", flush=True)
    return {"csv": csv_path, "json": json_path, "png": png_path}


def run_phase5_nmse_bundle(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> dict[str, pd.DataFrame]:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "nmse")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    mode = str(resolved_phase5.get("mode", "full"))
    save_png = bool(resolved_phase5.get("save_png", True))
    outputs: dict[str, pd.DataFrame] = {}
    if bool(resolved_phase5["nmse_vs_pdr"].get("enabled", True)):
        nmse_pdr = evaluate_phase5_nmse_vs_pdr(config, novelty_model=novelty_model, baseline_model=baseline_model, checkpoint_path=checkpoint_path, phase5_cfg=resolved_phase5)
        save_phase5_eval_outputs(
            nmse_pdr,
            "nmse",
            f"phase5_nmse_vs_pdr_{mode}",
            config,
            metadata=_phase5_metric_metadata(config, methods, "nmse_vs_pdr", dict(resolved_phase5["nmse_vs_pdr"]), resolved_phase5),
            save_png=save_png,
        )
        outputs["nmse_vs_pdr"] = nmse_pdr
    if bool(resolved_phase5["nmse_vs_snr"].get("enabled", True)):
        nmse_snr = evaluate_phase5_nmse_vs_snr(config, novelty_model=novelty_model, baseline_model=baseline_model, checkpoint_path=checkpoint_path, phase5_cfg=resolved_phase5)
        save_phase5_eval_outputs(
            nmse_snr,
            "nmse",
            f"phase5_nmse_vs_snr_{mode}",
            config,
            metadata=_phase5_metric_metadata(config, methods, "nmse_vs_snr", dict(resolved_phase5["nmse_vs_snr"]), resolved_phase5),
            save_png=save_png,
        )
        outputs["nmse_vs_snr"] = nmse_snr
    return outputs


def run_phase5_ber_bundle(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> dict[str, pd.DataFrame]:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    methods = _phase5_methods_for_metric(resolved_phase5, "ber")
    novelty_model, baseline_model = _load_phase5_models(
        config,
        methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    mode = str(resolved_phase5.get("mode", "full"))
    save_png = bool(resolved_phase5.get("save_png", True))
    outputs: dict[str, pd.DataFrame] = {}
    if bool(resolved_phase5["ber_vs_pdr"].get("enabled", True)):
        ber_pdr = evaluate_phase5_ber_vs_pdr(config, novelty_model=novelty_model, baseline_model=baseline_model, checkpoint_path=checkpoint_path, phase5_cfg=resolved_phase5)
        save_phase5_eval_outputs(
            ber_pdr,
            "ber",
            f"phase5_ber_vs_pdr_{mode}",
            config,
            metadata=_phase5_metric_metadata(config, methods, "ber_vs_pdr", dict(resolved_phase5["ber_vs_pdr"]), resolved_phase5),
            save_png=save_png,
        )
        outputs["ber_vs_pdr"] = ber_pdr
    if bool(resolved_phase5["ber_vs_snr"].get("enabled", True)):
        ber_snr = evaluate_phase5_ber_vs_snr(config, novelty_model=novelty_model, baseline_model=baseline_model, checkpoint_path=checkpoint_path, phase5_cfg=resolved_phase5)
        save_phase5_eval_outputs(
            ber_snr,
            "ber",
            f"phase5_ber_vs_snr_{mode}",
            config,
            metadata=_phase5_metric_metadata(config, methods, "ber_vs_snr", dict(resolved_phase5["ber_vs_snr"]), resolved_phase5),
            save_png=save_png,
        )
        outputs["ber_vs_snr"] = ber_snr
    return outputs


def run_phase5_full_evaluation(
    config: SystemConfig,
    novelty_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    phase5_cfg: dict | None = None,
) -> dict[str, pd.DataFrame]:
    resolved_phase5 = resolve_phase5_evaluation_cfg(config) if phase5_cfg is None else phase5_cfg
    all_methods = normalize_phase5_methods(
        _phase5_methods_for_metric(resolved_phase5, "nmse") + _phase5_methods_for_metric(resolved_phase5, "ber")
    )
    novelty_model, baseline_model = _load_phase5_models(
        config,
        all_methods,
        resolved_phase5,
        novelty_model=novelty_model,
        baseline_model=baseline_model,
        checkpoint_path=checkpoint_path,
    )
    outputs = {}
    outputs.update(
        run_phase5_nmse_bundle(
            config,
            novelty_model=novelty_model,
            baseline_model=baseline_model,
            checkpoint_path=checkpoint_path,
            phase5_cfg=resolved_phase5,
        )
    )
    outputs.update(
        run_phase5_ber_bundle(
            config,
            novelty_model=novelty_model,
            baseline_model=baseline_model,
            checkpoint_path=checkpoint_path,
            phase5_cfg=resolved_phase5,
        )
    )
    return outputs


__all__ = [
    "DEFAULT_PHASE4_INFERENCE",
    "DEFAULT_PHASE5_EVALUATION",
    "PHASE4_CHANNEL_MODES",
    "PHASE5_SUPPORTED_METHODS",
    "build_phase2_inputs_from_supports",
    "build_phase4_inputs_from_frame",
    "build_phase4_inputs_from_sample",
    "compare_phase4_frame_modes",
    "compare_phase4_sample_modes",
    "detect_phase4_frame",
    "detect_phase4_sample",
    "estimate_phase5_channels",
    "evaluate_phase5_ber_point",
    "evaluate_phase5_ber_vs_pdr",
    "evaluate_phase5_ber_vs_snr",
    "evaluate_phase5_nmse_point",
    "evaluate_phase5_nmse_vs_pdr",
    "evaluate_phase5_nmse_vs_snr",
    "frame_from_phase1_sample",
    "fuse_support_estimate",
    "infer_phase2_checkpoint",
    "infer_phase2_support",
    "infer_phase4_checkpoint",
    "infer_phase4_support",
    "normalize_phase5_methods",
    "resolve_phase4_inference_cfg",
    "resolve_phase5_evaluation_cfg",
    "run_phase5_ber_bundle",
    "run_phase5_full_evaluation",
    "run_phase5_nmse_bundle",
    "save_phase5_eval_outputs",
    "select_phase4_channel_outputs",
    "support_channels_to_complex_numpy",
]
