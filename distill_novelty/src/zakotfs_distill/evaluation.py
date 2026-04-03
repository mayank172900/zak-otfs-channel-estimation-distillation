from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from zakotfs.dataset import simulate_frame
from zakotfs.evaluation import cnn_enhance_support, detect_frame as baseline_detect_frame
from zakotfs.metrics import nmse
from zakotfs.params import SystemConfig
from zakotfs.plotting import curve_plot_columns, save_curve_plot
from zakotfs.training import load_cnn_checkpoint
from zakotfs.utils import bootstrap_mean_ci, results_dir, save_json, wilson_ci

from .training import load_student_checkpoint


DISTILL_METHODS: tuple[str, ...] = ("conventional", "teacher", "student", "perfect")


DEFAULT_EVAL = {
    "mode": "full",
    "checkpoint_path": None,
    "methods": {
        "nmse": ["conventional", "teacher", "student", "perfect"],
        "ber": ["conventional", "teacher", "student", "perfect"],
    },
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


def resolve_teacher_checkpoint(config: SystemConfig) -> Path:
    return _resolve_path(config, config.raw.get("teacher_checkpoint_path", "../logs/checkpoints/full_cnn_best.pt"))


def resolve_eval_cfg(config: SystemConfig) -> dict[str, Any]:
    cfg = _deep_update(DEFAULT_EVAL, config.raw.get("distill_evaluation", {}))
    if str(cfg.get("mode", "full")).lower() == "smoke":
        cfg = _deep_update(cfg, config.raw.get("smoke", {}).get("distill_evaluation", {}))
    return cfg


def normalize_methods(methods: str | Iterable[str]) -> list[str]:
    values = [str(methods)] if isinstance(methods, str) else [str(item) for item in methods]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        method = value.lower()
        if method not in DISTILL_METHODS:
            raise ValueError(f"Unsupported distill method '{value}'")
        if method not in seen:
            seen.add(method)
            normalized.append(method)
    if not normalized:
        raise ValueError("Method list must not be empty")
    return normalized


def student_enhance_support(model: torch.nn.Module, support_input: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_re = torch.from_numpy(np.asarray(support_input.real[None, None, :, :], dtype=np.float32)).to(device)
        x_im = torch.from_numpy(np.asarray(support_input.imag[None, None, :, :], dtype=np.float32)).to(device)
        y_re = model(x_re).cpu().numpy()[0, 0]
        y_im = model(x_im).cpu().numpy()[0, 0]
    return (y_re + 1j * y_im).astype(np.complex64)


def _load_models(
    config: SystemConfig,
    methods: Iterable[str],
    checkpoint_path: str | Path | None = None,
    teacher_model: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    needed = set(methods)
    if "teacher" in needed and teacher_model is None:
        teacher_model = load_cnn_checkpoint(config, checkpoint_path=resolve_teacher_checkpoint(config))
    if "student" in needed and student_model is None:
        resolved = None if checkpoint_path is None else _resolve_path(config, checkpoint_path)
        student_model = load_student_checkpoint(config, checkpoint_path=resolved)
    return teacher_model, student_model


def estimate_channels(
    frame,
    config: SystemConfig,
    methods: str | Iterable[str],
    checkpoint_path: str | Path | None = None,
    teacher_model: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
) -> dict[str, np.ndarray]:
    normalized = normalize_methods(methods)
    teacher_model, student_model = _load_models(
        config,
        normalized,
        checkpoint_path=checkpoint_path,
        teacher_model=teacher_model,
        student_model=student_model,
    )
    estimates: dict[str, np.ndarray] = {}
    if "conventional" in normalized:
        estimates["conventional"] = np.asarray(frame.h_hat_support_thr, dtype=np.complex64)
    if "perfect" in normalized:
        estimates["perfect"] = np.asarray(frame.h_eff_support, dtype=np.complex64)
    if "teacher" in normalized:
        assert teacher_model is not None
        device = next(teacher_model.parameters()).device
        estimates["teacher"] = cnn_enhance_support(teacher_model, frame.support_input, device)
    if "student" in normalized:
        assert student_model is not None
        device = next(student_model.parameters()).device
        estimates["student"] = student_enhance_support(student_model, frame.support_input, device)
    return {method: estimates[method] for method in normalized}


def _methods_for_metric(eval_cfg: dict[str, Any], metric_name: str) -> list[str]:
    methods_cfg = eval_cfg.get("methods", DEFAULT_EVAL["methods"])
    if isinstance(methods_cfg, dict):
        values = methods_cfg.get(metric_name, methods_cfg.get("all", []))
    else:
        values = methods_cfg
    return normalize_methods(values)


def _nmse_rows_for_point(
    config: SystemConfig,
    modulation: str,
    data_snr_db: float,
    pdr_db: float,
    realizations: int,
    methods: list[str],
    eval_cfg: dict[str, Any],
    seed: int,
    x_name: str,
    x_value: float,
    teacher_model: torch.nn.Module | None,
    student_model: torch.nn.Module | None,
) -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = {method: [] for method in methods}
    for _ in range(realizations):
        frame = simulate_frame(config, modulation, data_snr_db, pdr_db, rng)
        estimates = estimate_channels(
            frame,
            config,
            methods,
            checkpoint_path=eval_cfg.get("checkpoint_path"),
            teacher_model=teacher_model,
            student_model=student_model,
        )
        for method in methods:
            metric_samples[method].append(nmse(estimates[method], frame.h_eff_support))
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
    eval_cfg: dict[str, Any],
    point_cfg: dict[str, Any],
    seed: int,
    x_name: str,
    x_value: float,
    teacher_model: torch.nn.Module | None,
    student_model: torch.nn.Module | None,
) -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(seed)
    target_errors = int(point_cfg.get("target_bit_errors", 200))
    max_bits = int(point_cfg.get("max_bits", 2_000_000))
    min_frames = int(point_cfg.get("min_frames", 20))
    solver = str(point_cfg.get("solver", "dense"))
    err_counts = {method: 0 for method in methods}
    total_bits = 0
    frames = 0
    while total_bits < max_bits and (frames < min_frames or any(err_counts[m] < target_errors for m in methods)):
        frame = simulate_frame(config, modulation, data_snr_db, pdr_db, rng)
        estimates = estimate_channels(
            frame,
            config,
            methods,
            checkpoint_path=eval_cfg.get("checkpoint_path"),
            teacher_model=teacher_model,
            student_model=student_model,
        )
        for method in methods:
            _, bits_hat = baseline_detect_frame(frame, estimates[method], config, solver=solver)
            err_counts[method] += int(np.count_nonzero(bits_hat != frame.bits))
        total_bits += int(frame.bits.size)
        frames += 1
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
                "solver": solver,
                "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
            }
        )
    return rows


def _save_eval_outputs(df: pd.DataFrame, metric_name: str, prefix: str, config: SystemConfig, save_png: bool) -> None:
    out_dir = results_dir(config)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    png_path = out_dir / f"{prefix}.png"
    df.to_csv(csv_path, index=False)
    save_json(json_path, {"metric": metric_name, "records": df.to_dict(orient="records")})
    if save_png:
        x_col = "pdr_db" if "pdr_db" in df.columns else "data_snr_db"
        hue_col, style_col = curve_plot_columns(df)
        save_curve_plot(df, x_col, metric_name, hue_col, prefix.replace("_", " ").upper(), png_path, logy=True, style_col=style_col)
    print(f"[distill-eval] saved csv={csv_path}", flush=True)
    print(f"[distill-eval] saved json={json_path}", flush=True)
    if save_png:
        print(f"[distill-eval] saved png={png_path}", flush=True)


def run_distill_evaluation(
    config: SystemConfig,
    checkpoint_path: str | Path | None = None,
    teacher_model: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
) -> dict[str, pd.DataFrame]:
    eval_cfg = resolve_eval_cfg(config)
    if checkpoint_path is not None:
        eval_cfg["checkpoint_path"] = str(checkpoint_path)
    nmse_methods = _methods_for_metric(eval_cfg, "nmse")
    ber_methods = _methods_for_metric(eval_cfg, "ber")
    teacher_model, student_model = _load_models(
        config,
        set(nmse_methods + ber_methods),
        checkpoint_path=eval_cfg.get("checkpoint_path"),
        teacher_model=teacher_model,
        student_model=student_model,
    )
    mode = str(eval_cfg.get("mode", "full"))
    variant = str(config.raw.get("distill_model", {}).get("variant", "lite_l")).lower()
    save_png = bool(eval_cfg.get("save_png", True))
    outputs: dict[str, pd.DataFrame] = {}

    nmse_pdr_cfg = dict(eval_cfg["nmse_vs_pdr"])
    if bool(nmse_pdr_cfg.get("enabled", True)):
        rows: list[dict[str, float | str | int]] = []
        for idx, pdr_db in enumerate(list(nmse_pdr_cfg["pdr_db"])):
            rows.extend(
                _nmse_rows_for_point(
                    config,
                    str(nmse_pdr_cfg["modulation"]),
                    float(nmse_pdr_cfg["data_snr_db"]),
                    float(pdr_db),
                    int(nmse_pdr_cfg["realizations"]),
                    nmse_methods,
                    eval_cfg,
                    config.seed + 310 + idx,
                    "pdr_db",
                    float(pdr_db),
                    teacher_model,
                    student_model,
                )
            )
        df = pd.DataFrame(rows)
        _save_eval_outputs(df, "nmse", f"distill_nmse_vs_pdr_{variant}_{mode}", config, save_png)
        outputs["nmse_vs_pdr"] = df

    nmse_snr_cfg = dict(eval_cfg["nmse_vs_snr"])
    if bool(nmse_snr_cfg.get("enabled", True)):
        rows = []
        for idx, snr_db in enumerate(list(nmse_snr_cfg["data_snr_db"])):
            rows.extend(
                _nmse_rows_for_point(
                    config,
                    str(nmse_snr_cfg["modulation"]),
                    float(snr_db),
                    float(nmse_snr_cfg["pdr_db"]),
                    int(nmse_snr_cfg["realizations"]),
                    nmse_methods,
                    eval_cfg,
                    config.seed + 320 + idx,
                    "data_snr_db",
                    float(snr_db),
                    teacher_model,
                    student_model,
                )
            )
        df = pd.DataFrame(rows)
        _save_eval_outputs(df, "nmse", f"distill_nmse_vs_snr_{variant}_{mode}", config, save_png)
        outputs["nmse_vs_snr"] = df

    ber_pdr_cfg = dict(eval_cfg["ber_vs_pdr"])
    if bool(ber_pdr_cfg.get("enabled", True)):
        rows = []
        for idx, pdr_db in enumerate(list(ber_pdr_cfg["pdr_db"])):
            rows.extend(
                _ber_rows_for_point(
                    config,
                    str(ber_pdr_cfg["modulation"]),
                    float(ber_pdr_cfg["data_snr_db"]),
                    float(pdr_db),
                    ber_methods,
                    eval_cfg,
                    ber_pdr_cfg,
                    config.seed + 330 + idx,
                    "pdr_db",
                    float(pdr_db),
                    teacher_model,
                    student_model,
                )
            )
        df = pd.DataFrame(rows)
        _save_eval_outputs(df, "ber", f"distill_ber_vs_pdr_{variant}_{mode}", config, save_png)
        outputs["ber_vs_pdr"] = df

    ber_snr_cfg = dict(eval_cfg["ber_vs_snr"])
    if bool(ber_snr_cfg.get("enabled", True)):
        rows = []
        modulations = [str(item) for item in ber_snr_cfg["modulation"]]
        snrs = [float(item) for item in ber_snr_cfg["data_snr_db"]]
        for midx, modulation in enumerate(modulations):
            for sidx, snr_db in enumerate(snrs):
                rows.extend(
                    _ber_rows_for_point(
                        config,
                        modulation,
                        float(snr_db),
                        float(ber_snr_cfg["pdr_db"]),
                        ber_methods,
                        eval_cfg,
                        ber_snr_cfg,
                        config.seed + 340 + 10 * midx + sidx,
                        "data_snr_db",
                        float(snr_db),
                        teacher_model,
                        student_model,
                    )
                )
        df = pd.DataFrame(rows)
        _save_eval_outputs(df, "ber", f"distill_ber_vs_snr_{variant}_{mode}", config, save_png)
        outputs["ber_vs_snr"] = df

    print(f"[distill-eval] completed sweeps={sorted(outputs.keys())}", flush=True)
    return outputs
