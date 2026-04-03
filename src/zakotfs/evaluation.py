from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .dataset import simulate_frame
from .estimators import pilot_cancellation_with_config
from .metrics import ber_from_bits, nmse
from .mmse import mmse_dense, mmse_dense_torch, mmse_iterative
from .modulation import hard_demodulate
from .params import SystemConfig
from .plotting import curve_plot_columns, save_curve_plot
from .utils import bootstrap_mean_ci, results_dir, save_json, wilson_ci


def cnn_enhance_support(model: torch.nn.Module, support_input: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_re = torch.from_numpy(np.asarray(support_input.real[None, None, :, :], dtype=np.float32)).to(device)
        x_im = torch.from_numpy(np.asarray(support_input.imag[None, None, :, :], dtype=np.float32)).to(device)
        y_re = model(x_re).cpu().numpy()[0, 0]
        y_im = model(x_im).cpu().numpy()[0, 0]
    return (y_re + 1j * y_im).astype(np.complex64)


def estimate_channels(frame, config: SystemConfig, model: torch.nn.Module | None = None) -> dict[str, np.ndarray]:
    estimates = {
        "conventional_raw": frame.h_hat_support_raw,
        "conventional": frame.h_hat_support_thr,
        "perfect": frame.h_eff_support,
    }
    if model is not None:
        device = next(model.parameters()).device
        support_cnn = cnn_enhance_support(model, frame.support_input, device)
        estimates["cnn"] = support_cnn
    return estimates


def _default_solver_name(config: SystemConfig, mode: str, point_cfg: dict) -> str:
    solver = str(point_cfg.get("solver", "")).lower()
    if solver:
        return solver
    use_dense = bool(point_cfg.get("use_dense", mode == "smoke"))
    if use_dense:
        configured_device = str(config.raw.get("device", "auto")).lower()
        if configured_device in {"auto", "cuda"} and torch.cuda.is_available():
            return "dense_torch"
        return "dense"
    return "iterative"


def detect_frame(frame, h_est: np.ndarray, config: SystemConfig, solver: str = "iterative") -> tuple[np.ndarray, np.ndarray]:
    y_data = pilot_cancellation_with_config(frame.y_dd, h_est, np.sqrt(frame.E_p) * frame.spread_dd, config)
    solver_name = str(solver).lower()
    if solver_name == "dense_torch":
        configured_device = str(config.raw.get("device", "auto")).lower()
        if configured_device == "cuda":
            device = torch.device("cuda")
        elif configured_device == "auto" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x_hat_dd = mmse_dense_torch(y_data, h_est, frame.noise_variance, config, E_d=frame.E_d, device=device)
    elif solver_name == "dense":
        x_hat_dd = mmse_dense(y_data, h_est, frame.noise_variance, config, E_d=frame.E_d)
    elif solver_name == "iterative":
        x_hat_dd = mmse_iterative(y_data, h_est, frame.noise_variance, config, E_d=frame.E_d)
    else:
        raise ValueError(f"Unknown MMSE solver '{solver}'")
    symbols_hat = x_hat_dd * np.sqrt(config.Q) / np.sqrt(frame.E_d)
    decided, bits = hard_demodulate(symbols_hat, frame.modulation)
    return decided, bits


def _resolve_point_cfg(config: SystemConfig, section: str, mode: str) -> dict:
    base = dict(config.raw["evaluation"][section])
    if mode == "smoke":
        base.update(config.raw.get("smoke", {}).get("evaluation", {}).get(section, {}))
    return base


def _curve_eval_nmse(
    config: SystemConfig,
    modulation: str,
    realizations: int,
    data_snr_db: list[float] | float,
    pdr_db: list[float] | float,
    model: torch.nn.Module | None,
    x_name: str,
    methods: list[str],
    seed_offset: int,
) -> pd.DataFrame:
    x_values = data_snr_db if isinstance(data_snr_db, list) else pdr_db if isinstance(pdr_db, list) else [0.0]
    rng = np.random.default_rng(config.seed + seed_offset)
    rows: list[dict[str, float | str]] = []
    for x_val in x_values:
        print(f"[eval:nmse] {x_name}={float(x_val):.3f} modulation={modulation} realizations={realizations}", flush=True)
        metric_samples: dict[str, list[float]] = {method: [] for method in methods}
        for idx in range(realizations):
            snr = float(x_val if x_name == "data_snr_db" else data_snr_db)
            pdr = float(x_val if x_name == "pdr_db" else pdr_db)
            frame = simulate_frame(config, modulation, snr, pdr, rng)
            estimates = estimate_channels(frame, config, model=model)
            for method in methods:
                h_est = estimates["cnn"] if method == "cnn" else estimates["perfect"] if method == "perfect" else estimates["conventional"]
                metric_samples[method].append(nmse(h_est, frame.h_eff_support))
            if (idx + 1) == 1 or (idx + 1) % 50 == 0 or (idx + 1) == realizations:
                print(f"[eval:nmse] {x_name}={float(x_val):.3f} progress={idx + 1}/{realizations}", flush=True)
        for method in methods:
            values = np.array(metric_samples[method], dtype=float)
            lo, hi = bootstrap_mean_ci(values, seed=config.seed + seed_offset + int(abs(hash((method, x_val))) % 1_000_000))
            print(
                f"[eval:nmse] {x_name}={float(x_val):.3f} method={method} mean={float(np.mean(values)):.6e} "
                f"ci=[{lo:.6e}, {hi:.6e}]",
                flush=True,
            )
            rows.append(
                {
                    x_name: float(x_val),
                    "method": method,
                    "nmse": float(np.mean(values)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "realizations": int(realizations),
                    "modulation": modulation,
                    "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
                }
            )
    return pd.DataFrame(rows)


def _curve_eval_ber(
    config: SystemConfig,
    modulation: str,
    data_snr_db: list[float] | float,
    pdr_db: list[float] | float,
    model: torch.nn.Module | None,
    x_name: str,
    methods: list[str],
    seed_offset: int,
    point_cfg: dict,
    mode: str,
) -> pd.DataFrame:
    x_values = data_snr_db if isinstance(data_snr_db, list) else pdr_db if isinstance(pdr_db, list) else [0.0]
    rng = np.random.default_rng(config.seed + seed_offset)
    rows: list[dict[str, float | str | int]] = []
    target_errors = int(point_cfg.get("target_bit_errors", 200))
    max_bits = int(point_cfg.get("max_bits", 2_000_000))
    min_frames = int(point_cfg.get("min_frames", 1 if mode == "smoke" else 20))
    solver_name = _default_solver_name(config, mode, point_cfg)
    for x_val in x_values:
        snr = float(x_val if x_name == "data_snr_db" else data_snr_db)
        pdr = float(x_val if x_name == "pdr_db" else pdr_db)
        print(
            f"[eval:ber] {x_name}={float(x_val):.3f} modulation={modulation} target_errors={target_errors} "
            f"max_bits={max_bits} min_frames={min_frames} solver={solver_name}",
            flush=True,
        )
        err_counts = {method: 0 for method in methods}
        total_bits = 0
        frames = 0
        while total_bits < max_bits and (frames < min_frames or any(err_counts[m] < target_errors for m in methods)):
            frame = simulate_frame(config, modulation, snr, pdr, rng)
            estimates = estimate_channels(frame, config, model=model)
            for method in methods:
                h_est = estimates["cnn"] if method == "cnn" else estimates["perfect"] if method == "perfect" else estimates["conventional"]
                _, bits_hat = detect_frame(frame, h_est, config, solver=solver_name)
                err_counts[method] += int(np.count_nonzero(bits_hat != frame.bits))
            total_bits += int(frame.bits.size)
            frames += 1
            if frames == 1 or frames % 10 == 0:
                summary = " ".join(f"{m}={err_counts[m] / max(total_bits, 1):.3e}" for m in methods)
                print(f"[eval:ber] {x_name}={float(x_val):.3f} frames={frames} bits={total_bits} {summary}", flush=True)
        for method in methods:
            ber = err_counts[method] / max(total_bits, 1)
            ci_low, ci_high = wilson_ci(err_counts[method], total_bits)
            print(
                f"[eval:ber] {x_name}={float(x_val):.3f} method={method} ber={ber:.6e} "
                f"errors={err_counts[method]} bits={total_bits} ci=[{ci_low:.6e}, {ci_high:.6e}]",
                flush=True,
            )
            rows.append(
                {
                    x_name: float(x_val),
                    "method": method,
                    "ber": float(ber),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "frames": int(frames),
                    "bits": int(total_bits),
                    "errors": int(err_counts[method]),
                    "target_bit_errors": int(target_errors),
                    "max_bits": int(max_bits),
                    "modulation": modulation,
                    "effective_channel_method": str(config.raw.get("numerics", {}).get("effective_channel_method", "fast")),
                    "solver": solver_name,
                }
            )
    return pd.DataFrame(rows)


def evaluate_nmse_vs_pdr(config: SystemConfig, model: torch.nn.Module | None, mode: str) -> pd.DataFrame:
    cfg = _resolve_point_cfg(config, "nmse_vs_pdr", mode)
    realizations = int(cfg.get("realizations", cfg.get("frames", 200)))
    return _curve_eval_nmse(config, cfg["modulation"], realizations, cfg["data_snr_db"], list(cfg["pdr_db"]), model, "pdr_db", ["conventional", "cnn"], 101)


def evaluate_nmse_vs_snr(config: SystemConfig, model: torch.nn.Module | None, mode: str) -> pd.DataFrame:
    cfg = _resolve_point_cfg(config, "nmse_vs_snr", mode)
    realizations = int(cfg.get("realizations", cfg.get("frames", 200)))
    return _curve_eval_nmse(config, cfg["modulation"], realizations, list(cfg["data_snr_db"]), cfg["pdr_db"], model, "data_snr_db", ["conventional", "cnn"], 102)


def evaluate_ber_vs_pdr(config: SystemConfig, model: torch.nn.Module | None, mode: str) -> pd.DataFrame:
    cfg = _resolve_point_cfg(config, "ber_vs_pdr", mode)
    return _curve_eval_ber(config, cfg["modulation"], cfg["data_snr_db"], list(cfg["pdr_db"]), model, "pdr_db", ["conventional", "cnn", "perfect"], 103, cfg, mode)


def evaluate_ber_vs_snr(config: SystemConfig, model: torch.nn.Module | None, mode: str) -> pd.DataFrame:
    cfg = _resolve_point_cfg(config, "ber_vs_snr", mode)
    frames_out = []
    for modulation in cfg["modulation"]:
        frames_out.append(
            _curve_eval_ber(config, modulation, list(cfg["data_snr_db"]), cfg["pdr_db"], model, "data_snr_db", ["conventional", "cnn", "perfect"], 104 + len(frames_out), cfg, mode)
        )
    return pd.concat(frames_out, ignore_index=True)


def save_eval_outputs(df: pd.DataFrame, metric_name: str, prefix: str, config: SystemConfig) -> None:
    out_dir = results_dir(config)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    png_path = out_dir / f"{prefix}.png"
    df.to_csv(csv_path, index=False)
    save_json(json_path, {"records": df.to_dict(orient="records")})
    x_col = "pdr_db" if "pdr_db" in df.columns else "data_snr_db"
    title = prefix.replace("_", " ").upper()
    hue_col, style_col = curve_plot_columns(df)
    save_curve_plot(df, x_col, metric_name, hue_col, title, png_path, logy=True, style_col=style_col)
    print(f"[eval] saved csv={csv_path}", flush=True)
    print(f"[eval] saved json={json_path}", flush=True)
    print(f"[eval] saved png={png_path}", flush=True)


def compare_to_anchors(config: SystemConfig, nmse_pdr: pd.DataFrame, ber_pdr: pd.DataFrame, ber_snr: pd.DataFrame) -> dict:
    anchors = config.raw["anchors"]
    summary = {}
    conv_10 = float(nmse_pdr[(nmse_pdr["pdr_db"] == 10.0) & (nmse_pdr["method"] == "conventional")]["nmse"].iloc[0])
    cnn_10 = float(nmse_pdr[(nmse_pdr["pdr_db"] == 10.0) & (nmse_pdr["method"] == "cnn")]["nmse"].iloc[0])
    summary["nmse_vs_pdr"] = {
        "paper": anchors["nmse_vs_pdr"]["pdr_db_10"],
        "reproduced": {"conventional": conv_10, "cnn": cnn_10},
    }
    mask = (ber_pdr["pdr_db"] == 5.0)
    summary["ber_vs_pdr"] = {
        "paper": anchors["ber_vs_pdr"]["pdr_db_5_bpsk_snr_db_18"],
        "reproduced": {
            method: float(ber_pdr[mask & (ber_pdr["method"] == method)]["ber"].iloc[0]) for method in ["conventional", "cnn", "perfect"]
        },
    }
    mask_bpsk = (ber_snr["data_snr_db"] == 18.0) & (ber_snr["modulation"] == "bpsk")
    mask_qam8 = (ber_snr["data_snr_db"] == 18.0) & (ber_snr["modulation"] == "8qam_cross")
    summary["ber_vs_snr_bpsk"] = {
        "paper": anchors["ber_vs_snr"]["bpsk_snr_db_18_pdr_db_5"],
        "reproduced": {
            method: float(ber_snr[mask_bpsk & (ber_snr["method"] == method)]["ber"].iloc[0]) for method in ["conventional", "cnn", "perfect"]
        },
    }
    summary["ber_vs_snr_8qam"] = {
        "paper": anchors["ber_vs_snr"]["qam8_snr_db_18_pdr_db_5"],
        "reproduced": {
            method: float(ber_snr[mask_qam8 & (ber_snr["method"] == method)]["ber"].iloc[0]) for method in ["conventional", "cnn", "perfect"]
        },
    }
    return summary
