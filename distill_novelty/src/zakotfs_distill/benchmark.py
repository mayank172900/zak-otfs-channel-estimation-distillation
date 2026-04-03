from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zakotfs.dataset import simulate_frame
from zakotfs.evaluation import cnn_enhance_support
from zakotfs.params import SystemConfig
from zakotfs.training import load_cnn_checkpoint
from zakotfs.utils import results_dir, save_json

from .evaluation import resolve_teacher_checkpoint, student_enhance_support
from .model import instantiate_student_model
from .training import load_student_checkpoint


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_device(config: SystemConfig) -> torch.device:
    configured = str(config.raw.get("device", "auto")).lower()
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.device=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if configured == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(configured)


def _resolve_benchmark_cfg(config: SystemConfig) -> dict[str, Any]:
    cfg = dict(config.raw.get("benchmark", {}))
    if str(cfg.get("mode", "full")).lower() == "smoke":
        cfg = _deep_update(cfg, config.raw.get("smoke", {}).get("benchmark", {}))
    return cfg


def _timeit(fn, iterations: int, device: torch.device) -> float:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(samples))


def benchmark_student_models(config: SystemConfig, checkpoint_path: str | Path | None = None) -> Path:
    cfg = _resolve_benchmark_cfg(config)
    device = _resolve_device(config)
    variant = str(config.raw.get("distill_model", {}).get("variant", "lite_l")).lower()
    teacher_path = resolve_teacher_checkpoint(config)
    student_path = checkpoint_path
    if student_path is None:
        mode = str(cfg.get("mode", "full")).lower()
        if mode == "smoke":
            student_path = config.root / "logs" / "checkpoints" / f"smoke_{variant}_distill_best.pt"
        else:
            student_path = config.root / "logs" / "checkpoints" / f"full_{variant}_distill_best.pt"
    student_path = Path(student_path).resolve()
    teacher_model = load_cnn_checkpoint(config, checkpoint_path=teacher_path)
    student_model = load_student_checkpoint(config, checkpoint_path=student_path)
    frame = simulate_frame(
        config,
        str(cfg.get("modulation", "bpsk")),
        float(cfg.get("data_snr_db", 15.0)),
        float(cfg.get("pdr_db", 5.0)),
        np.random.default_rng(config.seed + 909),
    )
    support_input = np.asarray(frame.support_input, dtype=np.complex64)
    iterations = int(cfg.get("iterations", 50))
    warmup = int(cfg.get("warmup", 5))
    for _ in range(max(0, warmup)):
        _ = cnn_enhance_support(teacher_model, support_input, next(teacher_model.parameters()).device)
        _ = student_enhance_support(student_model, support_input, next(student_model.parameters()).device)
    teacher_ms = _timeit(lambda: cnn_enhance_support(teacher_model, support_input, next(teacher_model.parameters()).device), iterations, device)
    student_ms = _timeit(lambda: student_enhance_support(student_model, support_input, next(student_model.parameters()).device), iterations, device)
    payload = {
        "mode": str(cfg.get("mode", "full")),
        "device": str(device),
        "iterations": iterations,
        "warmup": warmup,
        "teacher_checkpoint_path": str(teacher_path),
        "student_checkpoint_path": str(student_path),
        "teacher_params": int(sum(param.numel() for param in teacher_model.parameters())),
        "student_params": int(sum(param.numel() for param in student_model.parameters())),
        "teacher_mean_ms": teacher_ms,
        "student_mean_ms": student_ms,
        "student_speedup_vs_teacher": float(teacher_ms / student_ms) if student_ms > 0.0 else float("inf"),
    }
    output_path = results_dir(config) / f"distill_benchmark_{variant}_{str(cfg.get('mode', 'full')).lower()}.json"
    save_json(output_path, payload)
    print(f"[distill-benchmark] saved={output_path}", flush=True)
    return output_path
