#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
PHYSICS_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
for candidate in (PHYSICS_ROOT / "src", REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config
from zakotfs.utils import results_dir, save_json
from zakotfs_physics.physics_targets import forward_physics_target, forward_physics_target_torch


def _resolve_device(config) -> torch.device:
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


def _timeit(fn, iterations: int) -> tuple[float, list[float]]:
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(samples_ms)), samples_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive versus cached physics operators.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "phase3_smoke.yaml",
        help="Path to the config used to generate one representative frame.",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Number of timed iterations after warmup.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to results/phase5_physics_operator_benchmark.json under the config root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = _resolve_device(config)
    frame = simulate_frame(config, "bpsk", 15.0, 5.0, np.random.default_rng(config.seed + 123))
    h_true = np.asarray(frame.support_true, dtype=np.complex64)
    spread_dd = np.asarray(frame.spread_dd, dtype=np.complex64)
    h_true_torch = torch.stack(
        [
            torch.from_numpy(h_true.real.astype(np.float32)),
            torch.from_numpy(h_true.imag.astype(np.float32)),
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    for _ in range(max(0, args.warmup)):
        forward_physics_target(h_true, spread_dd, frame.E_p, config)
        _ = forward_physics_target_torch(h_true_torch, config)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    naive_mean_ms, naive_samples = _timeit(lambda: forward_physics_target(h_true, spread_dd, frame.E_p, config), args.iterations)

    def _cached_call() -> None:
        _ = forward_physics_target_torch(h_true_torch, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    cached_mean_ms, cached_samples = _timeit(_cached_call, args.iterations)
    speedup = naive_mean_ms / cached_mean_ms if cached_mean_ms > 0.0 else float("inf")

    payload = {
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "naive_mean_ms": naive_mean_ms,
        "cached_mean_ms": cached_mean_ms,
        "speedup": speedup,
        "naive_samples_ms": naive_samples,
        "cached_samples_ms": cached_samples,
    }

    output_path = args.output
    if output_path is None:
        output_path = results_dir(config) / "phase5_physics_operator_benchmark.json"
    elif not output_path.is_absolute():
        output_path = (config.root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[physics-benchmark] saved={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
