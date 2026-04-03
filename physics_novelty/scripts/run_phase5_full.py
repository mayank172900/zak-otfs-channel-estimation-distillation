#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PHYSICS_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
for candidate in (PHYSICS_ROOT / "src", REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zakotfs.params import load_config
from zakotfs_physics.evaluation import run_phase5_full_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Phase 5 evaluation bundle.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "phase5_full.yaml",
        help="Path to the Phase 5 YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional novelty model checkpoint override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_phase5_full_evaluation(config, checkpoint_path=args.checkpoint)
    print(f"[physics-phase5] completed sweeps={sorted(outputs.keys())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
