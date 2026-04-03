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
from zakotfs_physics.training import train_phase2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 3 physics-consistent model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "phase3_train.yaml",
        help="Path to the Phase 3 YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = train_phase2(config)
    print(f"[physics-phase3] checkpoint={checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
