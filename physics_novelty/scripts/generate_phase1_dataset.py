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
from zakotfs_physics.dataset import generate_phase1_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Phase 1 physics novelty dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "phase1_train.yaml",
        help="Path to the Phase 1 YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the dataset even if the manifest already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = generate_phase1_dataset(config, force=args.force)
    print(f"[physics-novelty] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
