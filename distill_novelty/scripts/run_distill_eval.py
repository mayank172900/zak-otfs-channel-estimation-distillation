#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
DISTILL_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
for candidate in (DISTILL_ROOT / "src", REPO_ROOT / "src", REPO_ROOT / "physics_novelty" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zakotfs.params import load_config
from zakotfs_distill.evaluation import run_distill_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distillation evaluation sweeps.")
    parser.add_argument("--config", type=Path, default=DISTILL_ROOT / "configs" / "distill_eval_full.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_distill_evaluation(config, checkpoint_path=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
