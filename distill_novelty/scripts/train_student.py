#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
DISTILL_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
for candidate in (DISTILL_ROOT / "src", REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zakotfs.params import load_config
from zakotfs_distill.training import train_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight distilled student model.")
    parser.add_argument("--config", type=Path, default=DISTILL_ROOT / "configs" / "distill_train.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = train_student(config)
    print(f"[distill-train] checkpoint={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
