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
from zakotfs_distill.benchmark import benchmark_student_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the teacher and distilled student models.")
    parser.add_argument("--config", type=Path, default=DISTILL_ROOT / "configs" / "distill_eval_full.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_path = benchmark_student_models(config, checkpoint_path=args.checkpoint)
    print(f"[distill-benchmark] output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
