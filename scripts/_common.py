from __future__ import annotations

import argparse
from pathlib import Path

from zakotfs.params import load_config


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    return parser


def load_cfg(path: Path):
    return load_config(path)
