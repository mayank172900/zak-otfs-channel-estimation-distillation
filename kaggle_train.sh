#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m pip install -r requirements.txt
export PYTHONPATH=src
python3 scripts/train_cnn.py --config configs/train.yaml --mode full
