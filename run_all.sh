#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python scripts/reproduce_all.py --mode full
