from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zakotfs.dataset import generate_dataset
from zakotfs.evaluation import (
    compare_to_anchors,
    evaluate_ber_vs_pdr,
    evaluate_ber_vs_snr,
    evaluate_nmse_vs_pdr,
    evaluate_nmse_vs_snr,
    save_eval_outputs,
)
from zakotfs.params import load_config
from zakotfs.training import load_cnn_checkpoint, train_cnn
from zakotfs.utils import results_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows-friendly end-to-end reproduction runner.")
    parser.add_argument("--train-config", default="configs/train_fast.yaml")
    parser.add_argument("--eval-config", default="configs/eval_fast.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["nmse_pdr", "nmse_snr", "ber_pdr", "ber_snr"],
        default=["nmse_pdr", "nmse_snr", "ber_pdr", "ber_snr"],
    )
    args = parser.parse_args()

    train_cfg = load_config(args.train_config)
    eval_cfg = load_config(args.eval_config)

    checkpoint_path = None
    if not args.skip_train:
        train_path = generate_dataset(train_cfg, "train", mode=args.mode, force=False)
        val_path = generate_dataset(train_cfg, "val", mode=args.mode, force=False)
        checkpoint_path = train_cnn(train_cfg, train_path=train_path, val_path=val_path, mode=args.mode)

    model = load_cnn_checkpoint(eval_cfg, checkpoint_path)
    section_set = set(args.sections)
    outputs: dict[str, pd.DataFrame] = {}
    if "nmse_pdr" in section_set:
        outputs["nmse_pdr"] = evaluate_nmse_vs_pdr(eval_cfg, model, mode=args.mode)
        save_eval_outputs(outputs["nmse_pdr"], "nmse", f"nmse_vs_pdr_{args.mode}", eval_cfg)
    if "nmse_snr" in section_set:
        outputs["nmse_snr"] = evaluate_nmse_vs_snr(eval_cfg, model, mode=args.mode)
        save_eval_outputs(outputs["nmse_snr"], "nmse", f"nmse_vs_snr_{args.mode}", eval_cfg)
    if "ber_pdr" in section_set:
        outputs["ber_pdr"] = evaluate_ber_vs_pdr(eval_cfg, model, mode=args.mode)
        save_eval_outputs(outputs["ber_pdr"], "ber", f"ber_vs_pdr_{args.mode}", eval_cfg)
    if "ber_snr" in section_set:
        outputs["ber_snr"] = evaluate_ber_vs_snr(eval_cfg, model, mode=args.mode)
        save_eval_outputs(outputs["ber_snr"], "ber", f"ber_vs_snr_{args.mode}", eval_cfg)

    if {"nmse_pdr", "ber_pdr", "ber_snr"}.issubset(outputs):
        summary = compare_to_anchors(eval_cfg, outputs["nmse_pdr"], outputs["ber_pdr"], outputs["ber_snr"])
        summary_path = results_dir(eval_cfg) / f"reproduction_summary_{args.mode}.json"
        save_json(summary_path, summary)
        print(f"[repro] saved summary={summary_path}", flush=True)
    else:
        print(f"[repro] partial run completed sections={sorted(section_set)}", flush=True)


if __name__ == "__main__":
    main()
