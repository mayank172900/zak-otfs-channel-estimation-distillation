from __future__ import annotations

from pathlib import Path

from _common import common_parser, load_cfg
from zakotfs.dataset import generate_dataset
from zakotfs.evaluation import (
    compare_to_anchors,
    evaluate_ber_vs_pdr,
    evaluate_ber_vs_snr,
    evaluate_nmse_vs_pdr,
    evaluate_nmse_vs_snr,
    save_eval_outputs,
)
from zakotfs.training import load_cnn_checkpoint, train_cnn
from zakotfs.utils import results_dir, save_json


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/train.yaml")
    args = parser.parse_args()
    train_cfg = load_cfg(Path("configs/train.yaml"))
    eval_cfg = load_cfg(Path("configs/eval.yaml"))
    print(f"[script] reproduce_all mode={args.mode}", flush=True)
    train_path = generate_dataset(train_cfg, "train", mode=args.mode, force=False)
    val_path = generate_dataset(train_cfg, "val", mode=args.mode, force=False)
    ckpt = train_cnn(train_cfg, train_path=train_path, val_path=val_path, mode=args.mode)
    model = load_cnn_checkpoint(eval_cfg, ckpt)
    print("[script] running NMSE/BER evaluations", flush=True)
    nmse_pdr = evaluate_nmse_vs_pdr(eval_cfg, model, mode=args.mode)
    nmse_snr = evaluate_nmse_vs_snr(eval_cfg, model, mode=args.mode)
    ber_pdr = evaluate_ber_vs_pdr(eval_cfg, model, mode=args.mode)
    ber_snr = evaluate_ber_vs_snr(eval_cfg, model, mode=args.mode)
    save_eval_outputs(nmse_pdr, "nmse", f"nmse_vs_pdr_{args.mode}", eval_cfg)
    save_eval_outputs(nmse_snr, "nmse", f"nmse_vs_snr_{args.mode}", eval_cfg)
    save_eval_outputs(ber_pdr, "ber", f"ber_vs_pdr_{args.mode}", eval_cfg)
    save_eval_outputs(ber_snr, "ber", f"ber_vs_snr_{args.mode}", eval_cfg)
    anchor_summary = compare_to_anchors(eval_cfg, nmse_pdr, ber_pdr, ber_snr)
    save_json(results_dir(eval_cfg) / f"anchor_comparison_{args.mode}.json", anchor_summary)
    print(f"[script] saved anchor summary to {results_dir(eval_cfg) / f'anchor_comparison_{args.mode}.json'}", flush=True)

if __name__ == "__main__":
    main()
