from __future__ import annotations

from pathlib import Path

from _common import common_parser, load_cfg
from zakotfs_novelty.adapter import generate_adapter_dataset
from zakotfs_novelty.evaluation import (
    evaluate_ber_vs_pdr,
    evaluate_ber_vs_snr,
    evaluate_nmse_vs_pdr,
    evaluate_nmse_vs_snr,
    save_eval_outputs,
)
from zakotfs_novelty.training import load_adapter_checkpoint, train_adapter
from zakotfs_novelty.adapter import load_frozen_backbone


def _apply_available_methods(cfg, adapters: dict[str, object]) -> None:
    available = ["conventional", "cnn"] + sorted(adapters.keys())
    available_with_perfect = available + ["perfect"]
    for section in ["nmse_vs_pdr", "nmse_vs_snr"]:
        cfg.raw["evaluation"][section]["methods"] = [
            method for method in cfg.raw["evaluation"][section]["methods"] if method in available
        ]
    for section in ["ber_vs_pdr", "ber_vs_snr"]:
        cfg.raw["evaluation"][section]["methods"] = [
            method for method in cfg.raw["evaluation"][section]["methods"] if method in available_with_perfect
        ]


def main() -> None:
    parser = common_parser("novelty_paper/configs/adapter_train.yaml")
    parser.add_argument("--eval-config", type=Path, default=Path("novelty_paper/configs/adapter_eval.yaml"))
    parser.add_argument("--train-kinds", nargs="+", choices=["generic", "fb_lara"], default=["generic", "fb_lara"])
    parser.add_argument("--sections", nargs="+", choices=["nmse_pdr", "nmse_snr", "ber_pdr", "ber_snr"], default=["nmse_pdr", "nmse_snr"])
    parser.add_argument("--backbone-checkpoint", default=None)
    args = parser.parse_args()
    train_cfg = load_cfg(args.config)
    eval_cfg = load_cfg(args.eval_config)
    if args.backbone_checkpoint:
        train_cfg.raw.setdefault("backbone", {})["checkpoint_path"] = str(args.backbone_checkpoint)
        eval_cfg.raw.setdefault("backbone", {})["checkpoint_path"] = str(args.backbone_checkpoint)
    train_path = generate_adapter_dataset(train_cfg, "train", force=False)
    val_path = generate_adapter_dataset(train_cfg, "val", force=False)
    adapter_paths = {}
    for kind in args.train_kinds:
        adapter_paths[kind] = train_adapter(train_cfg, train_path, val_path, adapter_kind=kind, mode=args.mode)
    backbone, _ = load_frozen_backbone(eval_cfg)
    adapters = {kind: load_adapter_checkpoint(eval_cfg, Path(path), adapter_kind=kind) for kind, path in adapter_paths.items()}
    _apply_available_methods(eval_cfg, adapters)
    if "nmse_pdr" in args.sections:
        df = evaluate_nmse_vs_pdr(eval_cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "nmse", f"adapter_nmse_vs_pdr_{args.mode}", eval_cfg)
    if "nmse_snr" in args.sections:
        df = evaluate_nmse_vs_snr(eval_cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "nmse", f"adapter_nmse_vs_snr_{args.mode}", eval_cfg)
    if "ber_pdr" in args.sections:
        df = evaluate_ber_vs_pdr(eval_cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "ber", f"adapter_ber_vs_pdr_{args.mode}", eval_cfg)
    if "ber_snr" in args.sections:
        df = evaluate_ber_vs_snr(eval_cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "ber", f"adapter_ber_vs_snr_{args.mode}", eval_cfg)


if __name__ == "__main__":
    main()
