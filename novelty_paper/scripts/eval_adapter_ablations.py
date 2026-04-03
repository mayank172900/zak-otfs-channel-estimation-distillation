from __future__ import annotations

from pathlib import Path

from _common import common_parser, load_cfg
from zakotfs_novelty.adapter import load_frozen_backbone
from zakotfs_novelty.evaluation import (
    evaluate_ber_vs_pdr,
    evaluate_ber_vs_snr,
    evaluate_nmse_vs_pdr,
    evaluate_nmse_vs_snr,
    save_eval_outputs,
)
from zakotfs_novelty.training import load_adapter_checkpoint


def _available_methods(cfg, adapters: dict[str, object]) -> dict[str, list[str]]:
    available = ["conventional", "cnn"] + sorted(adapters.keys())
    available_with_perfect = available + ["perfect"]
    return {
        "nmse_vs_pdr": [method for method in cfg.raw["evaluation"]["nmse_vs_pdr"]["methods"] if method in available],
        "nmse_vs_snr": [method for method in cfg.raw["evaluation"]["nmse_vs_snr"]["methods"] if method in available],
        "ber_vs_pdr": [method for method in cfg.raw["evaluation"]["ber_vs_pdr"]["methods"] if method in available_with_perfect],
        "ber_vs_snr": [method for method in cfg.raw["evaluation"]["ber_vs_snr"]["methods"] if method in available_with_perfect],
    }


def main() -> None:
    parser = common_parser("novelty_paper/configs/adapter_eval.yaml")
    parser.add_argument("--sections", nargs="+", choices=["nmse_pdr", "nmse_snr", "ber_pdr", "ber_snr"], default=["nmse_pdr", "nmse_snr"])
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--generic-checkpoint", type=Path, default=None)
    parser.add_argument("--fb-lara-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    backbone_checkpoint = Path(args.backbone_checkpoint) if args.backbone_checkpoint else None
    backbone, _ = load_frozen_backbone(cfg, checkpoint_path=backbone_checkpoint)
    adapters = {}
    generic_path = args.generic_checkpoint or Path(str(cfg.raw["adapter"]["checkpoint_paths"]["generic"]))
    fb_lara_path = args.fb_lara_checkpoint or Path(str(cfg.raw["adapter"]["checkpoint_paths"]["fb_lara"]))
    if not generic_path.is_absolute():
        generic_path = (cfg.root / generic_path).resolve()
    if not fb_lara_path.is_absolute():
        fb_lara_path = (cfg.root / fb_lara_path).resolve()
    if generic_path.exists():
        adapters["generic"] = load_adapter_checkpoint(cfg, generic_path, adapter_kind="generic")
    if fb_lara_path.exists():
        adapters["fb_lara"] = load_adapter_checkpoint(cfg, fb_lara_path, adapter_kind="fb_lara")
    methods = _available_methods(cfg, adapters)
    cfg.raw["evaluation"]["nmse_vs_pdr"]["methods"] = methods["nmse_vs_pdr"]
    cfg.raw["evaluation"]["nmse_vs_snr"]["methods"] = methods["nmse_vs_snr"]
    cfg.raw["evaluation"]["ber_vs_pdr"]["methods"] = methods["ber_vs_pdr"]
    cfg.raw["evaluation"]["ber_vs_snr"]["methods"] = methods["ber_vs_snr"]
    print(f"[script] eval sections={args.sections} methods={methods}", flush=True)
    if "nmse_pdr" in args.sections:
        df = evaluate_nmse_vs_pdr(cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "nmse", f"adapter_nmse_vs_pdr_{args.mode}", cfg)
    if "nmse_snr" in args.sections:
        df = evaluate_nmse_vs_snr(cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "nmse", f"adapter_nmse_vs_snr_{args.mode}", cfg)
    if "ber_pdr" in args.sections:
        df = evaluate_ber_vs_pdr(cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "ber", f"adapter_ber_vs_pdr_{args.mode}", cfg)
    if "ber_snr" in args.sections:
        df = evaluate_ber_vs_snr(cfg, backbone, adapters, mode=args.mode)
        save_eval_outputs(df, "ber", f"adapter_ber_vs_snr_{args.mode}", cfg)


if __name__ == "__main__":
    main()
