from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs.evaluation import evaluate_nmse_vs_snr, save_eval_outputs
from zakotfs.training import load_cnn_checkpoint


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] eval_nmse_vs_snr config={args.config} mode={args.mode}", flush=True)
    model = load_cnn_checkpoint(cfg)
    df = evaluate_nmse_vs_snr(cfg, model, mode=args.mode)
    save_eval_outputs(df, "nmse", f"nmse_vs_snr_{args.mode}", cfg)
    print("[script] eval_nmse_vs_snr complete", flush=True)


if __name__ == "__main__":
    main()
