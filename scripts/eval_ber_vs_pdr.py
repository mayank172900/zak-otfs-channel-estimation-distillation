from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs.evaluation import evaluate_ber_vs_pdr, save_eval_outputs
from zakotfs.training import load_cnn_checkpoint


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] eval_ber_vs_pdr config={args.config} mode={args.mode}", flush=True)
    model = load_cnn_checkpoint(cfg)
    df = evaluate_ber_vs_pdr(cfg, model, mode=args.mode)
    save_eval_outputs(df, "ber", f"ber_vs_pdr_{args.mode}", cfg)
    print("[script] eval_ber_vs_pdr complete", flush=True)


if __name__ == "__main__":
    main()
