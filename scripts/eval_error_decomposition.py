from __future__ import annotations

from dataclasses import asdict

from _common import common_parser, load_cfg
from zakotfs.diagnostics import error_decomposition
from zakotfs.plotting import save_heatmaps
from zakotfs.utils import results_dir, save_json


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] eval_error_decomposition config={args.config} mode={args.mode}", flush=True)
    components, summary = error_decomposition(cfg)
    out_dir = results_dir(cfg)
    save_heatmaps(
        [
            components["truth"],
            components["read_off"],
            components["effective_term"],
            components["alias_term"],
            components["data_term"],
            components["noise_term"],
        ],
        ["Truth", "Read-off", "Effective", "Alias", "Data interference", "Noise"],
        out_dir / f"error_decomposition_{args.mode}.png",
    )
    save_json(out_dir / f"error_decomposition_{args.mode}.json", asdict(summary))
    print(f"[script] saved error decomposition artifacts in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
