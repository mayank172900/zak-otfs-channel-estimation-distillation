from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import common_parser, load_cfg
from zakotfs.dataset import simulate_frame
from zakotfs.evaluation import cnn_enhance_support
from zakotfs.lattice import derive_support_geometry, enumerate_lattice_points
from zakotfs.plotting import save_heatmaps
from zakotfs.training import load_cnn_checkpoint
from zakotfs.utils import results_dir


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] reproduce_fig2_fig3_fig4 config={args.config} mode={args.mode}", flush=True)
    rng = np.random.default_rng(cfg.seed + 77)
    frame = simulate_frame(cfg, "bpsk", data_snr_db=15.0, pdr_db=5.0, rng=rng)
    out_dir = results_dir(cfg)
    save_heatmaps(
        [frame.x_dd, frame.h_eff, frame.y_dd, frame.ambiguity],
        ["Transmitted frame", "True effective channel", "Received frame", "Cross-ambiguity"],
        out_dir / f"fig2_heatmaps_{args.mode}.png",
    )
    points = enumerate_lattice_points(cfg, radius_k=64, radius_l=96)
    g = derive_support_geometry(cfg)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=12, c="goldenrod")
    rect_x = [g.k_min, g.k_min, g.k_max, g.k_max, g.k_min]
    rect_y = [g.l_min, g.l_max, g.l_max, g.l_min, g.l_min]
    ax.plot(rect_x, rect_y, c="crimson")
    ax.set_title("Twisted lattice and support")
    ax.set_xlabel("k")
    ax.set_ylabel("l")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig3_lattice_{args.mode}.png", dpi=180)
    plt.close(fig)
    model = load_cnn_checkpoint(cfg)
    support_cnn = cnn_enhance_support(model, frame.support_input, next(model.parameters()).device)
    save_heatmaps(
        [frame.support_true, frame.support_input, support_cnn],
        ["True support", "Read-off input", "CNN output"],
        out_dir / f"fig4_cnn_support_{args.mode}.png",
    )
    print(f"[script] saved figure set in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
