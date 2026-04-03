from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import common_parser, load_cfg
from zakotfs.channel import PhysicalChannel, effective_channel_support_fast, effective_channel_support_reference
from zakotfs.metrics import nmse
from zakotfs.plotting import save_heatmaps
from zakotfs.utils import save_json


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/system.yaml")
    parser.add_argument("--case", choices=["integer", "fractional_delay", "fractional_doppler", "mixed"], default="mixed")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    cases = {
        "integer": (0.0, 0.0),
        "fractional_delay": (0.71e-6, 0.0),
        "fractional_doppler": (0.0, 220.0),
        "mixed": (1.09e-6, -300.0),
    }
    tau, nu = cases[args.case]
    channel = PhysicalChannel(
        gains=np.array([1.0 + 0.0j], dtype=np.complex64),
        delays_s=np.array([tau], dtype=float),
        dopplers_hz=np.array([nu], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )
    h_ref = effective_channel_support_reference(channel, cfg)
    h_fast = effective_channel_support_fast(channel, cfg)
    out_dir = Path("results/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"single_path_{args.case}"
    save_heatmaps(
        [h_ref, h_fast, h_fast - h_ref],
        ["Reference", "Fast", "Difference"],
        out_dir / f"{stem}.png",
    )
    save_json(
        out_dir / f"{stem}.json",
        {
            "case": args.case,
            "delay_s": tau,
            "doppler_hz": nu,
            "nmse_fast_vs_reference": float(nmse(h_fast, h_ref)),
            "reference_peak": list(map(int, np.unravel_index(np.argmax(np.abs(h_ref)), h_ref.shape))),
            "fast_peak": list(map(int, np.unravel_index(np.argmax(np.abs(h_fast)), h_fast.shape))),
        },
    )


if __name__ == "__main__":
    main()
