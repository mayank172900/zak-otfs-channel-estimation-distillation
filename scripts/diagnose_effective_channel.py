from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zakotfs.channel import PhysicalChannel, effective_channel_support_fast, effective_channel_support_reference, sample_vehicular_a_channel
from zakotfs.metrics import nmse
from zakotfs.operators import apply_support_operator
from zakotfs.params import load_config
from zakotfs.utils import results_dir, save_json


def _single_path(delay_s: float, doppler_hz: float) -> PhysicalChannel:
    return PhysicalChannel(
        gains=np.array([1.0 + 0.0j], dtype=np.complex64),
        delays_s=np.array([delay_s], dtype=float),
        dopplers_hz=np.array([doppler_hz], dtype=float),
        thetas_rad=np.array([0.0], dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fast and reference effective-channel constructions.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--num-random", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 900)
    x = (rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))).astype(np.complex64)

    rows: list[dict[str, float | int | str]] = []
    controlled = [
        ("single_path_integer", _single_path(0.0, 0.0)),
        ("single_path_fractional", _single_path(0.71e-6, 220.0)),
    ]
    for case_name, channel in controlled:
        h_fast = effective_channel_support_fast(channel, cfg)
        h_ref = effective_channel_support_reference(channel, cfg)
        rows.append(
            {
                "case": case_name,
                "index": 0,
                "support_nmse": float(nmse(h_fast, h_ref)),
                "output_nmse": float(nmse(apply_support_operator(h_fast, x, cfg), apply_support_operator(h_ref, x, cfg))),
            }
        )

    for idx in range(args.num_random):
        channel = sample_vehicular_a_channel(cfg, rng)
        h_fast = effective_channel_support_fast(channel, cfg)
        h_ref = effective_channel_support_reference(channel, cfg)
        rows.append(
            {
                "case": "vehicular_a_random",
                "index": idx,
                "support_nmse": float(nmse(h_fast, h_ref)),
                "output_nmse": float(nmse(apply_support_operator(h_fast, x, cfg), apply_support_operator(h_ref, x, cfg))),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "effective_channel_method_diagnostics.csv"
    json_path = out_dir / "effective_channel_method_diagnostics.json"
    df.to_csv(csv_path, index=False)
    save_json(
        json_path,
        {
            "records": df.to_dict(orient="records"),
            "summary": {
                "support_nmse_mean": float(df["support_nmse"].mean()),
                "support_nmse_max": float(df["support_nmse"].max()),
                "output_nmse_mean": float(df["output_nmse"].mean()),
                "output_nmse_max": float(df["output_nmse"].max()),
            },
        },
    )
    print(f"[diag] saved csv={csv_path}", flush=True)
    print(f"[diag] saved json={json_path}", flush=True)


if __name__ == "__main__":
    main()
