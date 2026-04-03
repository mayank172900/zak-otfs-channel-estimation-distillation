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

from zakotfs.channel import effective_channel_support, sample_vehicular_a_channel
from zakotfs.estimators import pilot_cancellation_with_config
from zakotfs.mmse import mmse_dense
from zakotfs.modulation import hard_demodulate
from zakotfs.operators import apply_support_operator
from zakotfs.params import load_config
from zakotfs.utils import results_dir, save_json, snr_to_noise_variance
from zakotfs.waveform import data_symbols, data_waveform, spread_pilot, superimposed_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Check perfect-CSI BER invariance across PDR after exact pilot cancellation.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--modulation", default="bpsk")
    parser.add_argument("--data-snr-db", type=float, default=18.0)
    parser.add_argument("--pdr-db", nargs="+", type=float, default=[0.0, 5.0, 20.0, 25.0, 30.0, 35.0])
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 901)
    symbols, bits = data_symbols(args.modulation, cfg, rng)
    data_dd = data_waveform(symbols)
    spread_dd = spread_pilot(cfg)
    E_d = 1.0
    noise_variance = snr_to_noise_variance(E_d, args.data_snr_db, cfg.Q)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))
    )
    channel = sample_vehicular_a_channel(cfg, rng)
    h_eff_support = effective_channel_support(channel, cfg)
    ideal = apply_support_operator(h_eff_support, np.sqrt(E_d) * data_dd, cfg) + noise

    rows: list[dict[str, float]] = []
    for pdr_db in args.pdr_db:
        E_p = 10.0 ** (pdr_db / 10.0) * E_d
        x_dd = superimposed_frame(data_dd, spread_dd, E_d=E_d, E_p=E_p)
        y_dd = apply_support_operator(h_eff_support, x_dd, cfg) + noise
        y_data = pilot_cancellation_with_config(y_dd, h_eff_support, np.sqrt(E_p) * spread_dd, cfg)
        x_hat = mmse_dense(y_data, h_eff_support, noise_variance, cfg, E_d=E_d)
        _, bits_hat = hard_demodulate(x_hat * np.sqrt(cfg.Q) / np.sqrt(E_d), args.modulation)
        rows.append(
            {
                "pdr_db": float(pdr_db),
                "ber": float(np.count_nonzero(bits_hat != bits) / bits.size),
                "cancel_residual_norm": float(np.linalg.norm(y_data - ideal)),
                "cancel_reference_norm": float(np.linalg.norm(ideal)),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "perfect_csi_pdr_diagnostics.csv"
    json_path = out_dir / "perfect_csi_pdr_diagnostics.json"
    df.to_csv(csv_path, index=False)
    save_json(
        json_path,
        {
            "records": df.to_dict(orient="records"),
            "summary": {
                "ber_min": float(df["ber"].min()),
                "ber_max": float(df["ber"].max()),
                "cancel_residual_norm_max": float(df["cancel_residual_norm"].max()),
            },
        },
    )
    print(f"[diag] saved csv={csv_path}", flush=True)
    print(f"[diag] saved json={json_path}", flush=True)


if __name__ == "__main__":
    main()
