from pathlib import Path

import pandas as pd

from zakotfs.compat import strict_zip
from zakotfs.plotting import curve_plot_columns, save_curve_plot


def test_curve_plot_separates_modulations_when_present(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"data_snr_db": 18.0, "ber": 2.5e-2, "method": "conventional", "modulation": "bpsk"},
            {"data_snr_db": 18.0, "ber": 4.7e-4, "method": "cnn", "modulation": "bpsk"},
            {"data_snr_db": 18.0, "ber": 3.0e-4, "method": "perfect", "modulation": "bpsk"},
            {"data_snr_db": 18.0, "ber": 1.6e-1, "method": "conventional", "modulation": "8qam_cross"},
            {"data_snr_db": 18.0, "ber": 4.0e-2, "method": "cnn", "modulation": "8qam_cross"},
            {"data_snr_db": 18.0, "ber": 2.5e-2, "method": "perfect", "modulation": "8qam_cross"},
        ]
    )
    hue_col, style_col = curve_plot_columns(df)
    assert hue_col == "method"
    assert style_col == "modulation"
    assert len(set(strict_zip(df[hue_col], df[style_col]))) == 6

    out_path = tmp_path / "ber_vs_snr.png"
    save_curve_plot(df, "data_snr_db", "ber", hue_col, "BER vs SNR", out_path, logy=True, style_col=style_col)
    assert out_path.exists()
