from pathlib import Path

import numpy as np

from zakotfs.channel import effective_channel_support, sample_vehicular_a_channel
from zakotfs.estimators import pilot_cancellation_with_config
from zakotfs.mmse import mmse_dense
from zakotfs.modulation import hard_demodulate
from zakotfs.operators import apply_support_operator
from zakotfs.params import load_config
from zakotfs.utils import snr_to_noise_variance
from zakotfs.waveform import data_symbols, data_waveform, spread_pilot, superimposed_frame


def test_perfect_csi_detection_is_nearly_pdr_invariant_after_exact_cancellation():
    cfg = load_config(Path("configs/system.yaml"))
    rng = np.random.default_rng(cfg.seed + 444)
    symbols, bits = data_symbols("bpsk", cfg, rng)
    data_dd = data_waveform(symbols)
    spread_dd = spread_pilot(cfg)
    E_d = 1.0
    noise_variance = snr_to_noise_variance(E_d, 18.0, cfg.Q)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal((cfg.M, cfg.N)) + 1j * rng.standard_normal((cfg.M, cfg.N))
    )
    channel = sample_vehicular_a_channel(cfg, rng)
    h_eff_support = effective_channel_support(channel, cfg)
    y_data_ideal = apply_support_operator(h_eff_support, np.sqrt(E_d) * data_dd, cfg)

    pdrs = [0.0, 5.0, 20.0, 35.0]
    cancelled_frames = []
    bers = []
    for pdr_db in pdrs:
        E_p = 10.0 ** (pdr_db / 10.0) * E_d
        x_dd = superimposed_frame(data_dd, spread_dd, E_d=E_d, E_p=E_p)
        y_dd = apply_support_operator(h_eff_support, x_dd, cfg) + noise
        y_data = pilot_cancellation_with_config(y_dd, h_eff_support, np.sqrt(E_p) * spread_dd, cfg)
        cancelled_frames.append(y_data)
        x_hat = mmse_dense(y_data, h_eff_support, noise_variance, cfg, E_d=E_d)
        _, bits_hat = hard_demodulate(x_hat * np.sqrt(cfg.Q) / np.sqrt(E_d), "bpsk")
        bers.append(np.count_nonzero(bits_hat != bits) / bits.size)

    reference = y_data_ideal + noise
    for y_data in cancelled_frames:
        rel_residual = np.linalg.norm(y_data - reference) / np.linalg.norm(reference)
        assert rel_residual < 5e-5
    assert max(bers) - min(bers) <= (1.0 / bits.size)
