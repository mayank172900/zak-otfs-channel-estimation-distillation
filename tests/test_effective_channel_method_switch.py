from pathlib import Path

import numpy as np

from zakotfs.channel import effective_channel_support_fast, effective_channel_support_reference
from zakotfs.dataset import simulate_frame
from zakotfs.metrics import nmse
from zakotfs.params import load_config


def test_simulate_frame_honors_effective_channel_method_config():
    cfg_fast = load_config(Path("configs/system.yaml"))
    cfg_ref = load_config(Path("configs/system.yaml"))
    cfg_fast.raw["numerics"]["effective_channel_method"] = "fast"
    cfg_ref.raw["numerics"]["effective_channel_method"] = "reference"

    seed = cfg_fast.seed + 222
    frame_fast = simulate_frame(cfg_fast, "bpsk", 15.0, 5.0, np.random.default_rng(seed))
    frame_ref = simulate_frame(cfg_ref, "bpsk", 15.0, 5.0, np.random.default_rng(seed))

    assert np.allclose(frame_fast.physical_channel.gains, frame_ref.physical_channel.gains)
    assert np.allclose(frame_fast.physical_channel.delays_s, frame_ref.physical_channel.delays_s)
    assert np.allclose(frame_fast.physical_channel.dopplers_hz, frame_ref.physical_channel.dopplers_hz)

    expected_fast = effective_channel_support_fast(frame_fast.physical_channel, cfg_fast)
    expected_ref = effective_channel_support_reference(frame_ref.physical_channel, cfg_ref)
    assert np.allclose(frame_fast.h_eff_support, expected_fast, atol=1e-6, rtol=1e-6)
    assert np.allclose(frame_ref.h_eff_support, expected_ref, atol=1e-6, rtol=1e-6)
    assert nmse(frame_fast.h_eff_support, frame_ref.h_eff_support) > 1e-10
