from __future__ import annotations

from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config
from zakotfs_physics.baseline_bridge import load_frozen_baseline
from zakotfs_physics.evaluation import (
    estimate_phase5_channels,
    evaluate_phase5_ber_point,
    evaluate_phase5_nmse_point,
    normalize_phase5_methods,
    run_phase5_full_evaluation,
)
from zakotfs_physics.model import instantiate_phase2_model


PHYSICS_ROOT = Path(__file__).resolve().parents[1]
PHASE5_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase5_smoke.yaml"


def _prepare_phase5_smoke_config(tmp_path: Path):
    config = load_config(PHASE5_SMOKE_CONFIG_PATH)
    config.raw["paths"]["results_dir"] = str((tmp_path / "physics_results").resolve())
    config.raw["paths"]["logs_dir"] = str((tmp_path / "physics_logs").resolve())
    config.raw["paths"]["report_dir"] = str((tmp_path / "physics_report").resolve())
    config.raw["device"] = "cpu"
    return config


def test_phase5_method_naming_and_selection_correctness(tmp_path: Path) -> None:
    config = _prepare_phase5_smoke_config(tmp_path)
    methods = normalize_phase5_methods(["conventional", "baseline", "baseline", "refined", "blended", "perfect"])
    assert methods == ["conventional", "baseline", "refined", "blended", "perfect"]

    model = instantiate_phase2_model(config)
    baseline_model, _, _ = load_frozen_baseline(config)
    frame = simulate_frame(config, "bpsk", 15.0, 5.0, rng=np.random.default_rng(config.seed + 5))
    estimates = estimate_phase5_channels(frame, config, methods, novelty_model=model, baseline_model=baseline_model)

    assert tuple(estimates.keys()) == tuple(methods)
    assert all(value.shape == frame.h_eff_support.shape for value in estimates.values())


def test_phase5_single_point_nmse_schema(tmp_path: Path) -> None:
    config = _prepare_phase5_smoke_config(tmp_path)
    model = instantiate_phase2_model(config)
    baseline_model, _, _ = load_frozen_baseline(config)

    df = evaluate_phase5_nmse_point(
        config,
        modulation="bpsk",
        data_snr_db=15.0,
        pdr_db=5.0,
        realizations=2,
        methods=["conventional", "baseline", "refined", "blended", "perfect"],
        novelty_model=model,
        baseline_model=baseline_model,
    )

    expected_columns = {"method", "nmse", "ci_low", "ci_high", "realizations", "modulation", "data_snr_db", "pdr_db"}
    assert expected_columns.issubset(set(df.columns))
    assert set(df["method"]) == {"conventional", "baseline", "refined", "blended", "perfect"}


def test_phase5_single_point_ber_schema(tmp_path: Path) -> None:
    config = _prepare_phase5_smoke_config(tmp_path)
    model = instantiate_phase2_model(config)
    baseline_model, _, _ = load_frozen_baseline(config)

    df = evaluate_phase5_ber_point(
        config,
        modulation="bpsk",
        data_snr_db=18.0,
        pdr_db=5.0,
        methods=["conventional", "baseline", "refined", "blended", "perfect"],
        target_bit_errors=1,
        max_bits=2048,
        min_frames=1,
        solver="dense",
        novelty_model=model,
        baseline_model=baseline_model,
    )

    expected_columns = {"method", "ber", "ci_low", "ci_high", "frames", "bits", "errors", "solver"}
    assert expected_columns.issubset(set(df.columns))
    assert set(df["method"]) == {"conventional", "baseline", "refined", "blended", "perfect"}


def test_phase5_smoke_bundle_writes_outputs(tmp_path: Path) -> None:
    config = _prepare_phase5_smoke_config(tmp_path)
    model = instantiate_phase2_model(config)
    baseline_model, _, _ = load_frozen_baseline(config)

    outputs = run_phase5_full_evaluation(config, novelty_model=model, baseline_model=baseline_model)

    assert set(outputs.keys()) == {"nmse_vs_pdr", "nmse_vs_snr", "ber_vs_pdr", "ber_vs_snr"}
    out_dir = Path(config.root / config.raw["paths"]["results_dir"])
    expected_prefixes = (
        "phase5_nmse_vs_pdr_smoke",
        "phase5_nmse_vs_snr_smoke",
        "phase5_ber_vs_pdr_smoke",
        "phase5_ber_vs_snr_smoke",
    )
    for prefix in expected_prefixes:
        assert (out_dir / f"{prefix}.csv").exists()
        assert (out_dir / f"{prefix}.json").exists()
        assert (out_dir / f"{prefix}.png").exists()
