from __future__ import annotations

from pathlib import Path

import numpy as np

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config
from zakotfs_physics.baseline_bridge import baseline_estimate_support, load_frozen_baseline
from zakotfs_physics.dataset import Phase1MemmapDataset, generate_phase1_dataset, load_phase1_manifest, open_phase1_arrays
from zakotfs_physics.physics_targets import forward_physics_target


PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase1_smoke.yaml"


def _smoke_config(tmp_path: Path):
    config = load_config(SMOKE_CONFIG_PATH)
    config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    return config


def _quantize_complex(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex64)
    return (arr.real.astype(np.float16).astype(np.float32) + 1j * arr.imag.astype(np.float16).astype(np.float32)).astype(np.complex64)


def test_phase1_manifest_and_shapes(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    manifest_path = generate_phase1_dataset(config, force=True)
    manifest = load_phase1_manifest(manifest_path)
    arrays = open_phase1_arrays(manifest_path)
    dataset = Phase1MemmapDataset(manifest_path)

    assert manifest["split"] == "smoke"
    assert manifest["size"] == 6
    assert manifest["per_pdr"] == 1
    assert manifest["storage_dtype"] == "float16"
    assert manifest["tensor_names"] == ["H_obs", "H_true", "H_thr", "H_base", "H_phys_true"]
    assert arrays["h_obs_re"].shape == (6, *manifest["shape"])
    assert arrays["h_obs_re"].dtype == np.float16
    assert arrays["h_phys_true_im"].dtype == np.float16
    assert arrays["pdr_db"].shape == (6,)
    assert arrays["sample_seed"].dtype == np.int64
    sample = dataset[0]
    assert sample["H_obs"].shape == tuple(manifest["shape"])
    assert sample["H_base"].shape == tuple(manifest["shape"])
    assert sample["sample_index"] == 0


def test_phase1_regeneration_is_deterministic(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    manifest_path = generate_phase1_dataset(config, force=True)
    first_manifest = load_phase1_manifest(manifest_path)
    first_snapshot = {name: np.array(array) for name, array in open_phase1_arrays(manifest_path).items()}

    manifest_path = generate_phase1_dataset(config, force=True)
    second_manifest = load_phase1_manifest(manifest_path)
    second_snapshot = {name: np.array(array) for name, array in open_phase1_arrays(manifest_path).items()}

    assert first_manifest == second_manifest
    for name in first_snapshot:
        assert np.array_equal(first_snapshot[name], second_snapshot[name]), name


def test_phase1_stored_tensors_match_frame_generation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    manifest_path = generate_phase1_dataset(config, force=True)
    manifest = load_phase1_manifest(manifest_path)
    dataset = Phase1MemmapDataset(manifest_path)

    for index in (0, len(dataset) - 1):
        sample = dataset[index]
        frame = simulate_frame(
            config,
            str(manifest["modulation"]),
            float(manifest["snr_db"]),
            float(sample["pdr_db"]),
            np.random.default_rng(int(sample["sample_seed"])),
        )
        expected_phys = forward_physics_target(frame.support_true, frame.spread_dd, frame.E_p, config)
        assert np.array_equal(sample["H_obs"], _quantize_complex(frame.support_input))
        assert np.array_equal(sample["H_true"], _quantize_complex(frame.support_true))
        assert np.array_equal(sample["H_thr"], _quantize_complex(frame.h_hat_support_thr))
        assert np.array_equal(sample["H_phys_true"], _quantize_complex(expected_phys))


def test_phase1_stored_baseline_matches_frozen_checkpoint(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    manifest_path = generate_phase1_dataset(config, force=True)
    manifest = load_phase1_manifest(manifest_path)
    dataset = Phase1MemmapDataset(manifest_path)
    backbone, _, _ = load_frozen_baseline(config)

    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        frame = simulate_frame(
            config,
            str(manifest["modulation"]),
            float(manifest["snr_db"]),
            float(sample["pdr_db"]),
            np.random.default_rng(int(sample["sample_seed"])),
        )
        expected = baseline_estimate_support(backbone, frame.support_input)
        assert np.array_equal(sample["H_base"], _quantize_complex(expected))
