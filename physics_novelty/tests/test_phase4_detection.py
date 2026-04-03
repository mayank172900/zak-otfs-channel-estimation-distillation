from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from zakotfs.params import load_config
from zakotfs_physics.baseline_bridge import load_frozen_baseline
from zakotfs_physics.dataset import generate_phase1_dataset
from zakotfs_physics.evaluation import (
    compare_phase4_sample_modes,
    detect_phase4_frame,
    detect_phase4_sample,
    frame_from_phase1_sample,
    fuse_support_estimate,
    infer_phase4_support,
)
from zakotfs_physics.model import instantiate_phase2_model
from zakotfs_physics.dataset import Phase1MemmapDataset
from zakotfs_physics.evaluation import build_phase4_inputs_from_sample
from zakotfs_physics.dataset import load_phase1_manifest


PHYSICS_ROOT = Path(__file__).resolve().parents[1]
PHASE1_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase1_smoke.yaml"
PHASE4_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase4_smoke.yaml"


def _prepare_phase4_smoke_setup(tmp_path: Path):
    phase1_config = load_config(PHASE1_SMOKE_CONFIG_PATH)
    phase1_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase1_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase1_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase1_config.raw["device"] = "cpu"
    manifest_path = generate_phase1_dataset(phase1_config, force=True)

    phase4_config = load_config(PHASE4_SMOKE_CONFIG_PATH)
    phase4_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase4_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase4_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase4_config.raw["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase4_config.raw["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase4_config.raw["smoke"]["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase4_config.raw["smoke"]["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase4_config.raw["device"] = "cpu"
    return manifest_path, phase4_config


def test_phase4_confidence_shape_and_range(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase4_smoke_setup(tmp_path)
    sample = Phase1MemmapDataset(manifest_path)[0]
    inputs = build_phase4_inputs_from_sample(sample)["inputs"]
    model = instantiate_phase2_model(config)

    outputs = infer_phase4_support(model, inputs, config)

    assert outputs["confidence"].shape[0] == 1
    assert outputs["confidence"].shape[-2:] == inputs.shape[-2:]
    assert torch.all(outputs["confidence"] > 0.0)
    assert torch.all(outputs["confidence"] <= 1.0)


def test_phase4_fusion_modes_on_controlled_tensors() -> None:
    h_base = torch.zeros((2, 1, 3), dtype=torch.float32)
    h_hat = torch.ones((2, 1, 3), dtype=torch.float32)
    confidence = torch.tensor([[[0.0, 0.25, 1.0]]], dtype=torch.float32)

    baseline = fuse_support_estimate(h_hat, h_base, confidence, mode="baseline")
    refined = fuse_support_estimate(h_hat, h_base, confidence, mode="refined")
    blended = fuse_support_estimate(h_hat, h_base, confidence, mode="blended")
    thresholded = fuse_support_estimate(h_hat, h_base, confidence, mode="thresholded", confidence_threshold=0.5)

    assert torch.allclose(baseline["h_use"], h_base)
    assert torch.allclose(refined["h_use"], h_hat)
    assert torch.allclose(blended["h_use"], torch.tensor([[[0.0, 0.25, 1.0]], [[0.0, 0.25, 1.0]]]))
    assert torch.allclose(thresholded["h_use"], torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]))


def test_phase4_blended_mode_respects_confidence_scale() -> None:
    h_base = torch.zeros((2, 1, 3), dtype=torch.float32)
    h_hat = torch.ones((2, 1, 3), dtype=torch.float32)
    confidence = torch.tensor([[[0.2, 0.5, 1.0]]], dtype=torch.float32)

    blended = fuse_support_estimate(h_hat, h_base, confidence, mode="blended", confidence_scale=0.25)

    expected_weight = torch.tensor([[[0.05, 0.125, 0.25]]], dtype=torch.float32)
    expected_use = torch.tensor([[[0.05, 0.125, 0.25]], [[0.05, 0.125, 0.25]]], dtype=torch.float32)
    assert torch.allclose(blended["fusion_weight"], expected_weight)
    assert torch.allclose(blended["h_use"], expected_use)


def test_phase4_frame_detection_runs_on_smoke_data(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase4_smoke_setup(tmp_path)
    sample = Phase1MemmapDataset(manifest_path)[0]
    manifest = load_phase1_manifest(manifest_path)
    frame = frame_from_phase1_sample(config, manifest, sample)
    model = instantiate_phase2_model(config)
    baseline_model, _, _ = load_frozen_baseline(config)

    outputs = detect_phase4_frame(frame, config, model, baseline_model=baseline_model, channel_mode="blended")

    assert outputs["bits_hat"].shape == frame.bits.shape
    assert outputs["h_use"].shape == outputs["h_hat"].shape
    assert np.isfinite(outputs["ber"])


def test_phase4_sample_detection_and_mode_comparison_run(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase4_smoke_setup(tmp_path)
    model = instantiate_phase2_model(config)
    manifest = load_phase1_manifest(manifest_path)
    sample = Phase1MemmapDataset(manifest_path)[0]
    frame = frame_from_phase1_sample(config, manifest, sample)

    sample_outputs = detect_phase4_sample(manifest_path, 0, config, model, channel_mode="refined")
    comparison = compare_phase4_sample_modes(manifest_path, 0, config, model, modes=("baseline", "refined", "blended"))

    assert sample_outputs["bits_hat"].shape == frame.bits.shape
    assert sample_outputs["channel_mode"] == "refined"
    assert tuple(comparison.keys()) == ("baseline", "refined", "blended")
    assert all(result["bits_hat"].shape == frame.bits.shape for result in comparison.values())
    assert all(np.isfinite(result["ber"]) for result in comparison.values())
