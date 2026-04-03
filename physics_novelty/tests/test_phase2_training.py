from __future__ import annotations

from pathlib import Path

import torch

from zakotfs.params import load_config
from zakotfs.utils import load_json
from zakotfs_physics.dataset import PHASE2_INPUT_CHANNEL_NAMES, Phase2TrainingDataset, generate_phase1_dataset
from zakotfs_physics.evaluation import infer_phase2_checkpoint, infer_phase2_support
from zakotfs_physics.model import instantiate_phase2_model, predict_phase2
from zakotfs_physics.training import load_phase2_checkpoint, phase2_loss, train_phase2


PHYSICS_ROOT = Path(__file__).resolve().parents[1]
PHASE1_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase1_smoke.yaml"
PHASE2_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase2_smoke.yaml"


def _prepare_smoke_setup(tmp_path: Path):
    phase1_config = load_config(PHASE1_SMOKE_CONFIG_PATH)
    phase1_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase1_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase1_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase1_config.raw["device"] = "cpu"
    manifest_path = generate_phase1_dataset(phase1_config, force=True)

    phase2_config = load_config(PHASE2_SMOKE_CONFIG_PATH)
    phase2_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase2_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase2_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase2_config.raw["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase2_config.raw["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase2_config.raw["smoke"]["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase2_config.raw["smoke"]["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase2_config.raw["device"] = "cpu"
    return manifest_path, phase2_config


def test_phase2_dataset_channel_layout(tmp_path: Path) -> None:
    manifest_path, _ = _prepare_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    sample = dataset[0]

    assert dataset.input_channel_names == PHASE2_INPUT_CHANNEL_NAMES
    assert sample["inputs"].shape[0] == 6
    assert sample["h_true"].shape[0] == 2
    assert sample["h_base"].shape[0] == 2
    assert torch.allclose(sample["inputs"][4:6], sample["h_base"])
    assert "h_phys_true" in sample


def test_phase2_model_output_shapes(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    model = instantiate_phase2_model(config)
    batch = dataset[0]["inputs"].unsqueeze(0)
    predictions = predict_phase2(model, batch)

    assert predictions["delta"].shape == (1, 2, *batch.shape[-2:])
    assert predictions["uncertainty"].shape == (1, 1, *batch.shape[-2:])
    assert predictions["h_hat"].shape == (1, 2, *batch.shape[-2:])
    assert predictions["confidence"].shape == (1, 1, *batch.shape[-2:])


def test_phase2_forward_and_loss_are_finite(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    model = instantiate_phase2_model(config)
    batch = {
        "inputs": torch.stack([dataset[0]["inputs"], dataset[1]["inputs"]], dim=0),
        "h_true": torch.stack([dataset[0]["h_true"], dataset[1]["h_true"]], dim=0),
        "h_base": torch.stack([dataset[0]["h_base"], dataset[1]["h_base"]], dim=0),
    }
    predictions = predict_phase2(model, batch["inputs"])
    losses = phase2_loss(predictions, batch["h_true"], lambda_unc=0.1, lambda_delta=0.05, beta=0.01)

    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_rec"])
    assert torch.isfinite(losses["loss_unc"])
    assert torch.isfinite(losses["loss_delta"])
    assert float(losses["loss_delta"].detach().cpu()) >= 0.0


def test_phase2_inference_helper_runs_on_smoke_data(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    model = instantiate_phase2_model(config)
    sample_inputs = dataset[0]["inputs"]

    outputs = infer_phase2_support(model, sample_inputs)

    assert outputs["h_hat"].shape == (2, *sample_inputs.shape[-2:])
    assert outputs["uncertainty"].shape == (1, *sample_inputs.shape[-2:])
    assert torch.all(outputs["confidence"] > 0.0)


def test_phase2_smoke_training_run_completes(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    checkpoint_path = train_phase2(config, train_manifest_path=manifest_path, val_manifest_path=manifest_path)
    history_path = Path(config.root / config.raw["paths"]["logs_dir"] / "train_history_phase2_smoke.json")

    assert checkpoint_path.exists()
    assert history_path.exists()
    history = load_json(history_path)
    assert len(history["history"]) == 1

    model = load_phase2_checkpoint(config, checkpoint_path=checkpoint_path)
    dataset = Phase2TrainingDataset(manifest_path)
    outputs = infer_phase2_checkpoint(config, dataset[0]["inputs"], checkpoint_path=checkpoint_path)
    assert model is not None
    assert outputs["h_hat"].shape == (2, *dataset[0]["inputs"].shape[-2:])
