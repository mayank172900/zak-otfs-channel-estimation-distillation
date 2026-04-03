from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config
from zakotfs.utils import load_json
from zakotfs_physics.dataset import Phase2TrainingDataset, generate_phase1_dataset
from zakotfs_physics.model import instantiate_phase2_model, predict_phase2
from zakotfs_physics.physics_targets import forward_physics_target, forward_physics_target_torch
from zakotfs_physics.training import load_phase2_checkpoint, phase2_loss, train_phase2


PHYSICS_ROOT = Path(__file__).resolve().parents[1]
PHASE1_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase1_smoke.yaml"
PHASE3_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase3_smoke.yaml"


def _prepare_phase3_smoke_setup(tmp_path: Path):
    phase1_config = load_config(PHASE1_SMOKE_CONFIG_PATH)
    phase1_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase1_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase1_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase1_config.raw["device"] = "cpu"
    manifest_path = generate_phase1_dataset(phase1_config, force=True)

    phase3_config = load_config(PHASE3_SMOKE_CONFIG_PATH)
    phase3_config.raw["paths"]["results_dir"] = str((tmp_path / "results").resolve())
    phase3_config.raw["paths"]["logs_dir"] = str((tmp_path / "logs").resolve())
    phase3_config.raw["paths"]["report_dir"] = str((tmp_path / "report").resolve())
    phase3_config.raw["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase3_config.raw["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase3_config.raw["smoke"]["phase2_dataset"]["train_manifest_path"] = str(manifest_path)
    phase3_config.raw["smoke"]["phase2_dataset"]["val_manifest_path"] = str(manifest_path)
    phase3_config.raw["device"] = "cpu"
    return manifest_path, phase3_config


def test_phase3_physics_operator_output_shape(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase3_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    sample = dataset[0]

    output = forward_physics_target_torch(sample["h_true"], config)

    assert output.shape == sample["h_phys_true"].shape


def test_phase3_physics_operator_matches_regenerated_forward_target(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase3_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    manifest = load_json(Path(manifest_path))
    sample = dataset[0]
    frame = simulate_frame(
        config,
        str(manifest["modulation"]),
        float(manifest["snr_db"]),
        float(sample["pdr_db"]),
        np.random.default_rng(int(sample["sample_seed"])),
    )
    expected = forward_physics_target(frame.support_true, frame.spread_dd, frame.E_p, config)
    predicted = forward_physics_target_torch(
        torch.stack(
            [
                torch.from_numpy(frame.support_true.real.astype(np.float32)),
                torch.from_numpy(frame.support_true.imag.astype(np.float32)),
            ],
            dim=0,
        ),
        config,
    )

    assert torch.allclose(predicted, torch.from_numpy(np.stack([expected.real, expected.imag], axis=0)), atol=1.0e-4, rtol=1.0e-4)
    assert torch.allclose(
        forward_physics_target_torch(sample["h_true"], config),
        sample["h_phys_true"],
        atol=5.0e-2,
        rtol=5.0e-2,
    )


def test_phase3_loss_is_finite_and_differentiable(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase3_smoke_setup(tmp_path)
    dataset = Phase2TrainingDataset(manifest_path)
    model = instantiate_phase2_model(config)
    batch_inputs = torch.stack([dataset[0]["inputs"], dataset[1]["inputs"]], dim=0)
    h_true = torch.stack([dataset[0]["h_true"], dataset[1]["h_true"]], dim=0)
    h_phys_true = torch.stack([dataset[0]["h_phys_true"], dataset[1]["h_phys_true"]], dim=0)

    predictions = predict_phase2(model, batch_inputs)
    losses = phase2_loss(
        predictions,
        h_true,
        h_phys_true=h_phys_true,
        config=config,
        lambda_unc=0.1,
        beta=0.01,
        enable_physics_loss=True,
        lambda_phys=0.2,
    )
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_phys"])
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_phase3_smoke_training_run_completes(tmp_path: Path) -> None:
    manifest_path, config = _prepare_phase3_smoke_setup(tmp_path)
    checkpoint_path = train_phase2(config, train_manifest_path=manifest_path, val_manifest_path=manifest_path)
    history_path = Path(config.root / config.raw["paths"]["logs_dir"] / "train_history_phase2_smoke.json")

    assert checkpoint_path.exists()
    history = load_json(history_path)
    assert len(history["history"]) == 1
    assert history["history"][0]["train_phys"] >= 0.0
    model = load_phase2_checkpoint(config, checkpoint_path=checkpoint_path)
    assert model is not None
