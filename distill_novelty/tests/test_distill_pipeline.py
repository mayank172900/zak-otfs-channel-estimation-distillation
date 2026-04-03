from __future__ import annotations

from pathlib import Path

import torch

from zakotfs.params import load_config
from zakotfs.utils import load_json
from zakotfs_physics.dataset import generate_phase1_dataset
from zakotfs_distill.benchmark import benchmark_student_models
from zakotfs_distill.dataset import DistillDataset
from zakotfs_distill.evaluation import run_distill_evaluation
from zakotfs_distill.model import STUDENT_SPECS, instantiate_student_model
from zakotfs_distill.training import distill_loss, predict_student_support, train_student


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL_ROOT = REPO_ROOT / "distill_novelty"
PHYSICS_ROOT = REPO_ROOT / "physics_novelty"
PHASE1_SMOKE_CONFIG_PATH = PHYSICS_ROOT / "configs" / "phase1_smoke.yaml"
DISTILL_SMOKE_CONFIG_PATH = DISTILL_ROOT / "configs" / "distill_smoke.yaml"


def _prepare_smoke_setup(tmp_path: Path):
    phase1_config = load_config(PHASE1_SMOKE_CONFIG_PATH)
    phase1_config.raw["paths"]["results_dir"] = str((tmp_path / "physics_results").resolve())
    phase1_config.raw["paths"]["logs_dir"] = str((tmp_path / "physics_logs").resolve())
    phase1_config.raw["paths"]["report_dir"] = str((tmp_path / "physics_report").resolve())
    phase1_config.raw["device"] = "cpu"
    manifest_path = generate_phase1_dataset(phase1_config, force=True)

    distill_config = load_config(DISTILL_SMOKE_CONFIG_PATH)
    distill_config.raw["paths"]["results_dir"] = str((tmp_path / "distill_results").resolve())
    distill_config.raw["paths"]["logs_dir"] = str((tmp_path / "distill_logs").resolve())
    distill_config.raw["paths"]["report_dir"] = str((tmp_path / "distill_report").resolve())
    distill_config.raw["distill_dataset"]["train_manifest_path"] = str(manifest_path)
    distill_config.raw["distill_dataset"]["val_manifest_path"] = str(manifest_path)
    distill_config.raw["smoke"]["distill_dataset"]["train_manifest_path"] = str(manifest_path)
    distill_config.raw["smoke"]["distill_dataset"]["val_manifest_path"] = str(manifest_path)
    distill_config.raw["device"] = "cpu"
    return manifest_path, distill_config


def test_distill_dataset_shapes(tmp_path: Path) -> None:
    manifest_path, _ = _prepare_smoke_setup(tmp_path)
    dataset = DistillDataset(manifest_path)
    sample = dataset[0]

    assert sample["support_input"].ndim == 2
    assert sample["teacher_target"].shape == sample["support_input"].shape
    assert sample["truth_target"].shape == sample["support_input"].shape
    assert sample["support_input"].dtype == torch.complex64


def test_distill_model_param_reduction(tmp_path: Path) -> None:
    _, config = _prepare_smoke_setup(tmp_path)
    model = instantiate_student_model(config)

    assert model.num_parameters == 40049
    assert STUDENT_SPECS["lite_l"].channels == (32, 16, 16)


def test_distill_forward_and_loss_are_finite(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    dataset = DistillDataset(manifest_path)
    model = instantiate_student_model(config)
    batch_input = torch.stack([dataset[0]["support_input"], dataset[1]["support_input"]], dim=0)
    batch_teacher = torch.stack([dataset[0]["teacher_target"], dataset[1]["teacher_target"]], dim=0)
    batch_truth = torch.stack([dataset[0]["truth_target"], dataset[1]["truth_target"]], dim=0)

    prediction = predict_student_support(model, batch_input)
    losses = distill_loss(prediction, batch_teacher, batch_truth, distill_weight=0.8, truth_weight=0.2)

    assert prediction.shape == batch_input.shape
    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_teacher"])
    assert torch.isfinite(losses["loss_truth"])


def test_distill_smoke_training_and_eval_run(tmp_path: Path) -> None:
    manifest_path, config = _prepare_smoke_setup(tmp_path)
    checkpoint_path = train_student(config, train_manifest_path=manifest_path, val_manifest_path=manifest_path)

    assert checkpoint_path.exists()
    history_path = Path(config.root / config.raw["paths"]["logs_dir"] / "train_history_distill_lite_l_smoke.json")
    assert history_path.exists()
    history = load_json(history_path)
    assert len(history["history"]) == 1

    outputs = run_distill_evaluation(config, checkpoint_path=checkpoint_path)
    benchmark_path = benchmark_student_models(config, checkpoint_path=checkpoint_path)

    assert benchmark_path.exists()
    assert set(outputs.keys()) == {"ber_vs_pdr", "ber_vs_snr", "nmse_vs_pdr", "nmse_vs_snr"}
    assert (Path(config.root / config.raw["paths"]["results_dir"]) / "distill_nmse_vs_pdr_lite_l_smoke.json").exists()
