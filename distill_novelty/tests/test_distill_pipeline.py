from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from zakotfs.cnn_model import PaperCNN
from zakotfs.params import load_config
from zakotfs.utils import load_json
from zakotfs_distill.benchmark import benchmark_student_models
from zakotfs_distill.dataset import DistillDataset
from zakotfs_distill.evaluation import run_distill_evaluation
from zakotfs_distill.model import STUDENT_SPECS, instantiate_student_model
from zakotfs_distill.training import distill_loss, predict_student_support, train_student


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL_ROOT = REPO_ROOT / "distill_novelty"
DISTILL_SMOKE_CONFIG_PATH = DISTILL_ROOT / "configs" / "distill_smoke.yaml"


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _create_synthetic_manifest(tmp_path: Path, size: int = 4, h: int = 7, w: int = 9) -> Path:
    data_dir = tmp_path / "phase1"
    sample_index = np.arange(size, dtype=np.int64)
    sample_seed = 1000 + sample_index
    pdr_db = np.linspace(0.0, 5.0, size, dtype=np.float32)
    data_snr_db = np.full(size, 15.0, dtype=np.float32)
    E_p = np.full(size, 1.0, dtype=np.float32)
    rho_d = np.full(size, 0.5, dtype=np.float32)
    rho_p = np.full(size, 0.5, dtype=np.float32)
    noise_variance = np.full(size, 0.1, dtype=np.float32)

    base = np.random.default_rng(7).standard_normal((size, h, w), dtype=np.float32)
    _write_array(data_dir / "smoke_h_obs_re.npy", base)
    _write_array(data_dir / "smoke_h_obs_im.npy", base * 0.1)
    _write_array(data_dir / "smoke_h_base_re.npy", base * 0.8)
    _write_array(data_dir / "smoke_h_base_im.npy", base * 0.05)
    _write_array(data_dir / "smoke_h_true_re.npy", base * 0.85)
    _write_array(data_dir / "smoke_h_true_im.npy", base * 0.08)
    _write_array(data_dir / "smoke_pdr_db.npy", pdr_db)
    _write_array(data_dir / "smoke_sample_index.npy", sample_index)
    _write_array(data_dir / "smoke_sample_seed.npy", sample_seed)
    _write_array(data_dir / "smoke_data_snr_db.npy", data_snr_db)
    _write_array(data_dir / "smoke_E_p.npy", E_p)
    _write_array(data_dir / "smoke_rho_d.npy", rho_d)
    _write_array(data_dir / "smoke_rho_p.npy", rho_p)
    _write_array(data_dir / "smoke_noise_variance.npy", noise_variance)

    manifest = {
        "generator": "distill_test_manifest",
        "split": "smoke",
        "size": size,
        "shape": [h, w],
        "include_physics_target": False,
        "h_obs_re_path": "smoke_h_obs_re.npy",
        "h_obs_im_path": "smoke_h_obs_im.npy",
        "h_base_re_path": "smoke_h_base_re.npy",
        "h_base_im_path": "smoke_h_base_im.npy",
        "h_true_re_path": "smoke_h_true_re.npy",
        "h_true_im_path": "smoke_h_true_im.npy",
        "pdr_db_path": "smoke_pdr_db.npy",
        "sample_index_path": "smoke_sample_index.npy",
        "sample_seed_path": "smoke_sample_seed.npy",
        "data_snr_db_path": "smoke_data_snr_db.npy",
        "E_p_path": "smoke_E_p.npy",
        "rho_d_path": "smoke_rho_d.npy",
        "rho_p_path": "smoke_rho_p.npy",
        "noise_variance_path": "smoke_noise_variance.npy",
    }
    manifest_path = data_dir / "smoke_phase1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _prepare_smoke_setup(tmp_path: Path):
    manifest_path = _create_synthetic_manifest(tmp_path)
    teacher_ckpt = tmp_path / "teacher" / "full_cnn_best.pt"
    teacher_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": PaperCNN().state_dict()}, teacher_ckpt)

    distill_config = load_config(DISTILL_SMOKE_CONFIG_PATH)
    distill_config.raw["paths"]["results_dir"] = str((tmp_path / "distill_results").resolve())
    distill_config.raw["paths"]["logs_dir"] = str((tmp_path / "distill_logs").resolve())
    distill_config.raw["paths"]["report_dir"] = str((tmp_path / "distill_report").resolve())
    distill_config.raw["teacher_checkpoint_path"] = str(teacher_ckpt)
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
