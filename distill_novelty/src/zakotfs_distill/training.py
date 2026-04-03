from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from zakotfs.params import SystemConfig
from zakotfs.utils import ensure_dir, save_json, set_global_seed

from .dataset import DistillDataset
from .model import instantiate_student_model


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _logs_dir(config: SystemConfig) -> Path:
    return ensure_dir(config.root / config.raw["paths"]["logs_dir"])


def _resolve_device(config: SystemConfig) -> torch.device:
    configured = str(config.raw.get("device", "auto")).lower()
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.device=cuda but torch.cuda.is_available() is False")
        name = "cuda"
    elif configured == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        name = configured
    if name == "cpu" and configured == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        name = "mps"
    return torch.device(name)


def _sections(config: SystemConfig) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_cfg = dict(config.raw["distill_dataset"])
    training_cfg = dict(config.raw["distill_training"])
    mode = str(training_cfg.get("mode", "full"))
    if mode == "smoke":
        smoke = config.raw.get("smoke", {})
        dataset_cfg = _deep_update(dataset_cfg, smoke.get("distill_dataset", {}))
        training_cfg = _deep_update(training_cfg, smoke.get("distill_training", {}))
    return mode, dataset_cfg, training_cfg


def _resolve_path(config: SystemConfig, value: str | Path) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (config.root / target).resolve()


def _nmse_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    num = ((pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2).sum(dim=(1, 2))
    den = (target.real**2 + target.imag**2).sum(dim=(1, 2)).clamp_min(eps)
    return num / den


def predict_student_support(model: torch.nn.Module, support_input: torch.Tensor) -> torch.Tensor:
    single = support_input.ndim == 2
    if single:
        support_input = support_input.unsqueeze(0)
    x_re = support_input.real.unsqueeze(1).to(dtype=torch.float32)
    x_im = support_input.imag.unsqueeze(1).to(dtype=torch.float32)
    y_re = model(x_re)[:, 0]
    y_im = model(x_im)[:, 0]
    output = torch.complex(y_re, y_im)
    return output[0] if single else output


def distill_loss(
    prediction: torch.Tensor,
    teacher_target: torch.Tensor,
    truth_target: torch.Tensor,
    distill_weight: float = 0.8,
    truth_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    loss_teacher = _nmse_complex(prediction, teacher_target).mean()
    loss_truth = _nmse_complex(prediction, truth_target).mean()
    total = float(distill_weight) * loss_teacher + float(truth_weight) * loss_truth
    return {
        "loss_total": total,
        "loss_teacher": loss_teacher,
        "loss_truth": loss_truth,
    }


def _mean_metric(records: list[dict[str, float]], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else float("nan")


def train_student(
    config: SystemConfig,
    train_manifest_path: Path | None = None,
    val_manifest_path: Path | None = None,
) -> Path:
    set_global_seed(config.seed)
    mode, dataset_cfg, training_cfg = _sections(config)
    device = _resolve_device(config)
    model = instantiate_student_model(config).to(device=device, dtype=torch.float32)
    train_path = _resolve_path(config, train_manifest_path or dataset_cfg["train_manifest_path"])
    val_path = _resolve_path(config, val_manifest_path or dataset_cfg["val_manifest_path"])
    mmap_mode = str(dataset_cfg.get("mmap_mode", "r"))
    train_ds = DistillDataset(train_path, mmap_mode=mmap_mode)
    val_ds = DistillDataset(val_path, mmap_mode=mmap_mode)
    num_workers = int(dataset_cfg.get("num_workers", 0))
    print(
        f"[distill-train] mode={mode} device={device} batch_size={training_cfg['batch_size']} "
        f"epochs={training_cfg['epochs']} train_samples={len(train_ds)} val_samples={len(val_ds)}",
        flush=True,
    )
    print(f"[distill-train] train_source={train_path} val_source={val_path}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(config.seed + 212),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training_cfg["scheduler"]["factor"]),
        patience=int(training_cfg["scheduler"]["patience"]),
        min_lr=float(training_cfg["scheduler"]["min_lr"]),
    )
    logs_dir = _logs_dir(config)
    ckpt_dir = ensure_dir(logs_dir / "checkpoints")
    variant = str(config.raw["distill_model"].get("variant", "lite_l")).lower()
    ckpt_path = ckpt_dir / f"{mode}_{variant}_{training_cfg['checkpoint_name']}"
    latest_path = ckpt_dir / f"{mode}_{variant}_distill_latest.pt"
    history_path = logs_dir / f"train_history_distill_{variant}_{mode}.json"
    history: list[dict[str, float]] = []
    best_val = float("inf")
    start = time.time()
    start_epoch = 0
    epochs_without_improvement = 0
    early_stop_patience = int(training_cfg.get("early_stop_patience", training_cfg["epochs"]))

    if bool(training_cfg.get("auto_resume", True)) and latest_path.exists():
        payload = torch.load(latest_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = list(payload.get("history", []))
        best_val = float(payload.get("best_val_loss", float("inf")))
        start_epoch = int(payload.get("epoch", 0))
        start = time.time() - float(payload.get("elapsed_s", 0.0))
        print(f"[distill-train] auto-resume from {latest_path} epoch={start_epoch + 1}", flush=True)

    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        print(f"[distill-train] epoch {epoch + 1}/{int(training_cfg['epochs'])} started", flush=True)
        model.train()
        train_records: list[dict[str, float]] = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            support_input = batch["support_input"].to(device=device, dtype=torch.complex64)
            teacher_target = batch["teacher_target"].to(device=device, dtype=torch.complex64)
            truth_target = batch["truth_target"].to(device=device, dtype=torch.complex64)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_student_support(model, support_input)
            losses = distill_loss(
                prediction,
                teacher_target,
                truth_target,
                distill_weight=float(training_cfg.get("distill_weight", 0.8)),
                truth_weight=float(training_cfg.get("truth_weight", 0.2)),
            )
            total = losses["loss_total"]
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch + 1} batch={batch_idx}")
            total.backward()
            optimizer.step()
            train_records.append({key: float(value.detach().cpu()) for key, value in losses.items()})

        model.eval()
        val_records: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                support_input = batch["support_input"].to(device=device, dtype=torch.complex64)
                teacher_target = batch["teacher_target"].to(device=device, dtype=torch.complex64)
                truth_target = batch["truth_target"].to(device=device, dtype=torch.complex64)
                prediction = predict_student_support(model, support_input)
                losses = distill_loss(
                    prediction,
                    teacher_target,
                    truth_target,
                    distill_weight=float(training_cfg.get("distill_weight", 0.8)),
                    truth_weight=float(training_cfg.get("truth_weight", 0.2)),
                )
                val_records.append({key: float(value.detach().cpu()) for key, value in losses.items()})

        train_total = _mean_metric(train_records, "loss_total")
        val_total = _mean_metric(val_records, "loss_total")
        scheduler.step(val_total)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_total,
            "val_loss": val_total,
            "train_teacher": _mean_metric(train_records, "loss_teacher"),
            "train_truth": _mean_metric(train_records, "loss_truth"),
            "val_teacher": _mean_metric(val_records, "loss_teacher"),
            "val_truth": _mean_metric(val_records, "loss_truth"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - start),
        }
        history.append(record)
        print(
            f"[distill-train] epoch {epoch + 1} done train_loss={record['train_loss']:.6f} "
            f"val_loss={record['val_loss']:.6f} teacher={record['val_teacher']:.6f} truth={record['val_truth']:.6f}",
            flush=True,
        )
        improved = record["val_loss"] < best_val
        if improved:
            best_val = record["val_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val,
            "elapsed_s": record["elapsed_s"],
            "model_variant": str(config.raw["distill_model"].get("variant", "lite_l")),
            "num_parameters": int(model.num_parameters),
        }
        torch.save(payload, latest_path)
        if bool(training_cfg.get("save_epoch_checkpoints", True)):
            torch.save(payload, ckpt_dir / f"{mode}_distill_epoch_{epoch + 1:03d}.pt")
        if improved:
            torch.save(payload, ckpt_path)
        save_json(history_path, {"history": history, "best_val_loss": best_val})
        if epochs_without_improvement >= early_stop_patience:
            print(f"[distill-train] early stop after epoch {epoch + 1}", flush=True)
            break

    print(f"[distill-train] finished best_val_loss={best_val:.6f} history={history_path}", flush=True)
    return ckpt_path


def load_student_checkpoint(config: SystemConfig, checkpoint_path: Path | None = None) -> torch.nn.Module:
    mode, _, training_cfg = _sections(config)
    if checkpoint_path is None:
        variant = str(config.raw["distill_model"].get("variant", "lite_l")).lower()
        checkpoint_path = _logs_dir(config) / "checkpoints" / f"{mode}_{variant}_{training_cfg['checkpoint_name']}"
    device = _resolve_device(config)
    model = instantiate_student_model(config).to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[distill-train] loaded checkpoint {checkpoint_path} on device={device}", flush=True)
    return model
