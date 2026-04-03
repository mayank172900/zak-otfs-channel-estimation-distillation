from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from zakotfs.params import SystemConfig
from zakotfs.utils import ensure_dir, save_json, set_global_seed

from .dataset import Phase2TrainingDataset
from .model import instantiate_phase2_model, predict_phase2
from .physics_targets import forward_physics_target_torch


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


def _resolve_torch_device(config: SystemConfig) -> torch.device:
    configured_device = str(config.raw.get("device", "auto")).lower()
    if configured_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.device=cuda but torch.cuda.is_available() is False")
        device_name = "cuda"
    elif configured_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = configured_device
    if device_name == "cpu" and configured_device == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    return torch.device(device_name)


def _pin_memory_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def _phase2_sections(config: SystemConfig) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_cfg = dict(config.raw["phase2_dataset"])
    training_cfg = dict(config.raw["phase2_training"])
    mode = str(training_cfg.get("mode", "full"))
    if mode == "smoke":
        smoke_cfg = config.raw.get("smoke", {})
        dataset_cfg = _deep_update(dataset_cfg, smoke_cfg.get("phase2_dataset", {}))
        training_cfg = _deep_update(training_cfg, smoke_cfg.get("phase2_training", {}))
    return mode, dataset_cfg, training_cfg


def _resolve_path(config: SystemConfig, value: str | Path) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return (config.root / target).resolve()


def phase2_loss(
    predictions: dict[str, torch.Tensor],
    h_true: torch.Tensor,
    h_phys_true: torch.Tensor | None = None,
    config: SystemConfig | None = None,
    lambda_unc: float = 0.1,
    lambda_delta: float = 0.0,
    beta: float = 0.01,
    enable_physics_loss: bool = False,
    lambda_phys: float = 0.0,
) -> dict[str, torch.Tensor]:
    h_hat = predictions["h_hat"]
    h_base = predictions["h_base"]
    delta = predictions["delta"]
    uncertainty = predictions["uncertainty"]
    eps = 1.0e-12
    sq_error = (h_hat - h_true) ** 2
    err_energy = sq_error.sum(dim=1, keepdim=True)
    target_energy = (h_true**2).sum(dim=(1, 2, 3))
    rec = ((h_hat - h_true) ** 2).sum(dim=(1, 2, 3)) / target_energy.clamp_min(eps)
    unc = torch.mean(torch.exp(-uncertainty) * err_energy + float(beta) * uncertainty)
    base_energy = (h_base**2).sum(dim=(1, 2, 3)).clamp_min(eps)
    delta_penalty = (((delta) ** 2).sum(dim=(1, 2, 3)) / base_energy).mean()
    h_phys_hat: torch.Tensor | None = None
    if enable_physics_loss:
        if h_phys_true is None:
            raise ValueError("Physics loss requires h_phys_true")
        if config is None:
            raise ValueError("Physics loss requires config")
        h_phys_hat = forward_physics_target_torch(h_hat, config)
        phys_den = (h_phys_true**2).sum(dim=(1, 2, 3)).clamp_min(eps)
        phys_num = ((h_phys_hat - h_phys_true) ** 2).sum(dim=(1, 2, 3))
        phys = (phys_num / phys_den).mean()
    else:
        phys = torch.zeros((), device=h_true.device, dtype=h_true.dtype)
    total = rec.mean() + float(lambda_unc) * unc + float(lambda_delta) * delta_penalty + float(lambda_phys) * phys
    return {
        "loss_total": total,
        "loss_rec": rec.mean(),
        "loss_unc": unc,
        "loss_delta": delta_penalty,
        "loss_phys": phys,
    }


def _mean_metric(records: list[dict[str, float]], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else float("nan")


def train_phase2(
    config: SystemConfig,
    train_manifest_path: Path | None = None,
    val_manifest_path: Path | None = None,
) -> Path:
    set_global_seed(config.seed)
    mode, dataset_cfg, training_cfg = _phase2_sections(config)
    device = _resolve_torch_device(config)
    model = instantiate_phase2_model(config).to(device=device, dtype=torch.float32)

    train_path = _resolve_path(config, train_manifest_path or dataset_cfg["train_manifest_path"])
    val_path = _resolve_path(config, val_manifest_path or dataset_cfg["val_manifest_path"])
    train_ds = Phase2TrainingDataset(train_path)
    val_ds = Phase2TrainingDataset(val_path)
    num_workers = int(dataset_cfg.get("num_workers", 0))
    print(
        f"[phase2-train] mode={mode} device={device} batch_size={training_cfg['batch_size']} "
        f"epochs={training_cfg['epochs']} train_samples={len(train_ds)} val_samples={len(val_ds)} num_workers={num_workers}",
        flush=True,
    )
    print(f"[phase2-train] train_source={train_path} val_source={val_path}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(device),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(config.seed + 101),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(device),
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
    ckpt_path = ckpt_dir / f"{mode}_{training_cfg['checkpoint_name']}"
    latest_ckpt_path = ckpt_dir / f"{mode}_phase2_latest.pt"
    history_path = logs_dir / f"train_history_phase2_{mode}.json"
    history: list[dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = int(training_cfg.get("early_stop_patience", training_cfg["epochs"]))
    start = time.time()
    start_epoch = 0

    if bool(training_cfg.get("auto_resume", True)) and latest_ckpt_path.exists():
        payload = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = list(payload.get("history", []))
        best_val = float(payload.get("best_val_loss", float("inf")))
        start_epoch = int(payload.get("epoch", 0))
        start = time.time() - float(payload.get("elapsed_s", 0.0))
        print(
            f"[phase2-train] auto-resume from {latest_ckpt_path} starting at epoch {start_epoch + 1}/{int(training_cfg['epochs'])}",
            flush=True,
        )

    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        print(f"[phase2-train] epoch {epoch + 1}/{int(training_cfg['epochs'])} started", flush=True)
        model.train()
        train_records: list[dict[str, float]] = []
        num_train_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            inputs = batch["inputs"].to(device=device, dtype=torch.float32)
            h_true = batch["h_true"].to(device=device, dtype=torch.float32)
            h_phys_true = batch["h_phys_true"].to(device=device, dtype=torch.float32) if "h_phys_true" in batch else None
            optimizer.zero_grad(set_to_none=True)
            predictions = predict_phase2(model, inputs)
            losses = phase2_loss(
                predictions,
                h_true,
                h_phys_true=h_phys_true,
                config=config,
                lambda_unc=float(training_cfg.get("lambda_unc", 0.1)),
                lambda_delta=float(training_cfg.get("lambda_delta", 0.0)),
                beta=float(training_cfg.get("beta", 0.01)),
                enable_physics_loss=bool(training_cfg.get("enable_physics_loss", False)),
                lambda_phys=float(training_cfg.get("lambda_phys", 0.0)),
            )
            total = losses["loss_total"]
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch + 1} batch={batch_idx}")
            total.backward()
            optimizer.step()
            train_records.append({key: float(value.detach().cpu()) for key, value in losses.items()})
            if batch_idx == 1 or batch_idx == num_train_batches or batch_idx % 100 == 0:
                latest = train_records[-1]
                print(
                    f"[phase2-train] epoch {epoch + 1} batch {batch_idx}/{num_train_batches} "
                    f"total={latest['loss_total']:.6f} rec={latest['loss_rec']:.6f} "
                    f"unc={latest['loss_unc']:.6f} delta={latest['loss_delta']:.6f} phys={latest['loss_phys']:.6f}",
                    flush=True,
                )

        model.eval()
        val_records: list[dict[str, float]] = []
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                inputs = batch["inputs"].to(device=device, dtype=torch.float32)
                h_true = batch["h_true"].to(device=device, dtype=torch.float32)
                h_phys_true = batch["h_phys_true"].to(device=device, dtype=torch.float32) if "h_phys_true" in batch else None
                predictions = predict_phase2(model, inputs)
                losses = phase2_loss(
                    predictions,
                    h_true,
                    h_phys_true=h_phys_true,
                    config=config,
                    lambda_unc=float(training_cfg.get("lambda_unc", 0.1)),
                    lambda_delta=float(training_cfg.get("lambda_delta", 0.0)),
                    beta=float(training_cfg.get("beta", 0.01)),
                    enable_physics_loss=bool(training_cfg.get("enable_physics_loss", False)),
                    lambda_phys=float(training_cfg.get("lambda_phys", 0.0)),
                )
                val_records.append({key: float(value.detach().cpu()) for key, value in losses.items()})
                if batch_idx == 1 or batch_idx == num_val_batches or batch_idx % 100 == 0:
                    latest = val_records[-1]
                    print(
                        f"[phase2-train] epoch {epoch + 1} val_batch {batch_idx}/{num_val_batches} "
                        f"total={latest['loss_total']:.6f} rec={latest['loss_rec']:.6f} "
                        f"unc={latest['loss_unc']:.6f} delta={latest['loss_delta']:.6f} phys={latest['loss_phys']:.6f}",
                        flush=True,
                    )

        train_total = _mean_metric(train_records, "loss_total")
        val_total = _mean_metric(val_records, "loss_total")
        scheduler.step(val_total)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_total,
            "val_loss": val_total,
            "train_rec": _mean_metric(train_records, "loss_rec"),
            "train_unc": _mean_metric(train_records, "loss_unc"),
            "train_delta": _mean_metric(train_records, "loss_delta"),
            "train_phys": _mean_metric(train_records, "loss_phys"),
            "val_rec": _mean_metric(val_records, "loss_rec"),
            "val_unc": _mean_metric(val_records, "loss_unc"),
            "val_delta": _mean_metric(val_records, "loss_delta"),
            "val_phys": _mean_metric(val_records, "loss_phys"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - start),
        }
        history.append(record)
        print(
            f"[phase2-train] epoch {epoch + 1} done train_loss={record['train_loss']:.6f} "
            f"val_loss={record['val_loss']:.6f} lr={record['lr']:.6e} elapsed_s={record['elapsed_s']:.1f}",
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
            "model_config": dict(config.raw["phase2_model"]),
        }
        torch.save(payload, latest_ckpt_path)
        print(f"[phase2-train] latest checkpoint saved to {latest_ckpt_path}", flush=True)
        if bool(training_cfg.get("save_epoch_checkpoints", True)):
            epoch_ckpt_path = ckpt_dir / f"{mode}_phase2_epoch_{epoch + 1:03d}.pt"
            torch.save(payload, epoch_ckpt_path)
            print(f"[phase2-train] epoch checkpoint saved to {epoch_ckpt_path}", flush=True)
        if improved:
            torch.save(payload, ckpt_path)
            print(f"[phase2-train] new best checkpoint saved to {ckpt_path} with val_loss={best_val:.6f}", flush=True)
        save_json(history_path, {"history": history, "best_val_loss": best_val})
        if epochs_without_improvement >= early_stop_patience:
            print(
                f"[phase2-train] early stop after epoch {epoch + 1} with patience={early_stop_patience}",
                flush=True,
            )
            break

    print(f"[phase2-train] finished best_val_loss={best_val:.6f} history={history_path}", flush=True)
    return ckpt_path


def load_phase2_checkpoint(config: SystemConfig, checkpoint_path: Path | None = None) -> torch.nn.Module:
    mode, _, training_cfg = _phase2_sections(config)
    if checkpoint_path is None:
        checkpoint_path = _logs_dir(config) / "checkpoints" / f"{mode}_{training_cfg['checkpoint_name']}"
    device = _resolve_torch_device(config)
    model = instantiate_phase2_model(config).to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[phase2-train] loaded checkpoint {checkpoint_path} on device={device}", flush=True)
    return model
