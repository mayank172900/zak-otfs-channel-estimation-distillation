from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapter import AdapterFeatureDataset
from .cnn_model import GenericResidualAdapter, LatticeAliasAdapter, PaperCNN, ResidualAdapter
from .dataset import simulate_frame
from .params import SystemConfig, load_config
from .utils import load_json, logs_dir, pin_memory_enabled, resolve_torch_device, save_json, set_global_seed


class ComplexSupportDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        default_cfg = Path("novelty_paper/configs/train.yaml")
        if not default_cfg.exists():
            default_cfg = Path("configs/train.yaml")
        self.config = load_config(default_cfg) if self.path.suffix == ".json" else None
        self.cache_in_ram = bool(self.config.raw.get("dataset", {}).get("cache_in_ram", False)) if self.config is not None else False
        if self.path.suffix == ".npz":
            loaded = np.load(path)
            self.kind = "npz"
            self.inputs = loaded["inputs"].astype(np.complex64)
            self.targets = loaded["targets"].astype(np.complex64)
            self.meta = {}
        elif self.path.suffix == ".json":
            self.meta = load_json(self.path)
            generator = str(self.meta.get("generator", "on_the_fly"))
            if generator == "memmap_fp16":
                self.kind = "memmap_fp16"
                load_mode = None if self.cache_in_ram else "r"
                self.inputs_re = np.load(self._resolve_meta_path("inputs_re_path"), mmap_mode=load_mode)
                self.inputs_im = np.load(self._resolve_meta_path("inputs_im_path"), mmap_mode=load_mode)
                self.targets_re = np.load(self._resolve_meta_path("targets_re_path"), mmap_mode=load_mode)
                self.targets_im = np.load(self._resolve_meta_path("targets_im_path"), mmap_mode=load_mode)
                if self.cache_in_ram:
                    print(f"[train] caching dataset into RAM from {self.path}", flush=True)
                    self.inputs_re = np.asarray(self.inputs_re)
                    self.inputs_im = np.asarray(self.inputs_im)
                    self.targets_re = np.asarray(self.targets_re)
                    self.targets_im = np.asarray(self.targets_im)
                self.inputs = None
                self.targets = None
            else:
                self.kind = "generated"
                self.inputs = None
                self.targets = None
        else:
            raise ValueError(f"Unsupported dataset path: {path}")

    def _resolve_meta_path(self, key: str) -> str:
        target = Path(str(self.meta[key]))
        if target.is_absolute():
            return str(target)
        return str((self.path.parent / target).resolve())

    def __len__(self) -> int:
        return int(self.inputs.shape[0] if self.kind == "npz" else self.meta["size"])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "npz":
            return torch.from_numpy(self.inputs[index]).to(torch.complex64), torch.from_numpy(self.targets[index]).to(torch.complex64)
        if self.kind == "memmap_fp16":
            x_re = torch.from_numpy(np.asarray(self.inputs_re[index], dtype=np.float32))
            x_im = torch.from_numpy(np.asarray(self.inputs_im[index], dtype=np.float32))
            y_re = torch.from_numpy(np.asarray(self.targets_re[index], dtype=np.float32))
            y_im = torch.from_numpy(np.asarray(self.targets_im[index], dtype=np.float32))
            return torch.complex(x_re, x_im), torch.complex(y_re, y_im)
        pdrs = list(map(float, self.meta["pdr_db"]))
        per_pdr = int(self.meta["per_pdr"])
        pdr_idx = min(index // per_pdr, len(pdrs) - 1)
        pdr_db = pdrs[pdr_idx]
        sample_seed = int(self.meta["seed"]) + int(index)
        frame = simulate_frame(self.config, str(self.meta.get("modulation", "bpsk")), float(self.meta["snr_db"]), pdr_db, np.random.default_rng(sample_seed))
        return torch.from_numpy(frame.support_input).to(torch.complex64), torch.from_numpy(frame.support_true).to(torch.complex64)


def _loss_fn(model: PaperCNN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_re = x.real.unsqueeze(1).to(dtype=torch.float32)
    x_im = x.imag.unsqueeze(1).to(dtype=torch.float32)
    target_re = y.real.unsqueeze(1).to(dtype=torch.float32)
    target_im = y.imag.unsqueeze(1).to(dtype=torch.float32)
    pred_re = model(x_re)
    pred_im = model(x_im)
    eps = 1.0e-12
    num = ((pred_re - target_re) ** 2).sum(dim=(1, 2, 3)) + ((pred_im - target_im) ** 2).sum(dim=(1, 2, 3))
    den = (target_re**2).sum(dim=(1, 2, 3)) + (target_im**2).sum(dim=(1, 2, 3))
    return (num / den.clamp_min(eps)).mean()


def train_cnn(config: SystemConfig, train_path: Path, val_path: Path, mode: str = "full") -> Path:
    set_global_seed(config.seed)
    if mode == "smoke":
        training_cfg = {**config.raw["training"], **config.raw.get("smoke", {}).get("training", {})}
    else:
        training_cfg = config.raw["training"]
    device = resolve_torch_device(config)
    model = PaperCNN().to(device=device, dtype=torch.float32)
    train_ds = ComplexSupportDataset(train_path)
    val_ds = ComplexSupportDataset(val_path)
    num_workers = int(config.raw.get("dataset", {}).get("num_workers", 0))
    print(
        f"[train] mode={mode} device={device} batch_size={training_cfg['batch_size']} epochs={training_cfg['epochs']} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} num_workers={num_workers}",
        flush=True,
    )
    print(f"[train] train_source={train_path} val_source={val_path}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_enabled(device),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_enabled(device),
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training_cfg["scheduler"]["factor"]),
        patience=int(training_cfg["scheduler"]["patience"]),
        min_lr=float(training_cfg["scheduler"]["min_lr"]),
    )
    history: list[dict[str, float]] = []
    best_val = float("inf")
    ckpt_dir = logs_dir(config) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{mode}_{training_cfg['checkpoint_name']}"
    latest_ckpt_path = ckpt_dir / f"{mode}_cnn_latest.pt"
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
            f"[train] auto-resume from {latest_ckpt_path} starting at epoch {start_epoch + 1}/{int(training_cfg['epochs'])}",
            flush=True,
        )
    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        print(f"[train] epoch {epoch + 1}/{int(training_cfg['epochs'])} started", flush=True)
        model.train()
        train_losses = []
        num_train_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device=device, dtype=torch.complex64)
            y = y.to(device=device, dtype=torch.complex64)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_fn(model, x, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_train_batches:
                print(
                    f"[train] epoch {epoch + 1} batch {batch_idx}/{num_train_batches} loss={train_losses[-1]:.6f}",
                    flush=True,
                )
        model.eval()
        val_losses = []
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device=device, dtype=torch.complex64)
                y = y.to(device=device, dtype=torch.complex64)
                val_losses.append(float(_loss_fn(model, x, y).detach().cpu()))
                if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_val_batches:
                    print(
                        f"[train] epoch {epoch + 1} val_batch {batch_idx}/{num_val_batches} loss={val_losses[-1]:.6f}",
                        flush=True,
                    )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - start),
        }
        history.append(record)
        print(
            f"[train] epoch {epoch + 1} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={record['lr']:.6e} elapsed_s={record['elapsed_s']:.1f}",
            flush=True,
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        latest_payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val,
            "elapsed_s": record["elapsed_s"],
        }
        torch.save(latest_payload, latest_ckpt_path)
        print(f"[train] latest checkpoint saved to {latest_ckpt_path}", flush=True)
        if bool(training_cfg.get("save_epoch_checkpoints", True)):
            epoch_ckpt_path = ckpt_dir / f"{mode}_epoch_{epoch + 1:03d}.pt"
            torch.save(latest_payload, epoch_ckpt_path)
            print(f"[train] epoch checkpoint saved to {epoch_ckpt_path}", flush=True)
        if improved:
            torch.save(latest_payload, ckpt_path)
            print(f"[train] new best checkpoint saved to {ckpt_path} with val_loss={best_val:.6f}", flush=True)
    save_json(logs_dir(config) / f"train_history_{mode}.json", {"history": history, "best_val_loss": best_val})
    print(f"[train] finished best_val_loss={best_val:.6f} history={logs_dir(config) / f'train_history_{mode}.json'}", flush=True)
    return ckpt_path


def load_cnn_checkpoint(config: SystemConfig, checkpoint_path: Path | None = None) -> PaperCNN:
    if checkpoint_path is None:
        checkpoint_path = logs_dir(config) / "checkpoints" / "full_cnn_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = logs_dir(config) / "checkpoints" / "smoke_cnn_best.pt"
    device = resolve_torch_device(config)
    model = PaperCNN().to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[train] loaded checkpoint {checkpoint_path} on device={device}", flush=True)
    return model


def instantiate_adapter(adapter_kind: str) -> ResidualAdapter:
    kind = str(adapter_kind).lower()
    if kind == "generic":
        return GenericResidualAdapter()
    if kind == "fb_lara":
        return LatticeAliasAdapter()
    raise ValueError(f"Unknown adapter kind '{adapter_kind}'")


def _adapter_loss_fn(adapter: ResidualAdapter, x: torch.Tensor, residual_target: torch.Tensor, residual_weight: float) -> torch.Tensor:
    pred = adapter(x)
    h_base = x[:, 2:4]
    h_true = h_base + residual_target
    eps = 1.0e-12
    err_num = ((pred - residual_target) ** 2).sum(dim=(1, 2, 3))
    delta_num = (pred**2).sum(dim=(1, 2, 3))
    denom = (h_true**2).sum(dim=(1, 2, 3)).clamp_min(eps)
    return ((err_num / denom) + residual_weight * (delta_num / denom)).mean()


def train_adapter(
    config: SystemConfig,
    train_path: Path,
    val_path: Path,
    adapter_kind: str,
    mode: str = "full",
) -> Path:
    set_global_seed(config.seed)
    if mode == "smoke":
        training_cfg = {**config.raw["adapter_training"], **config.raw.get("smoke", {}).get("adapter_training", {})}
    else:
        training_cfg = config.raw["adapter_training"]
    device = resolve_torch_device(config)
    adapter = instantiate_adapter(adapter_kind).to(device=device, dtype=torch.float32)
    train_ds = AdapterFeatureDataset(train_path, adapter_kind=adapter_kind)
    val_ds = AdapterFeatureDataset(val_path, adapter_kind=adapter_kind)
    num_workers = int(config.raw.get("adapter_dataset", {}).get("num_workers", 0))
    print(
        f"[adapter-train] mode={mode} kind={adapter_kind} device={device} batch_size={training_cfg['batch_size']} "
        f"epochs={training_cfg['epochs']} train_samples={len(train_ds)} val_samples={len(val_ds)} num_workers={num_workers}",
        flush=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_enabled(device),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_enabled(device),
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(training_cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training_cfg["scheduler"]["factor"]),
        patience=int(training_cfg["scheduler"]["patience"]),
        min_lr=float(training_cfg["scheduler"]["min_lr"]),
    )
    residual_weight = float(training_cfg.get("residual_penalty", 1.0e-3))
    history: list[dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = int(training_cfg.get("early_stop_patience", training_cfg["epochs"]))
    ckpt_dir = logs_dir(config) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{mode}_{adapter_kind}_best.pt"
    latest_ckpt_path = ckpt_dir / f"{mode}_{adapter_kind}_latest.pt"
    history_path = logs_dir(config) / f"train_history_{adapter_kind}_{mode}.json"
    start = time.time()
    start_epoch = 0
    if bool(training_cfg.get("auto_resume", True)) and latest_ckpt_path.exists():
        payload = torch.load(latest_ckpt_path, map_location=device)
        adapter.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = list(payload.get("history", []))
        best_val = float(payload.get("best_val_loss", float("inf")))
        start_epoch = int(payload.get("epoch", 0))
        start = time.time() - float(payload.get("elapsed_s", 0.0))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        print(
            f"[adapter-train] auto-resume from {latest_ckpt_path} starting at epoch {start_epoch + 1}/{int(training_cfg['epochs'])}",
            flush=True,
        )
    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        print(f"[adapter-train] epoch {epoch + 1}/{int(training_cfg['epochs'])} started", flush=True)
        adapter.train()
        train_losses: list[float] = []
        num_train_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = _adapter_loss_fn(adapter, x, y, residual_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_train_batches:
                print(
                    f"[adapter-train] epoch {epoch + 1} batch {batch_idx}/{num_train_batches} loss={train_losses[-1]:.6f}",
                    flush=True,
                )
        adapter.eval()
        val_losses: list[float] = []
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                val_losses.append(float(_adapter_loss_fn(adapter, x, y, residual_weight).detach().cpu()))
                if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_val_batches:
                    print(
                        f"[adapter-train] epoch {epoch + 1} val_batch {batch_idx}/{num_val_batches} loss={val_losses[-1]:.6f}",
                        flush=True,
                    )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - start),
        }
        history.append(record)
        payload = {
            "epoch": epoch + 1,
            "adapter_kind": str(adapter_kind),
            "state_dict": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val,
            "elapsed_s": record["elapsed_s"],
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(payload, latest_ckpt_path)
        print(
            f"[adapter-train] epoch {epoch + 1} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={record['lr']:.6e} elapsed_s={record['elapsed_s']:.1f}",
            flush=True,
        )
        print(f"[adapter-train] latest checkpoint saved to {latest_ckpt_path}", flush=True)
        if bool(training_cfg.get("save_epoch_checkpoints", True)):
            epoch_ckpt_path = ckpt_dir / f"{mode}_{adapter_kind}_epoch_{epoch + 1:03d}.pt"
            torch.save(payload, epoch_ckpt_path)
            print(f"[adapter-train] epoch checkpoint saved to {epoch_ckpt_path}", flush=True)
        if improved:
            torch.save(payload, ckpt_path)
            print(f"[adapter-train] new best checkpoint saved to {ckpt_path} with val_loss={best_val:.6f}", flush=True)
        if epochs_without_improvement >= early_stop_patience:
            print(
                f"[adapter-train] early stop at epoch {epoch + 1} after {epochs_without_improvement} non-improving epochs",
                flush=True,
            )
            break
    save_json(history_path, {"history": history, "best_val_loss": best_val, "adapter_kind": str(adapter_kind)})
    print(f"[adapter-train] finished best_val_loss={best_val:.6f} history={history_path}", flush=True)
    return ckpt_path


def load_adapter_checkpoint(config: SystemConfig, checkpoint_path: Path | None = None, adapter_kind: str | None = None) -> ResidualAdapter:
    if checkpoint_path is None:
        if adapter_kind is None:
            adapter_kind = str(config.raw.get("adapter", {}).get("default_kind", "fb_lara"))
        checkpoint_path = logs_dir(config) / "checkpoints" / f"full_{adapter_kind}_best.pt"
    device = resolve_torch_device(config)
    payload = torch.load(checkpoint_path, map_location=device)
    resolved_kind = str(adapter_kind or payload.get("adapter_kind", "fb_lara"))
    model = instantiate_adapter(resolved_kind).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[adapter-train] loaded {resolved_kind} checkpoint {checkpoint_path} on device={device}", flush=True)
    return model
