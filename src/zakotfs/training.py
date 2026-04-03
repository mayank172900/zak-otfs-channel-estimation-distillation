from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cnn_model import PaperCNN
from .dataset import simulate_frame
from .params import SystemConfig, load_config
from .utils import load_json, logs_dir, save_json, set_global_seed


class ComplexSupportDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.config = load_config(Path("configs/train.yaml")) if self.path.suffix == ".json" else None
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
    device = torch.device(device_name)
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
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
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
    device = torch.device(device_name)
    model = PaperCNN().to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[train] loaded checkpoint {checkpoint_path} on device={device}", flush=True)
    return model
