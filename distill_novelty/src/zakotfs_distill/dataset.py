from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .phase1_data import load_phase1_manifest, open_phase1_arrays


def _stack_complex(arrays: dict[str, np.ndarray], stem: str, index: int) -> np.ndarray:
    re = np.asarray(arrays[f"{stem}_re"][index], dtype=np.float32)
    im = np.asarray(arrays[f"{stem}_im"][index], dtype=np.float32)
    return (re + 1j * im).astype(np.complex64)


class DistillDataset(Dataset):
    def __init__(self, manifest_path: str | Path, mmap_mode: str = "r") -> None:
        self.path = Path(manifest_path)
        self.meta = load_phase1_manifest(self.path)
        self.arrays = open_phase1_arrays(self.path, mmap_mode=mmap_mode)

    def __len__(self) -> int:
        return int(self.meta["size"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h_obs = _stack_complex(self.arrays, "h_obs", index)
        h_base = _stack_complex(self.arrays, "h_base", index)
        h_true = _stack_complex(self.arrays, "h_true", index)
        return {
            "support_input": torch.from_numpy(h_obs),
            "teacher_target": torch.from_numpy(h_base),
            "truth_target": torch.from_numpy(h_true),
            "pdr_db": torch.tensor(float(self.arrays["pdr_db"][index]), dtype=torch.float32),
            "sample_index": torch.tensor(int(self.arrays["sample_index"][index]), dtype=torch.int64),
            "sample_seed": torch.tensor(int(self.arrays["sample_seed"][index]), dtype=torch.int64),
        }
