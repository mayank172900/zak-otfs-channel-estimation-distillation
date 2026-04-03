from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zakotfs_novelty.adapter import AdapterFeatureDataset


def test_adapter_dataset_slices_generic_channels(tmp_path: Path) -> None:
    inputs = np.arange(8 * 2 * 2, dtype=np.float32).reshape(1, 8, 2, 2)
    targets = np.ones((1, 2, 2, 2), dtype=np.float32)
    inputs_path = tmp_path / "inputs.npy"
    targets_path = tmp_path / "targets.npy"
    pdr_path = tmp_path / "pdr.npy"
    np.save(inputs_path, inputs)
    np.save(targets_path, targets)
    np.save(pdr_path, np.array([5.0], dtype=np.float32))
    manifest = {
        "size": 1,
        "inputs_path": inputs_path.name,
        "targets_path": targets_path.name,
        "pdr_labels_path": pdr_path.name,
    }
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    generic = AdapterFeatureDataset(manifest_path, adapter_kind="generic")
    fb_lara = AdapterFeatureDataset(manifest_path, adapter_kind="fb_lara")
    x_generic, y_generic = generic[0]
    x_fb_lara, y_fb_lara = fb_lara[0]
    assert x_generic.shape == (6, 2, 2)
    assert x_fb_lara.shape == (8, 2, 2)
    assert np.allclose(x_generic.numpy()[0], inputs[0, 0])
    assert np.allclose(x_generic.numpy()[4], inputs[0, 6])
    assert np.allclose(x_generic.numpy()[5], inputs[0, 7])
    assert np.allclose(y_generic.numpy(), targets[0])
    assert np.allclose(y_fb_lara.numpy(), targets[0])
