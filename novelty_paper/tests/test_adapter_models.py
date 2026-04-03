from __future__ import annotations

import torch

from zakotfs_novelty.cnn_model import GenericResidualAdapter, LatticeAliasAdapter


def test_fb_lara_zero_init_outputs_zero_residual() -> None:
    model = LatticeAliasAdapter()
    x = torch.randn(3, 8, 27, 43, dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 2, 27, 43)
    assert torch.count_nonzero(y).item() == 0
    assert model.num_parameters == 6594


def test_generic_adapter_zero_init_outputs_zero_residual() -> None:
    model = GenericResidualAdapter()
    x = torch.randn(2, 6, 27, 43, dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 2, 27, 43)
    assert torch.count_nonzero(y).item() == 0
