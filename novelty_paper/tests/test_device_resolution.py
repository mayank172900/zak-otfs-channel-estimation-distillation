from __future__ import annotations

from zakotfs_novelty.utils import resolve_torch_device


def test_resolve_torch_device_cpu_explicit() -> None:
    assert resolve_torch_device(explicit="cpu").type == "cpu"
