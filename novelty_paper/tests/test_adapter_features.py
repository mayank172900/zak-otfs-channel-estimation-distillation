from __future__ import annotations

import numpy as np

from zakotfs_novelty.adapter import build_fb_lara_features, feature_channel_indices, residual_target


def test_fb_lara_feature_stack_layout() -> None:
    h_raw = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    h_base = np.array([[0.5 + 1.5j, 2.5 + 3.5j]], dtype=np.complex64)
    h_alias = np.array([[0.1 - 0.2j, -0.3 + 0.4j]], dtype=np.complex64)
    features = build_fb_lara_features(h_raw, h_base, h_alias)
    target = residual_target(h_raw, h_base)
    assert features.shape == (8, 1, 2)
    assert np.allclose(features[0], [[1.0, 3.0]])
    assert np.allclose(features[1], [[2.0, 4.0]])
    assert np.allclose(features[4], [[0.1, -0.3]])
    assert np.allclose(features[5], [[-0.2, 0.4]])
    assert np.allclose(features[6], [[0.5, 0.5]])
    assert np.allclose(features[7], [[0.5, 0.5]])
    assert target.shape == (2, 1, 2)
    assert np.allclose(target[0], [[0.5, 0.5]])
    assert np.allclose(target[1], [[0.5, 0.5]])
    assert feature_channel_indices("generic") == (0, 1, 2, 3, 6, 7)
    assert feature_channel_indices("fb_lara") == tuple(range(8))
