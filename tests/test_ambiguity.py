import numpy as np

from zakotfs.ambiguity import cross_ambiguity


def test_cross_ambiguity_delta_reference_returns_signal():
    M, N = 5, 7
    a = np.zeros((M, N), dtype=np.complex64)
    a[2, 3] = 2.0 + 1.0j
    b = np.zeros((M, N), dtype=np.complex64)
    b[0, 0] = 1.0 + 0.0j
    amb = cross_ambiguity(a, b)
    assert np.allclose(amb, a)
