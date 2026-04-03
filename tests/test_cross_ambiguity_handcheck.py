import numpy as np

from zakotfs.ambiguity import cross_ambiguity


def _manual_cross_ambiguity(a: np.ndarray, b: np.ndarray, k: int, l: int) -> np.complex64:
    M, N = a.shape
    Q = M * N
    acc = 0.0j
    for kp in range(M):
        for lp in range(N):
            shifted_k = kp - k
            shifted_l = lp - l
            base_k = shifted_k % M
            base_l = shifted_l % N
            wrap_k = (shifted_k - base_k) // M
            b_shifted = b[base_k, base_l] * np.exp(1j * 2 * np.pi * wrap_k * base_l / N)
            acc += a[kp, lp] * np.conjugate(b_shifted) * np.exp(-1j * 2 * np.pi * l * shifted_k / Q)
    return np.complex64(acc)


def test_cross_ambiguity_matches_hand_computation():
    a = np.array(
        [
            [1 + 0j, 2 - 1j, 0.5 + 0.25j],
            [-1j, 0.0 + 0j, 1.0 + 2.0j],
            [0.25 - 0.5j, -0.75 + 0.1j, 0.5 + 0j],
        ],
        dtype=np.complex64,
    )
    b = np.array(
        [
            [0.25 + 0j, -1.0 + 0.5j, 0.5 - 0.25j],
            [1.0 + 0j, -0.5j, -0.75 + 0j],
            [0.3 + 0.1j, 0.0 + 0j, 1.0 - 1.0j],
        ],
        dtype=np.complex64,
    )
    amb = cross_ambiguity(a, b)
    for k, l in [(0, 0), (1, 1), (2, 2)]:
        assert np.allclose(amb[k, l], _manual_cross_ambiguity(a, b, k, l), atol=1e-6, rtol=1e-6)
