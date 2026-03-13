from __future__ import annotations

import numpy as np

from src.qec.discovery.swap_helpers import deterministic_two_edge_swap


def test_deterministic_two_edge_swap_preserves_invariants() -> None:
    H = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
    ], dtype=np.float64)

    out1 = deterministic_two_edge_swap(H)
    out2 = deterministic_two_edge_swap(H)

    np.testing.assert_array_equal(out1, out2)
    assert out1.shape == H.shape
    assert float(out1.sum()) == float(H.sum())
    np.testing.assert_array_equal(out1.sum(axis=0), H.sum(axis=0))
    np.testing.assert_array_equal(out1.sum(axis=1), H.sum(axis=1))
