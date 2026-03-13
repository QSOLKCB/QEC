from __future__ import annotations

import numpy as np

from src.qec.discovery.nb_flow_mutation import NBFlowMutationConfig, NonBacktrackingFlowMutator


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def test_near_degenerate_modes_are_stable_and_deterministic() -> None:
    H = _matrix()
    mut = NonBacktrackingFlowMutator(config=NBFlowMutationConfig(enabled=True, max_flow_edges=8))

    field = mut._analyzer.build_flow_field(H)
    de = len(field.directed_edges)
    assert de > 0

    vals = np.array([2.0 + 0.0j, 2.0 + 1e-12j], dtype=np.complex128)
    vec1 = np.ones(de, dtype=np.complex128)
    vec2 = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(de)], dtype=np.complex128)
    vecs = np.column_stack([vec1, vec2])

    mut._analyzer.compute_modes = lambda _H: (vals, vecs)  # type: ignore[assignment]

    out_a, log_a = mut.mutate(H)
    out_b, log_b = mut.mutate(H)

    np.testing.assert_array_equal(out_a, out_b)
    assert log_a == log_b
    if log_a:
        assert int(log_a[0]["flow_mode_index"]) == 1
