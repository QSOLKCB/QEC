from __future__ import annotations

import numpy as np

from src.qec.analysis.subgraph_extractor import extract_induced_subgraph


def test_subgraph_ab_parameters() -> None:
    H = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.float64)
    out = extract_induced_subgraph(H, [0, 1, 2])
    assert out["a"] == 3
    assert out["b"] == 1
    assert out["support_nodes"] == [0, 1, 2]


def test_subgraph_empty_support() -> None:
    H = np.zeros((2, 3), dtype=np.float64)
    out = extract_induced_subgraph(H, [])
    assert out["a"] == 0 and out["b"] == 0
