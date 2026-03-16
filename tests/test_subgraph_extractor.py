from __future__ import annotations

import numpy as np

from src.qec.analysis.api import extract_support_subgraph
from src.qec.analysis.subgraph_extractor import compute_trapping_parameters


def test_compute_trapping_parameters_ab_counts_unsatisfied_checks() -> None:
    H = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.float64)
    sub = extract_support_subgraph(H, [0, 1, 2])
    a, b = compute_trapping_parameters(sub)
    assert a == 3
    assert b == 1


def test_extract_support_subgraph_is_deterministic() -> None:
    H = np.array([
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.float64)
    s1 = extract_support_subgraph(H, [2, 0])
    s2 = extract_support_subgraph(H, [0, 2])
    assert s1["support_nodes"] == s2["support_nodes"] == [0, 2]
    assert s1["check_nodes"] == s2["check_nodes"]
    np.testing.assert_array_equal(s1["submatrix"].toarray(), s2["submatrix"].toarray())
