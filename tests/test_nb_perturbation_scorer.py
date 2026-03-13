from __future__ import annotations

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_perturbation_scorer import NBPerturbationScorer


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_perturbation_score_is_deterministic() -> None:
    H = _H()
    swap = (0, 1, 2, 5)
    scorer = NBPerturbationScorer()
    state = scorer.baseline_state(H)
    r1 = scorer.score_swap(H, swap, state)
    r2 = scorer.score_swap(H, swap, state)
    assert r1 == r2


def test_eigenvectors_are_l2_normalized() -> None:
    H = _H()
    scorer = NBPerturbationScorer()
    state = scorer.baseline_state(H)
    u = np.asarray(state["right_eigenvector"], dtype=np.complex128)
    v = np.asarray(state["left_eigenvector"], dtype=np.complex128)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-10)


def test_sparse_compatibility_and_local_nnz() -> None:
    H = scipy.sparse.csr_matrix(_H())
    scorer = NBPerturbationScorer()
    state = scorer.baseline_state(H)
    result = scorer.score_swap(H, (0, 1, 2, 5), state)
    assert result is not None
    assert isinstance(result["perturbation_nnz"], int)
    assert result["perturbation_nnz"] >= 0
    assert result["perturbation_nnz"] <= max(1, result.get("directed_edges_count", 1))


def test_invalid_swap_falls_back_with_none() -> None:
    H = _H()
    scorer = NBPerturbationScorer()
    state = scorer.baseline_state(H)
    result = scorer.score_swap(H, (0, 0, 0, 1), state)
    assert result is None


def test_invalid_first_order_state_falls_back_with_none() -> None:
    H = _H()
    scorer = NBPerturbationScorer()
    state = scorer.baseline_state(H)
    state["valid_first_order"] = False
    result = scorer.score_swap(H, (0, 1, 2, 5), state)
    assert result is None
