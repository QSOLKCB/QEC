from __future__ import annotations

import numpy as np
import pytest

from src.qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_hybrid_mode_is_opt_in_default_exact() -> None:
    H = _H()
    _, log = NBEigenmodeMutation(enabled=True).mutate(H)
    if log:
        assert log[0]["score_mode"] == "exact"


def test_topk_config_validation_in_initializer() -> None:
    with pytest.raises(ValueError, match="top_k_exact_recheck"):
        NBEigenmodeMutation(enabled=True, use_nb_perturbation_scoring=True, top_k_exact_recheck=0)


def test_topk_config_validation_against_candidate_count() -> None:
    H = _H()
    with pytest.raises(ValueError, match="candidate_count"):
        NBEigenmodeMutation(
            enabled=True,
            use_nb_perturbation_scoring=True,
            top_k_exact_recheck=1000,
        ).mutate(H)


def test_hybrid_shortlist_rechecks_only_topk() -> None:
    H = _H()
    _, log = NBEigenmodeMutation(
        enabled=True,
        use_nb_perturbation_scoring=True,
        top_k_exact_recheck=2,
    ).mutate(H)
    if log:
        assert log[0]["exact_rechecks"] <= 2
        assert log[0]["candidate_count"] >= log[0]["exact_rechecks"]


def test_pressure_weighting_active_in_hybrid_mode() -> None:
    H = _H()
    _, log = NBEigenmodeMutation(
        enabled=True,
        use_nb_perturbation_scoring=True,
        top_k_exact_recheck=2,
    ).mutate(H)
    if log:
        assert "pressure_weight" in log[0]
        assert "weighted_delta" in log[0]


def test_invariants_preserved_in_hybrid_mode() -> None:
    H = _H()
    out, _ = NBEigenmodeMutation(
        enabled=True,
        use_nb_perturbation_scoring=True,
        top_k_exact_recheck=3,
    ).mutate(H)
    assert out.shape == H.shape
    assert float(out.sum()) == float(H.sum())
    np.testing.assert_array_equal(out.sum(axis=0), H.sum(axis=0))
    np.testing.assert_array_equal(out.sum(axis=1), H.sum(axis=1))


def test_deterministic_selection_preserved_in_hybrid_mode() -> None:
    H = _H()
    mutator = NBEigenmodeMutation(
        enabled=True,
        use_nb_perturbation_scoring=True,
        top_k_exact_recheck=2,
    )
    out1, log1 = mutator.mutate(H)
    out2, log2 = mutator.mutate(H)
    np.testing.assert_array_equal(out1, out2)
    assert log1 == log2
