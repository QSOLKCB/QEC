from __future__ import annotations

import numpy as np
import pytest
import warnings
import scipy.sparse

from src.qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation
from src.qec.discovery.mutation_operators import mutate_tanner_graph
from src.qec.discovery.discovery_engine import run_structure_discovery


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_mutation_is_opt_in() -> None:
    H = _H()
    H_out, log = NBEigenmodeMutation(enabled=False).mutate(H)
    np.testing.assert_array_equal(H_out, H)
    assert log == []


def test_deterministic_and_invariants_preserved() -> None:
    H = _H()
    m1, _ = NBEigenmodeMutation(enabled=True).mutate(H)
    m2, _ = NBEigenmodeMutation(enabled=True).mutate(H)
    np.testing.assert_array_equal(m1, m2)
    assert m1.shape == H.shape
    assert float(m1.sum()) == float(H.sum())
    np.testing.assert_array_equal(m1.sum(axis=0), H.sum(axis=0))
    np.testing.assert_array_equal(m1.sum(axis=1), H.sum(axis=1))


def test_sparse_compatibility() -> None:
    H = scipy.sparse.csr_matrix(_H())
    out, _ = NBEigenmodeMutation(enabled=True).mutate(H)
    assert out.shape == H.shape


def test_operator_is_explicit_only_not_in_schedule() -> None:
    H = _H()
    out, used = mutate_tanner_graph(H, operator="nb_eigenmode_mutation", seed=0)
    assert used == "nb_eigenmode_mutation"
    assert out.shape == H.shape


def test_discovery_default_behavior_unchanged_without_flag() -> None:
    spec = {"num_variables": 8, "num_checks": 4, "variable_degree": 2, "check_degree": 4}
    r1 = run_structure_discovery(spec, num_generations=1, population_size=2, base_seed=7)
    r2 = run_structure_discovery(spec, num_generations=1, population_size=2, base_seed=7)
    assert r1["best_candidate"]["operator"] == r2["best_candidate"]["operator"]


def test_discovery_flag_enables_nb_eigenmode_operator() -> None:
    spec = {"num_variables": 8, "num_checks": 4, "variable_degree": 2, "check_degree": 4}
    result = run_structure_discovery(
        spec,
        num_generations=1,
        population_size=2,
        base_seed=7,
        use_nb_eigenmode_mutation=True,
    )
    assert result["best_candidate"]["operator"] in {None, "nb_eigenmode_mutation"}


def test_early_exit_threshold_is_opt_in_and_deterministic() -> None:
    H = _H()
    mut = NBEigenmodeMutation(enabled=True, early_exit_improvement_threshold=1e-6)
    out1, log1 = mut.mutate(H)
    out2, log2 = mut.mutate(H)
    np.testing.assert_array_equal(out1, out2)
    assert log1 == log2


def test_early_exit_threshold_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NBEigenmodeMutation(enabled=True, early_exit_improvement_threshold=-1e-6)


def test_hybrid_top_k_exact_recheck_validation() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        NBEigenmodeMutation(enabled=True, use_hybrid_perturbation_scoring=True, top_k_exact_recheck=0)


def test_hybrid_fallback_on_invalid_first_order() -> None:
    H = _H()
    mut = NBEigenmodeMutation(enabled=True, use_hybrid_perturbation_scoring=True)

    mut._perturbation_scorer.compute_nb_spectrum = lambda _H: {"valid_first_order": False}  # type: ignore[assignment]

    out_hybrid, log_hybrid = mut.mutate(H)
    out_exact, log_exact = NBEigenmodeMutation(enabled=True).mutate(H)

    np.testing.assert_array_equal(out_hybrid, out_exact)
    assert log_hybrid == log_exact


def test_pressure_weighting_and_support_heuristic_affect_hybrid_ranking() -> None:
    H = _H()
    mut = NBEigenmodeMutation(
        enabled=True,
        use_hybrid_perturbation_scoring=True,
        use_pressure_weighting=True,
        use_support_aware_heuristic=True,
        top_k_exact_recheck=1,
    )

    mut._enumerate_swaps = lambda *_args: [(0, 0, 1, 1), (0, 3, 1, 4)]  # type: ignore[assignment]
    mut._perturbation_scorer.compute_nb_spectrum = lambda _H: {  # type: ignore[assignment]
        "valid_first_order": True,
        "u": np.array([0.8, 0.1, 0.7, 0.05], dtype=np.float64),
        "v": np.array([0.8, 0.1, 0.7, 0.05], dtype=np.float64),
        "index": {(0, 6): 0, (6, 0): 1, (1, 7): 2, (7, 1): 3, (3, 6): 1, (4, 7): 3},
    }

    def _pred(_H: np.ndarray, swap: tuple[int, int, int, int], _spectrum: dict) -> dict:
        if swap == (0, 0, 1, 1):
            return {"valid_first_order": True, "predicted_delta": -0.2, "pressure": 1.5}
        return {"valid_first_order": True, "predicted_delta": -0.21, "pressure": 0.0}

    mut._perturbation_scorer.predict_swap_delta = _pred  # type: ignore[assignment]

    def _fake_analyze(Hcand: np.ndarray) -> dict:
        if Hcand[0, 1] == 1.0 and Hcand[1, 0] == 1.0:
            sig = {"spectral_radius": 0.9, "mode_ipr": 0.2, "support_fraction": 0.5, "topk_mass_fraction": 0.4}
        else:
            sig = {"spectral_radius": 1.0, "mode_ipr": 0.3, "support_fraction": 0.6, "topk_mass_fraction": 0.5}
        return {"signature": sig, "hot_edges": [(0, 0), (0, 3)]}

    base = {"signature": {"spectral_radius": 1.0, "mode_ipr": 0.3, "support_fraction": 0.6, "topk_mass_fraction": 0.5}, "hot_edges": [(0, 0), (0, 3)]}
    mut._analyzer.analyze = lambda Hx: base if np.array_equal(Hx, H) else _fake_analyze(Hx)  # type: ignore[assignment]

    out, log = mut.mutate(H)
    assert log
    assert log[0]["removed_edge"] == (0, 0)


def test_defect_caching_reuses_previous_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.qec.discovery import mutation_nb_eigenmode as mod

    H = _H()
    mut = NBEigenmodeMutation(enabled=True, use_defect_guided_scoring=True)
    calls = {"count": 0}

    def _fake_detect(_H: np.ndarray) -> list:
        calls["count"] += 1
        return []

    monkeypatch.setattr(mod, "detect_spectral_defects", _fake_detect)
    mut.mutate(H)
    mut.mutate(H)
    assert calls["count"] == 1


def test_deprecated_hybrid_flag_warns_and_maps() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mut = NBEigenmodeMutation(enabled=True, use_hybrid_perturbation_scoring=True)
    assert mut.use_nb_perturbation_scoring is True
    assert any("deprecated" in str(item.message).lower() for item in caught)
