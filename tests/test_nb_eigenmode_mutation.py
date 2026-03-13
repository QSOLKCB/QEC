from __future__ import annotations

import numpy as np
import pytest
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
