"""Tests for v68.1 diagnostics-aware multi-objective fitness.

Verifies deterministic outputs, float64 dtypes, ranking stability,
and compatibility with v68 fitness metrics.
"""
from __future__ import annotations

import numpy as np

from qec.decoder.ternary.ternary_rule_fitness import (
    compute_rule_fitness_metrics,
    rank_rules_by_fitness,
    compute_multiobjective_fitness,
    project_fitness_score,
    rank_rules_multiobjective,
)


def _sample_population() -> dict:
    return {
        "decoder_rule_population": [
            {
                "rule_name": "rule_a",
                "converged": True,
                "iterations": 5,
                "stability": 0.9,
                "entropy": 0.3,
                "conflict_density": 0.1,
                "trapping_indicator": 0.05,
            },
            {
                "rule_name": "rule_b",
                "converged": False,
                "iterations": 10,
                "stability": 0.4,
                "entropy": 0.8,
                "conflict_density": 0.5,
                "trapping_indicator": 0.3,
            },
            {
                "rule_name": "rule_c",
                "converged": True,
                "iterations": 3,
                "stability": 0.7,
                "entropy": 0.2,
                "conflict_density": 0.05,
                "trapping_indicator": 0.01,
            },
        ],
    }


def _sample_metrics() -> dict[str, dict[str, np.float64]]:
    return compute_rule_fitness_metrics(_sample_population())


def test_compute_multiobjective_fitness_determinism() -> None:
    metrics = _sample_metrics()
    v1 = compute_multiobjective_fitness(metrics)
    v2 = compute_multiobjective_fitness(metrics)
    assert sorted(v1.keys()) == sorted(v2.keys())
    for r in v1:
        for k in v1[r]:
            assert v1[r][k] == v2[r][k], f"Mismatch at {r}/{k}"


def test_fitness_vector_keys() -> None:
    metrics = _sample_metrics()
    vectors = compute_multiobjective_fitness(metrics)
    expected_keys = {
        "performance",
        "efficiency",
        "agreement",
        "entropy_penalty",
        "conflict_penalty",
        "trapping_penalty",
    }
    for r in vectors:
        assert set(vectors[r].keys()) == expected_keys


def test_fitness_vector_dtypes() -> None:
    metrics = _sample_metrics()
    vectors = compute_multiobjective_fitness(metrics)
    for r in vectors:
        for k, v in vectors[r].items():
            assert isinstance(v, np.float64), f"{r}/{k} is {type(v)}, expected np.float64"


def test_project_fitness_score_determinism() -> None:
    metrics = _sample_metrics()
    vectors = compute_multiobjective_fitness(metrics)
    for r in vectors:
        s1 = project_fitness_score(vectors[r])
        s2 = project_fitness_score(vectors[r])
        assert s1 == s2
        assert isinstance(s1, np.float64)


def test_project_fitness_score_values() -> None:
    f = {
        "performance": np.float64(1.0),
        "efficiency": np.float64(1.0),
        "agreement": np.float64(1.0),
        "entropy_penalty": np.float64(0.0),
        "conflict_penalty": np.float64(0.0),
        "trapping_penalty": np.float64(0.0),
    }
    score = project_fitness_score(f)
    expected = np.float64(2.0 * 1.0 + 1.5 * 1.0 + 1.0 * 1.0)
    assert np.isclose(score, expected)


def test_rank_rules_multiobjective_determinism() -> None:
    metrics = _sample_metrics()
    vectors = compute_multiobjective_fitness(metrics)
    r1 = rank_rules_multiobjective(vectors)
    r2 = rank_rules_multiobjective(vectors)
    assert len(r1) == len(r2)
    for (n1, s1), (n2, s2) in zip(r1, r2):
        assert n1 == n2
        assert s1 == s2


def test_rank_rules_multiobjective_ordering() -> None:
    metrics = _sample_metrics()
    vectors = compute_multiobjective_fitness(metrics)
    ranked = rank_rules_multiobjective(vectors)
    scores = [float(s) for _, s in ranked]
    # Scores should be non-increasing (best first)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score at {i} ({scores[i]}) < score at {i+1} ({scores[i+1]})"


def test_rank_rules_multiobjective_empty() -> None:
    assert rank_rules_multiobjective({}) == []


def test_compatibility_with_v68_metrics() -> None:
    """Ensure multi-objective fitness consumes v68 compute_rule_fitness_metrics output."""
    pop = _sample_population()
    metrics = compute_rule_fitness_metrics(pop)
    ranked_v68 = rank_rules_by_fitness(metrics)

    vectors = compute_multiobjective_fitness(metrics)
    ranked_mo = rank_rules_multiobjective(vectors)

    # Both should rank the same number of rules
    assert len(ranked_v68) == len(ranked_mo)
    # All rule names should be present
    v68_names = {name for name, _ in ranked_v68}
    mo_names = {name for name, _ in ranked_mo}
    assert v68_names == mo_names


def test_missing_diagnostics_safe_fallback() -> None:
    """Rules without diagnostics keys should get safe defaults."""
    pop = {
        "decoder_rule_population": [
            {
                "rule_name": "bare_rule",
                "converged": True,
                "iterations": 4,
            },
        ],
    }
    metrics = compute_rule_fitness_metrics(pop)
    vectors = compute_multiobjective_fitness(metrics)
    assert "bare_rule" in vectors
    v = vectors["bare_rule"]
    assert isinstance(v["agreement"], np.float64)
    assert v["agreement"] == np.float64(0.0)
    assert v["conflict_penalty"] == np.float64(0.0)


def test_api_wrappers() -> None:
    from qec.analysis.api import (
        compute_multiobjective_fitness as api_mo,
        project_fitness_score as api_proj,
        rank_rules_multiobjective as api_rank,
    )
    metrics = _sample_metrics()
    vectors = api_mo(metrics)
    assert len(vectors) == 3
    for r in vectors:
        score = api_proj(vectors[r])
        assert isinstance(score, np.float64)
    ranked = api_rank(vectors)
    assert len(ranked) == 3
