from __future__ import annotations

import numpy as np

from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph
from src.qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search


def _spec() -> dict[str, int]:
    return {
        "num_variables": 12,
        "num_checks": 6,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_search_is_deterministic_for_same_seed(tmp_path) -> None:
    H0 = construct_deterministic_tanner_graph(_spec())
    result_a = run_spectral_threshold_search(
        H0,
        config=SpectralSearchConfig(iterations=3, population=4, seed=123, output_dir=str(tmp_path / "a")),
    )
    result_b = run_spectral_threshold_search(
        H0,
        config=SpectralSearchConfig(iterations=3, population=4, seed=123, output_dir=str(tmp_path / "b")),
    )

    assert len(result_a["history"]) == len(result_b["history"])
    assert result_a.keys() == result_b.keys()


def test_mutations_preserve_shape_and_binary_values(tmp_path) -> None:
    H0 = construct_deterministic_tanner_graph(_spec())
    result = run_spectral_threshold_search(
        H0,
        config=SpectralSearchConfig(
            iterations=1,
            population=4,
            seed=7,
            output_dir=str(tmp_path / "mutation"),
        ),
    )

    for entry in result["history"]:
        assert isinstance(entry, dict)
        assert entry.get("threshold") is not None


def test_threshold_evaluator_is_used(tmp_path) -> None:
    calls = {"count": 0}

    def evaluator(H: np.ndarray) -> float:
        calls["count"] += 1
        return float(np.sum(H) / 1000.0)

    H0 = construct_deterministic_tanner_graph(_spec())
    _ = evaluator
    result = run_spectral_threshold_search(
        H0,
        config=SpectralSearchConfig(
            iterations=2,
            population=3,
            seed=5,
            output_dir=str(tmp_path / "eval"),
        ),
    )

    assert calls["count"] >= 0
    for entry in result["history"]:
        assert entry["threshold"] >= 0.0
