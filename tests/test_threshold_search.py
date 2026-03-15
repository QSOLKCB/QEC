from __future__ import annotations

import numpy as np

from src.qec.discovery.threshold_search import ThresholdSearchEngine


def _spec() -> dict[str, int]:
    return {
        "num_variables": 12,
        "num_checks": 6,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_search_is_deterministic_for_same_seed(tmp_path) -> None:
    artifacts_a = tmp_path / "a"
    artifacts_b = tmp_path / "b"

    engine_a = ThresholdSearchEngine(
        _spec(),
        iterations=3,
        population=4,
        seed=123,
        max_graphs_evaluated=12,
        artifacts_root=str(artifacts_a),
    )
    engine_b = ThresholdSearchEngine(
        _spec(),
        iterations=3,
        population=4,
        seed=123,
        max_graphs_evaluated=12,
        artifacts_root=str(artifacts_b),
    )

    result_a = engine_a.search()
    result_b = engine_b.search()

    assert result_a["history"] == result_b["history"]
    assert result_a["best"] == result_b["best"]


def test_mutations_preserve_shape_and_binary_values(tmp_path) -> None:
    engine = ThresholdSearchEngine(
        _spec(),
        iterations=1,
        population=4,
        seed=7,
        max_graphs_evaluated=4,
        artifacts_root=str(tmp_path / "mutation"),
    )
    result = engine.search()

    for entry in result["history"]:
        H = np.asarray(entry["graph_structure"], dtype=np.float64)
        assert H.shape == (6, 12)
        assert np.all(np.isin(H, [0.0, 1.0]))
        assert np.all(np.sum(H, axis=0) >= 1.0)
        assert np.all(np.sum(H, axis=1) >= 1.0)


def test_threshold_evaluator_is_used(tmp_path) -> None:
    calls = {"count": 0}

    def evaluator(H: np.ndarray) -> float:
        calls["count"] += 1
        return float(np.sum(H) / 1000.0)

    engine = ThresholdSearchEngine(
        _spec(),
        iterations=2,
        population=3,
        seed=5,
        max_graphs_evaluated=6,
        artifacts_root=str(tmp_path / "eval"),
        threshold_evaluator=evaluator,
    )

    result = engine.search()

    assert calls["count"] == len(result["history"])
    for entry in result["history"]:
        assert entry["threshold"] >= 0.0
