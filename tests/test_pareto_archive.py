from __future__ import annotations

import json

from src.qec.discovery.pareto_archive import ParetoArchive, ParetoMetrics


def test_deterministic_pareto_insertion_and_frontier_order() -> None:
    archive = ParetoArchive()

    graph_a = [[1.0, 0.0], [0.0, 1.0]]
    graph_b = [[1.0, 1.0], [0.0, 0.0]]
    graph_c = [[0.0, 1.0], [1.0, 0.0]]

    kept_a = archive.add_candidate(ParetoMetrics(0.04, 0.60, 0.30), graph_a)
    kept_b = archive.add_candidate(ParetoMetrics(0.05, 0.55, 0.40), graph_b)
    kept_c = archive.add_candidate(ParetoMetrics(0.03, 0.40, 0.20), graph_c)

    assert kept_a is True
    assert kept_b is True
    assert kept_c is False

    frontier = archive.get_frontier()
    assert len(frontier) == 2
    assert frontier[0]["graph"] == graph_a
    assert frontier[1]["graph"] == graph_b


def test_pareto_frontier_serialization_is_deterministic(tmp_path) -> None:
    archive = ParetoArchive()
    archive.add_candidate(ParetoMetrics(0.05, 0.5, 0.25), [[1.0, 0.0], [0.0, 1.0]])

    path = tmp_path / "pareto_frontier.json"
    archive.save_frontier(path)
    first = path.read_text(encoding="utf-8")

    archive.save_frontier(path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first) == json.loads(second)
