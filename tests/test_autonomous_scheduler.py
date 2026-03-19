"""Tests for v43.0.0 autonomous discovery scheduling."""

from __future__ import annotations

import json

import numpy as np

from qec.analysis.landscape_gaps import detect_landscape_gaps
from qec.analysis.scheduling_metrics import (
    landscape_gap_count,
    mean_gap_distance,
    scheduled_experiment_targets,
)
from qec.discovery.autonomous_scheduler import schedule_next_experiment
from qec.discovery.discovery_engine import run_structure_discovery
from qec.discovery.experiment_queue import ExperimentQueue
from qec.discovery.experiment_targets import choose_experiment_target


class _Memory:
    def __init__(self, centers: list[list[float]]) -> None:
        self._centers = np.asarray(centers, dtype=np.float64)

    def centers(self) -> np.ndarray:
        return self._centers


def _memory() -> _Memory:
    return _Memory(
        [
            [1.0, 0.2, 0.1, 0.3],
            [1.5, 0.4, 0.2, 0.4],
            [1.2, 0.3, 0.15, 0.5],
        ]
    )


def test_gap_detection_is_deterministic() -> None:
    mem = _memory()
    g1 = detect_landscape_gaps(mem, gap_radius=0.1, max_gaps=8)
    g2 = detect_landscape_gaps(mem, gap_radius=0.1, max_gaps=8)
    assert json.dumps([x.tolist() for x in g1], sort_keys=True) == json.dumps(
        [x.tolist() for x in g2], sort_keys=True,
    )


def test_target_selection_is_deterministic() -> None:
    gaps = detect_landscape_gaps(_memory(), gap_radius=0.1, max_gaps=8)
    t1 = choose_experiment_target(gaps)
    t2 = choose_experiment_target(gaps)
    if t1 is None:
        assert t2 is None
    else:
        assert np.allclose(t1, t2)


def test_scheduler_reproducibility() -> None:
    s1 = schedule_next_experiment(_memory(), gap_radius=0.1, max_gaps=8)
    s2 = schedule_next_experiment(_memory(), gap_radius=0.1, max_gaps=8)
    assert s1["strategy"] == "landscape_exploration"
    assert s1["gap_count"] == s2["gap_count"]
    assert json.dumps(
        None if s1["target_spectrum"] is None else np.asarray(s1["target_spectrum"]).tolist(),
        sort_keys=True,
    ) == json.dumps(
        None if s2["target_spectrum"] is None else np.asarray(s2["target_spectrum"]).tolist(),
        sort_keys=True,
    )


def test_experiment_queue_fifo_behavior() -> None:
    q = ExperimentQueue(max_length=4)
    assert q.size() == 0
    assert q.pop() is None

    q.push([1.0, 2.0])
    q.push([3.0, 4.0])
    assert q.size() == 2
    assert np.allclose(q.pop(), np.asarray([1.0, 2.0], dtype=np.float64))
    assert np.allclose(q.pop(), np.asarray([3.0, 4.0], dtype=np.float64))
    assert q.pop() is None


def test_scheduling_metrics_basic() -> None:
    mem = _memory()
    assert landscape_gap_count(mem, gap_radius=0.1, max_gaps=8) >= 0
    assert mean_gap_distance(mem, gap_radius=0.1, max_gaps=8) >= 0.0

    q = ExperimentQueue()
    q.push([0.1, 0.2])
    q.push([0.3, 0.4])
    targets = scheduled_experiment_targets(q)
    assert len(targets) == 2
    assert all(isinstance(t, np.ndarray) for t in targets)


def test_discovery_engine_autonomous_scheduler_deterministic() -> None:
    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    q1 = ExperimentQueue(max_length=16)
    q2 = ExperimentQueue(max_length=16)
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=17,
        enable_autonomous_scheduler=True,
        scheduler_gap_radius=0.1,
        scheduler_max_gaps=8,
        scheduler_queue=q1,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=17,
        enable_autonomous_scheduler=True,
        scheduler_gap_radius=0.1,
        scheduler_max_gaps=8,
        scheduler_queue=q2,
    )

    assert r1["scheduler_strategy"] == "landscape_exploration"
    assert r1["scheduler_strategy"] == r2["scheduler_strategy"]
    assert r1["landscape_gap_count"] == r2["landscape_gap_count"]
    assert json.dumps(r1.get("scheduled_target_spectrum"), sort_keys=True) == json.dumps(
        r2.get("scheduled_target_spectrum"), sort_keys=True,
    )
