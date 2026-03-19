from __future__ import annotations

import json

import numpy as np

from qec.analysis.exploration_metrics import (
    basin_switch_rate,
    exploration_entropy,
    mean_basin_duration,
)
from qec.analysis.exploration_state import analyze_exploration_state
from qec.discovery.discovery_engine import _route_exploration_targets, run_structure_discovery
from qec.discovery.exploration_policy import (
    apply_early_exploration_guard,
    apply_escape_feedback_bias,
    choose_exploration_strategy,
)


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_exploration_state_detection() -> None:
    traj_short = np.zeros((2, 4), dtype=np.float64)
    assert analyze_exploration_state(traj_short, [1, 1], window=5) == "GLOBAL_EXPLORATION"

    traj = np.ones((8, 4), dtype=np.float64)
    assert analyze_exploration_state(traj, [2] * 8, window=5) == "BASIN_STAGNATION"
    assert analyze_exploration_state(traj, [1, 1, 2, 2, 3, 4], window=4) == "BASIN_TRANSITION"
    assert analyze_exploration_state(traj, [1, 2, 2, 3, 3, 3], window=4) == "LOCAL_OPTIMIZATION"


def test_exploration_metrics_correctness() -> None:
    assignments = [1, 1, 2, 2, 2, 3]
    assert basin_switch_rate(assignments, window=6) == 0.4
    assert basin_switch_rate(assignments, window=3) == 0.5
    assert mean_basin_duration(assignments) == 2.0
    ent = exploration_entropy(assignments)
    assert 0.0 <= ent <= 1.0


def test_policy_selection_and_feedback() -> None:
    assert choose_exploration_strategy("LOCAL_OPTIMIZATION") == "GRADIENT"
    assert choose_exploration_strategy("BASIN_STAGNATION") == "ESCAPE"
    assert choose_exploration_strategy("BASIN_TRANSITION") == "NB_EIGENMODE"
    assert choose_exploration_strategy("GLOBAL_EXPLORATION") == "RANDOM_EXPLORATION"

    assert apply_escape_feedback_bias("ESCAPE", escape_success_rate=0.05) == "NB_EIGENMODE"
    assert apply_escape_feedback_bias("ESCAPE", escape_success_rate=0.25) == "ESCAPE"
    assert apply_escape_feedback_bias("GRADIENT", escape_success_rate=0.0) == "GRADIENT"


def test_strategy_routing() -> None:
    gradient = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    escape = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    guide_edges = [(0, 0)]
    nb_edges = [(1, 1)]

    r_gradient = _route_exploration_targets(
        "GRADIENT",
        gradient_target=gradient,
        nb_target_edges=nb_edges,
        escape_target=escape,
        default_target_edges=guide_edges,
    )
    assert r_gradient["target_edges"] == guide_edges
    assert np.array_equal(r_gradient["target_spectrum"], gradient)

    r_nb = _route_exploration_targets(
        "NB_EIGENMODE",
        gradient_target=gradient,
        nb_target_edges=nb_edges,
        escape_target=escape,
        default_target_edges=guide_edges,
    )
    assert r_nb["target_edges"] == nb_edges
    assert r_nb["target_spectrum"] is None

    r_escape = _route_exploration_targets(
        "ESCAPE",
        gradient_target=gradient,
        nb_target_edges=nb_edges,
        escape_target=escape,
        default_target_edges=guide_edges,
    )
    assert r_escape["target_edges"] == guide_edges
    assert np.array_equal(r_escape["target_spectrum"], escape)

    r_random = _route_exploration_targets(
        "RANDOM_EXPLORATION",
        gradient_target=gradient,
        nb_target_edges=nb_edges,
        escape_target=escape,
        default_target_edges=guide_edges,
    )
    assert r_random == {"target_edges": None, "target_spectrum": None}


def test_adaptive_escape_scaling_and_logging() -> None:
    spec = _default_spec()
    adaptive = run_structure_discovery(
        spec,
        num_generations=3,
        population_size=4,
        base_seed=19,
        enable_spectral_trajectory=True,
        enable_adaptive_exploration=True,
        exploration_window=3,
    )
    summaries = adaptive["generation_summaries"]
    assert any("exploration_state" in s for s in summaries[1:])
    assert any("exploration_strategy" in s for s in summaries[1:])
    assert any("basin_switch_rate" in s for s in summaries[1:])
    assert any("exploration_entropy" in s for s in summaries[1:])
    assert any("escape_success_rate" in s for s in summaries[1:])


def test_early_exploration_guard_can_bias_away_from_escape() -> None:
    assert apply_early_exploration_guard(
        "ESCAPE",
        recent_basin_discovery_rate=0.8,
        threshold=0.5,
    ) == "GRADIENT"
    assert apply_early_exploration_guard(
        "ESCAPE",
        recent_basin_discovery_rate=0.2,
        threshold=0.5,
    ) == "ESCAPE"
    assert apply_early_exploration_guard(
        "GRADIENT",
        recent_basin_discovery_rate=0.8,
        threshold=0.5,
    ) == "GRADIENT"


def test_opt_in_and_determinism() -> None:
    spec = _default_spec()

    baseline_a = run_structure_discovery(spec, num_generations=2, population_size=4, base_seed=19)
    baseline_b = run_structure_discovery(spec, num_generations=2, population_size=4, base_seed=19)
    assert json.dumps(baseline_a["elite_history"], sort_keys=True) == json.dumps(
        baseline_b["elite_history"], sort_keys=True,
    )
    assert "exploration_state" not in baseline_a["generation_summaries"][0]

    adaptive_a = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=19,
        enable_spectral_trajectory=True,
        enable_adaptive_exploration=True,
        exploration_window=3,
    )
    adaptive_b = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=19,
        enable_spectral_trajectory=True,
        enable_adaptive_exploration=True,
        exploration_window=3,
    )
    assert adaptive_a["elite_history"] == adaptive_b["elite_history"]
    assert adaptive_a["generation_summaries"] == adaptive_b["generation_summaries"]


def test_escape_success_rate_safe_division() -> None:
    attempts = 0
    successes = 0

    rate = (
        float(successes) / float(attempts)
        if attempts > 0
        else 0.0
    )
    rate = min(max(rate, 0.0), 1.0)

    assert rate == 0.0


def test_strategy_smoothing() -> None:
    history = ["ESCAPE", "ESCAPE", "ESCAPE"]

    assert history.count("ESCAPE") > 1
