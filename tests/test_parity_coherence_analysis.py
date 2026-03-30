"""Deterministic tests for v108.0.0 parity/coherence analysis."""

from __future__ import annotations

from qec.analysis.parity_coherence import run_parity_coherence_analysis
from qec.analysis.strategy_adapter import run_trajectory_analysis


REQUIRED_KEYS = {
    "parity_state",
    "global_probe_score",
    "local_probe_score",
    "probe_disagreement",
    "coherence_length",
    "parity_jump_detected",
    "parity_stability_score",
}


def test_stable_global_with_local_flicker() -> None:
    # Deltas: + + + - + + => global=5/6, local=1-2/5
    trajectory = [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
    out = run_parity_coherence_analysis(trajectory)

    assert out["parity_state"] == "stable_positive"
    assert out["global_probe_score"] > out["local_probe_score"]
    assert out["global_probe_score"] >= 0.8
    assert out["probe_disagreement"] > 0.0
    assert out["parity_jump_detected"] is False


def test_genuine_persistent_transition_detected() -> None:
    # Deltas: + + + - - - => persistent transition
    trajectory = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
    out = run_parity_coherence_analysis(trajectory)

    assert out["parity_jump_detected"] is True
    assert out["coherence_length"] == 3


def test_short_noisy_oscillation_no_false_jump() -> None:
    # Deltas alternate (+,-,+,-,+,-); no run reaches min_run=2
    trajectory = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    out = run_parity_coherence_analysis(trajectory)

    assert out["parity_jump_detected"] is False


def test_zero_delta_breaks_coherence_contiguity() -> None:
    # Signs: [+1, 0, +1] => two runs of length 1 (not contiguous length 2)
    trajectory = [0.0, 1.0, 1.0, 2.0]
    out = run_parity_coherence_analysis(trajectory)
    assert out["coherence_length"] == 1


def test_zero_delta_breaks_persistent_jump_detection() -> None:
    # Signs: [+,+,0,-,-] => zero breaks runs, so not adjacent persistent runs.
    trajectory = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]
    out = run_parity_coherence_analysis(trajectory)
    assert out["parity_jump_detected"] is False


def test_schema_and_boundedness() -> None:
    out = run_parity_coherence_analysis([1.0, 1.0, 1.0, 1.0])

    assert set(out.keys()) == REQUIRED_KEYS
    for key in (
        "global_probe_score",
        "local_probe_score",
        "probe_disagreement",
        "parity_stability_score",
    ):
        assert 0.0 <= out[key] <= 1.0

    assert isinstance(out["coherence_length"], int)
    assert out["coherence_length"] >= 0


def test_exact_determinism() -> None:
    trajectory = [0.2, 0.3, 0.5, 0.4, 0.6, 0.7]
    r1 = run_parity_coherence_analysis(trajectory)
    r2 = run_parity_coherence_analysis(trajectory)
    assert r1 == r2


def test_integration_opt_in_only() -> None:
    runs = [
        {"strategies": [{"name": "A", "metrics": {"design_score": 0.5}}]},
        {"strategies": [{"name": "A", "metrics": {"design_score": 0.6}}]},
    ]

    base = run_trajectory_analysis(runs)
    enabled = run_trajectory_analysis(runs, enable_parity_coherence=True)

    assert "parity_coherence" not in base
    assert "parity_coherence" in enabled
    assert set(enabled["parity_coherence"].keys()) == REQUIRED_KEYS


def test_integration_metric_and_aggregation_configurable() -> None:
    runs = [
        {"strategies": [
            {"name": "A", "metrics": {"design_score": 0.1, "score": 0.9}},
            {"name": "B", "metrics": {"design_score": 0.9, "score": 0.1}},
        ]},
        {"strategies": [
            {"name": "A", "metrics": {"design_score": 0.2, "score": 0.8}},
            {"name": "B", "metrics": {"design_score": 0.8, "score": 0.2}},
        ]},
    ]
    by_mean_design = run_trajectory_analysis(
        runs,
        enable_parity_coherence=True,
        parity_metric_key="design_score",
        parity_aggregation="mean",
    )
    by_max_score = run_trajectory_analysis(
        runs,
        enable_parity_coherence=True,
        parity_metric_key="score",
        parity_aggregation="max",
    )

    assert by_mean_design["parity_coherence"] != by_max_score["parity_coherence"]
