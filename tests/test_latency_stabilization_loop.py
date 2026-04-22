from __future__ import annotations

import math

import pytest

from qec.analysis.latency_stabilization_loop import (
    LATENCY_STABILIZATION_LOOP_VERSION,
    LatencyNodeSignal,
    LatencyPolicy,
    run_latency_stabilization_loop,
)


def _signal(
    node_id: str,
    *,
    latency_ms: float = 10.0,
    target_latency_ms: float = 9.0,
    max_acceptable_latency_ms: float = 20.0,
    latency_delta_ms: float = 0.2,
    jitter_ms: float = 0.3,
    utilization: float = 0.4,
) -> LatencyNodeSignal:
    return LatencyNodeSignal(
        node_id=node_id,
        latency_ms=latency_ms,
        target_latency_ms=target_latency_ms,
        max_acceptable_latency_ms=max_acceptable_latency_ms,
        latency_delta_ms=latency_delta_ms,
        jitter_ms=jitter_ms,
        utilization=utilization,
    )


def _policy(**overrides: float) -> LatencyPolicy:
    base = {
        "jitter_weight": 0.4,
        "drift_weight": 0.3,
        "utilization_weight": 0.2,
        "max_correction_strength": 1.0,
    }
    base.update(overrides)
    return LatencyPolicy(**base)


def test_deterministic_replay_identical_json_and_hash() -> None:
    signals = (_signal("n1"), _signal("n2", latency_ms=12.0, jitter_ms=0.8))
    policy = _policy()
    first = run_latency_stabilization_loop(signals, policy)
    second = run_latency_stabilization_loop(signals, policy)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_ordering_shuffled_input_identical_output() -> None:
    policy = _policy()
    ordered = run_latency_stabilization_loop((_signal("a"), _signal("b"), _signal("c")), policy)
    shuffled = run_latency_stabilization_loop((_signal("c"), _signal("a"), _signal("b")), policy)

    assert ordered.to_canonical_json() == shuffled.to_canonical_json()
    assert tuple(d.node_id for d in ordered.node_decisions) == ("a", "b", "c")


def test_bounded_outputs_within_unit_interval() -> None:
    receipt = run_latency_stabilization_loop(
        (
            _signal("n1", latency_delta_ms=1.0, jitter_ms=5.0, utilization=1.0),
            _signal("n2", latency_delta_ms=-5.0, jitter_ms=0.0, utilization=0.0),
        ),
        _policy(),
    )

    for decision in receipt.node_decisions:
        assert 0.0 <= decision.instability_pressure <= 1.0
        assert 0.0 <= decision.correction_strength <= 1.0
        assert 0.0 <= decision.stability_score <= 1.0
    assert 0.0 <= receipt.mesh_instability_pressure <= 1.0
    assert 0.0 <= receipt.mesh_stability_score <= 1.0


def test_classification_low_is_stable_high_is_critical() -> None:
    policy = _policy(jitter_weight=1.0, drift_weight=1.0, utilization_weight=1.0)
    low = run_latency_stabilization_loop((_signal("n-low", utilization=0.0, jitter_ms=0.0, latency_delta_ms=0.0),), policy)
    high = run_latency_stabilization_loop(
        (
            _signal(
                "n-high",
                latency_ms=100.0,
                target_latency_ms=1.0,
                max_acceptable_latency_ms=2.0,
                jitter_ms=100.0,
                latency_delta_ms=100.0,
                utilization=1.0,
            ),
        ),
        policy,
    )

    assert low.node_decisions[0].action_label == "stable"
    assert high.node_decisions[0].action_label == "critical"


def test_validation_nan_inf_and_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        _signal("n1", latency_ms=math.inf)
    with pytest.raises(ValueError, match="finite"):
        _signal("n1", latency_ms=math.nan)
    with pytest.raises(ValueError, match="max_acceptable_latency_ms must be > target_latency_ms"):
        _signal("n1", target_latency_ms=10.0, max_acceptable_latency_ms=10.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        _policy(jitter_weight=-0.1)
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        _policy(max_correction_strength=1.2)


def test_stability_score_is_one_minus_instability_pressure() -> None:
    receipt = run_latency_stabilization_loop((_signal("n1", latency_ms=13.0, jitter_ms=1.2),), _policy())
    decision = receipt.node_decisions[0]
    assert decision.stability_score == pytest.approx(1.0 - decision.instability_pressure)


def test_hash_stability_repeated_runs_identical() -> None:
    signals = (_signal("n1"), _signal("n2", jitter_ms=1.0), _signal("n3", utilization=0.9))
    policy = _policy()
    hashes = [run_latency_stabilization_loop(signals, policy).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1


def test_duplicate_node_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        run_latency_stabilization_loop((_signal("dup"), _signal("dup")), _policy())


def test_extreme_jitter_clamps_and_full_correction_strength() -> None:
    receipt = run_latency_stabilization_loop(
        (
            _signal(
                "n1",
                target_latency_ms=0.5,
                max_acceptable_latency_ms=1.0,
                jitter_ms=1.0e9,
                utilization=1.0,
                latency_delta_ms=1.0e9,
            ),
        ),
        _policy(jitter_weight=1.0, drift_weight=1.0, utilization_weight=1.0, max_correction_strength=1.0),
    )
    decision = receipt.node_decisions[0]

    assert decision.instability_pressure == 1.0
    assert decision.correction_strength == 1.0
    assert decision.action_label == "critical"


def test_empty_input_mesh_defaults_and_constants() -> None:
    receipt = run_latency_stabilization_loop((), _policy())
    assert receipt.version == LATENCY_STABILIZATION_LOOP_VERSION
    assert receipt.mesh_instability_pressure == 0.0
    assert receipt.mesh_stability_score == 1.0
    assert receipt.instability_count == 0
    assert receipt.control_mode == "latency_advisory"
    assert receipt.observatory_only is True
