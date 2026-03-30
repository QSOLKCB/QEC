"""Tests for src/qec/analysis/control_flow.py — v105.4.0."""

from qec.analysis.control_flow import (
    compute_damping_factor,
    compute_step_aggressiveness,
    compute_strategy_escalation,
    run_control_flow,
)


def _make_collapse_result(
    collapse_score=0.1,
    acceleration_peak=0.0,
    basin_switch_prediction=False,
    failure_risk=0.0,
    singularity_events=None,
):
    return {
        "failure_risk": failure_risk,
        "collapse_score": collapse_score,
        "acceleration_peak": acceleration_peak,
        "basin_switch_prediction": basin_switch_prediction,
        "singularity_events": singularity_events if singularity_events is not None else [],
    }


class TestComputeDampingFactor:
    def test_stable(self):
        assert compute_damping_factor(0.1) == 1.0

    def test_boundary_low(self):
        assert compute_damping_factor(0.3) == 1.0

    def test_moderate(self):
        assert compute_damping_factor(0.5) == 0.8

    def test_boundary_mid(self):
        assert compute_damping_factor(0.6) == 0.8

    def test_high(self):
        assert compute_damping_factor(0.9) == 0.6


class TestComputeStepAggressiveness:
    def test_zero_density(self):
        assert compute_step_aggressiveness(0.0) == 1.0

    def test_moderate_density(self):
        assert compute_step_aggressiveness(0.3) == 0.7

    def test_clamp_at_half(self):
        assert compute_step_aggressiveness(0.8) == 0.5

    def test_clamp_high(self):
        assert compute_step_aggressiveness(1.5) == 0.5


class TestComputeStrategyEscalation:
    def test_hold(self):
        assert compute_strategy_escalation(False) == "hold"

    def test_escalate(self):
        assert compute_strategy_escalation(True) == "escalate"


class TestRunControlFlow:
    def test_stable(self):
        result = run_control_flow(_make_collapse_result(
            collapse_score=0.1,
            acceleration_peak=0.0,
            basin_switch_prediction=False,
        ))
        assert result["damping_factor"] == 1.0
        assert result["step_aggressiveness"] == 1.0
        assert result["strategy_action"] == "hold"
        assert result["control_stability_score"] == 1.0

    def test_moderate_stress(self):
        result = run_control_flow(_make_collapse_result(
            collapse_score=0.5,
            acceleration_peak=0.2,
            basin_switch_prediction=False,
        ))
        assert result["damping_factor"] == 0.8
        assert result["step_aggressiveness"] == 0.8
        assert result["strategy_action"] == "hold"
        assert result["control_stability_score"] == 0.8

    def test_high_risk(self):
        result = run_control_flow(_make_collapse_result(
            collapse_score=0.9,
            acceleration_peak=0.4,
            basin_switch_prediction=True,
        ))
        assert result["damping_factor"] == 0.6
        assert result["step_aggressiveness"] == 0.6
        assert result["strategy_action"] == "escalate"
        assert result["control_stability_score"] == 0.6

    def test_determinism(self):
        cr = _make_collapse_result(
            collapse_score=0.45,
            acceleration_peak=0.15,
            basin_switch_prediction=True,
        )
        r1 = run_control_flow(cr)
        r2 = run_control_flow(cr)
        assert r1 == r2
