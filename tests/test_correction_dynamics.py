"""Tests for correction hysteresis and friction analysis (v96.3.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.correction_dynamics import (
    classify_dynamics,
    compute_friction_score,
    compute_mode_switches,
    compute_projection_churn,
    detect_hysteresis,
    detect_oscillation,
    extract_correction_traces,
    invariant_conflict_dissipation,
    print_dynamics_report,
    run_correction_dynamics,
    switching_instability_score,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_app(
    dfa_type="chain",
    n=10,
    before_mode="none",
    after_mode="d4",
    improved=True,
    before_stab=0.2,
    after_stab=0.5,
    before_comp=0.1,
    after_comp=0.3,
):
    """Build a single application record for testing."""
    return {
        "dfa_type": dfa_type,
        "n": n,
        "before_mode": before_mode,
        "after_mode": after_mode,
        "improved": improved,
        "before_metrics": {
            "stability_efficiency": before_stab,
            "compression_efficiency": before_comp,
        },
        "after_metrics": {
            "stability_efficiency": after_stab,
            "compression_efficiency": after_comp,
        },
    }


def _make_data(apps):
    """Wrap applications in the expected data format."""
    return {"applications": apps}


# ---------------------------------------------------------------------------
# PART 1 — extract_correction_traces
# ---------------------------------------------------------------------------


class TestExtractCorrectionTraces:
    def test_empty_input(self):
        assert extract_correction_traces({}) == []

    def test_non_dict_input(self):
        assert extract_correction_traces(None) == []
        assert extract_correction_traces([]) == []

    def test_no_applications(self):
        assert extract_correction_traces({"applications": []}) == []

    def test_single_application(self):
        data = _make_data([_make_app()])
        traces = extract_correction_traces(data)
        assert len(traces) == 1
        assert traces[0]["dfa_type"] == "chain"
        assert traces[0]["n"] == 10
        assert traces[0]["states_before"] == ["none"]
        assert traces[0]["states_after"] == ["d4"]
        assert traces[0]["modes"] == ["d4"]
        assert len(traces[0]["projection_distances"]) == 1

    def test_groups_by_system(self):
        apps = [
            _make_app(dfa_type="chain", n=5),
            _make_app(dfa_type="chain", n=5, before_mode="d4", after_mode="square"),
            _make_app(dfa_type="cycle", n=10),
        ]
        data = _make_data(apps)
        traces = extract_correction_traces(data)
        assert len(traces) == 2
        assert traces[0]["dfa_type"] == "chain"
        assert len(traces[0]["states_before"]) == 2
        assert traces[1]["dfa_type"] == "cycle"

    def test_no_mutation(self):
        data = _make_data([_make_app()])
        original = copy.deepcopy(data)
        extract_correction_traces(data)
        assert data == original

    def test_deterministic(self):
        data = _make_data([
            _make_app(dfa_type="chain", n=5),
            _make_app(dfa_type="cycle", n=10),
        ])
        r1 = extract_correction_traces(data)
        r2 = extract_correction_traces(data)
        assert r1 == r2

    def test_projection_distance(self):
        app = _make_app(before_stab=0.2, after_stab=0.5, before_comp=0.1, after_comp=0.4)
        traces = extract_correction_traces(_make_data([app]))
        # dist = |0.5-0.2| + |0.4-0.1| = 0.3 + 0.3 = 0.6
        assert abs(traces[0]["projection_distances"][0] - 0.6) < 1e-10


# ---------------------------------------------------------------------------
# PART 2 — Hysteresis Detection
# ---------------------------------------------------------------------------


class TestDetectOscillation:
    def test_empty(self):
        result = detect_oscillation([], [])
        assert result["oscillation_count"] == 0
        assert result["oscillation_ratio"] == 0.0

    def test_no_oscillation(self):
        result = detect_oscillation(["a", "b"], ["b", "c"])
        assert result["oscillation_count"] == 0

    def test_single_oscillation(self):
        # a->b then b->a
        result = detect_oscillation(["a", "b"], ["b", "a"])
        assert result["oscillation_count"] == 1
        assert result["oscillation_ratio"] == 1.0

    def test_multiple_oscillations(self):
        result = detect_oscillation(
            ["a", "b", "a", "b"],
            ["b", "a", "b", "a"],
        )
        assert result["oscillation_count"] == 3
        assert abs(result["oscillation_ratio"] - 1.0) < 1e-10

    def test_partial_oscillation(self):
        result = detect_oscillation(
            ["a", "b", "c"],
            ["b", "a", "d"],
        )
        # Only first pair oscillates: a->b, b->a
        assert result["oscillation_count"] == 1
        assert abs(result["oscillation_ratio"] - 0.5) < 1e-10


class TestDetectHysteresis:
    def test_empty(self):
        result = detect_hysteresis([], [])
        assert result["hysteresis_events"] == 0

    def test_no_hysteresis(self):
        result = detect_hysteresis(["a", "a"], ["b", "b"])
        assert result["hysteresis_events"] == 0

    def test_single_hysteresis(self):
        # Same input "a" -> different outputs "b" and "c"
        result = detect_hysteresis(["a", "a"], ["b", "c"])
        assert result["hysteresis_events"] == 1
        assert result["hysteresis_ratio"] == 1.0

    def test_multiple_inputs(self):
        result = detect_hysteresis(
            ["a", "a", "b", "b"],
            ["x", "y", "z", "z"],
        )
        # "a" maps to {x, y} (hysteresis), "b" maps to {z} (no hysteresis)
        assert result["hysteresis_events"] == 1
        assert abs(result["hysteresis_ratio"] - 0.5) < 1e-10

    def test_all_hysteresis(self):
        result = detect_hysteresis(
            ["a", "a", "b", "b"],
            ["x", "y", "z", "w"],
        )
        assert result["hysteresis_events"] == 2
        assert result["hysteresis_ratio"] == 1.0


# ---------------------------------------------------------------------------
# PART 3 — Mode Switch Friction
# ---------------------------------------------------------------------------


class TestComputeModeSwitches:
    def test_empty(self):
        result = compute_mode_switches([])
        assert result["switch_count"] == 0

    def test_single_mode(self):
        result = compute_mode_switches(["a"])
        assert result["switch_count"] == 0

    def test_no_switches(self):
        result = compute_mode_switches(["a", "a", "a"])
        assert result["switch_count"] == 0
        assert result["switch_ratio"] == 0.0

    def test_all_switches(self):
        result = compute_mode_switches(["a", "b", "a"])
        assert result["switch_count"] == 2
        assert result["switch_ratio"] == 1.0

    def test_partial_switches(self):
        result = compute_mode_switches(["a", "a", "b", "b"])
        assert result["switch_count"] == 1
        assert abs(result["switch_ratio"] - 1.0 / 3.0) < 1e-10


class TestSwitchingInstabilityScore:
    def test_empty(self):
        assert switching_instability_score([]) == 0.0

    def test_short_sequence(self):
        assert switching_instability_score(["a", "b"]) == 0.0

    def test_no_instability(self):
        assert switching_instability_score(["a", "b", "c"]) == 0.0

    def test_full_instability(self):
        # a->b->a pattern
        score = switching_instability_score(["a", "b", "a"])
        assert score == 1.0

    def test_repeated_reversals(self):
        score = switching_instability_score(["a", "b", "a", "b", "a"])
        # Reversals at positions 0,1,2 and 1,2,3 and 2,3,4 = 3 out of 3
        assert score == 1.0

    def test_partial_instability(self):
        score = switching_instability_score(["a", "b", "a", "c"])
        # Reversal at 0: a,b,a=yes. At 1: b,a,c=no. -> 1/2
        assert abs(score - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# PART 4 — Projection Churn
# ---------------------------------------------------------------------------


class TestComputeProjectionChurn:
    def test_empty(self):
        result = compute_projection_churn([], 0.5)
        assert result["total_projection"] == 0.0
        assert result["churn_score"] == 0.0

    def test_high_stability(self):
        result = compute_projection_churn([1.0, 1.0], 1.0)
        assert result["total_projection"] == 2.0
        assert result["churn_score"] == 0.0  # No churn when stable

    def test_low_stability(self):
        result = compute_projection_churn([1.0, 1.0], 0.0)
        assert result["total_projection"] == 2.0
        assert result["churn_score"] == 2.0

    def test_partial_stability(self):
        result = compute_projection_churn([0.5, 0.5], 0.5)
        assert abs(result["total_projection"] - 1.0) < 1e-10
        assert abs(result["churn_score"] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# PART 5 — Invariant Conflict Dissipation
# ---------------------------------------------------------------------------


class TestInvariantConflictDissipation:
    def test_empty_invariants(self):
        result = invariant_conflict_dissipation([], [])
        assert result["conflict_count"] == 0
        assert result["conflict_penalty"] == 0.0

    def test_no_conflicts(self):
        interactions = [
            {"pair": ("a", "b"), "type": "synergy", "evidence_count": 1}
        ]
        result = invariant_conflict_dissipation(["a", "b"], interactions)
        assert result["conflict_count"] == 0

    def test_single_conflict(self):
        interactions = [
            {"pair": ("a", "b"), "type": "conflict", "evidence_count": 1}
        ]
        result = invariant_conflict_dissipation(["a", "b"], interactions)
        assert result["conflict_count"] == 1
        assert abs(result["conflict_penalty"] - 0.5) < 1e-10

    def test_conflict_not_in_invariants(self):
        interactions = [
            {"pair": ("a", "b"), "type": "conflict", "evidence_count": 1}
        ]
        result = invariant_conflict_dissipation(["a", "c"], interactions)
        assert result["conflict_count"] == 0

    def test_multiple_evidence(self):
        interactions = [
            {"pair": ("a", "b"), "type": "conflict", "evidence_count": 3}
        ]
        result = invariant_conflict_dissipation(["a", "b"], interactions)
        assert result["conflict_count"] == 1
        # penalty = 0.5 + (3-1)*0.1 = 0.7
        assert abs(result["conflict_penalty"] - 0.7) < 1e-10


# ---------------------------------------------------------------------------
# PART 6 — Friction Score
# ---------------------------------------------------------------------------


class TestComputeFrictionScore:
    def test_stable_trace(self):
        trace = {
            "states_before": ["a", "a"],
            "states_after": ["b", "b"],
            "modes": ["b", "b"],
            "projection_distances": [0.1, 0.1],
            "stability_efficiency": 0.9,
        }
        result = compute_friction_score(trace)
        assert result["friction_score"] < 1.0
        assert "components" in result

    def test_frictional_trace(self):
        trace = {
            "states_before": ["a", "b", "a", "b"],
            "states_after": ["b", "a", "b", "a"],
            "modes": ["b", "a", "b", "a"],
            "projection_distances": [2.0, 2.0, 2.0, 2.0],
            "stability_efficiency": 0.0,
        }
        result = compute_friction_score(trace)
        assert result["friction_score"] > 2.5

    def test_components_present(self):
        trace = {
            "states_before": ["a"],
            "states_after": ["b"],
            "modes": ["b"],
            "projection_distances": [0.1],
            "stability_efficiency": 0.5,
        }
        result = compute_friction_score(trace)
        comps = result["components"]
        assert "oscillation" in comps
        assert "hysteresis" in comps
        assert "switching" in comps
        assert "churn" in comps
        assert "conflict" in comps

    def test_no_mutation(self):
        trace = {
            "states_before": ["a", "b"],
            "states_after": ["b", "a"],
            "modes": ["b", "a"],
            "projection_distances": [0.5, 0.5],
            "stability_efficiency": 0.5,
        }
        original = copy.deepcopy(trace)
        compute_friction_score(trace)
        assert trace == original

    def test_deterministic(self):
        trace = {
            "states_before": ["a", "b"],
            "states_after": ["b", "a"],
            "modes": ["b", "a"],
            "projection_distances": [0.5, 0.5],
            "stability_efficiency": 0.5,
        }
        r1 = compute_friction_score(trace)
        r2 = compute_friction_score(trace)
        assert r1 == r2

    def test_with_conflict_data(self):
        trace = {
            "states_before": ["a"],
            "states_after": ["b"],
            "modes": ["b"],
            "projection_distances": [0.1],
            "stability_efficiency": 0.5,
        }
        interactions = [
            {"pair": ("inv_a", "inv_b"), "type": "conflict", "evidence_count": 1}
        ]
        r_no = compute_friction_score(trace)
        r_yes = compute_friction_score(trace, interactions, ["inv_a", "inv_b"])
        assert r_yes["friction_score"] > r_no["friction_score"]


# ---------------------------------------------------------------------------
# PART 7 — Classification
# ---------------------------------------------------------------------------


class TestClassifyDynamics:
    def test_stable(self):
        assert classify_dynamics(0.0) == "stable"
        assert classify_dynamics(0.5) == "stable"
        assert classify_dynamics(0.99) == "stable"

    def test_adaptive(self):
        assert classify_dynamics(1.0) == "adaptive"
        assert classify_dynamics(1.5) == "adaptive"
        assert classify_dynamics(2.5) == "adaptive"

    def test_frictional(self):
        assert classify_dynamics(2.51) == "frictional"
        assert classify_dynamics(5.0) == "frictional"


# ---------------------------------------------------------------------------
# PART 8 — Full Pipeline
# ---------------------------------------------------------------------------


class TestRunCorrectionDynamics:
    def test_empty_input(self):
        result = run_correction_dynamics({})
        assert result == {"results": []}

    def test_single_system(self):
        data = _make_data([_make_app()])
        result = run_correction_dynamics(data)
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["dfa_type"] == "chain"
        assert r["n"] == 10
        assert "friction_score" in r
        assert "regime" in r
        assert "components" in r

    def test_multiple_systems(self):
        data = _make_data([
            _make_app(dfa_type="chain", n=5),
            _make_app(dfa_type="cycle", n=10),
        ])
        result = run_correction_dynamics(data)
        assert len(result["results"]) == 2

    def test_deterministic(self):
        data = _make_data([
            _make_app(dfa_type="chain", n=5),
            _make_app(dfa_type="cycle", n=10),
        ])
        r1 = run_correction_dynamics(data)
        r2 = run_correction_dynamics(data)
        assert r1 == r2

    def test_no_mutation(self):
        data = _make_data([_make_app()])
        original = copy.deepcopy(data)
        run_correction_dynamics(data)
        assert data == original


# ---------------------------------------------------------------------------
# PART 9 — Print Layer
# ---------------------------------------------------------------------------


class TestPrintDynamicsReport:
    def test_empty_report(self):
        text = print_dynamics_report({"results": []})
        assert "Correction Dynamics" in text

    def test_format(self):
        report = {
            "results": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "friction_score": 2.8,
                    "regime": "frictional",
                    "components": {
                        "oscillation": 0.6,
                        "hysteresis": 0.4,
                        "switching": 0.7,
                        "churn": 0.8,
                        "conflict": 0.3,
                    },
                }
            ]
        }
        text = print_dynamics_report(report)
        assert "DFA: cycle (n=10)" in text
        assert "friction_score: 2.8" in text
        assert "regime: frictional" in text
        assert "oscillation: 0.6" in text

    def test_no_n(self):
        report = {
            "results": [
                {
                    "dfa_type": "chain",
                    "n": None,
                    "friction_score": 0.5,
                    "regime": "stable",
                    "components": {},
                }
            ]
        }
        text = print_dynamics_report(report)
        assert "DFA: chain\n" in text
