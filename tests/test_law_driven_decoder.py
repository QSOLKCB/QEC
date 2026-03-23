"""Tests for the deterministic Law-Driven Decoder Engine (v97.8.0)."""

import numpy as np
import pytest

from qec.analysis.law_promotion import Condition, Law
from qec.decoder.law_driven_decoder import (
    ACTION_MAP,
    DecoderStrategy,
    adjust_damping,
    aggregate_actions,
    build_strategy,
    correction_mode,
    detect_oscillation,
    evaluate_run,
    extract_metrics,
    freeze_nodes,
    get_applicable_laws,
    make_state,
    map_law_to_action,
    resolve_conflicts,
    reweight_messages,
    run_decoder,
    schedule_update,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_law(
    law_id: str,
    action: str,
    conditions: list | None = None,
    confidence: float = 0.5,
    law_score: float = 0.5,
) -> Law:
    """Create a minimal Law for testing."""
    if conditions is None:
        conditions = []
    return Law(
        law_id=law_id,
        version=1,
        conditions=conditions,
        action=action,
        evidence=["test_run"],
        scores={"confidence": confidence, "law_score": law_score},
        created_at=0.0,
    )


# ---------------------------------------------------------------------------
# STEP 1 — ACTION PRIMITIVES
# ---------------------------------------------------------------------------


class TestActionPrimitives:
    def test_adjust_damping_scales_values(self):
        state = make_state([2.0, 4.0, 6.0])
        result = adjust_damping(state, alpha=0.5)
        np.testing.assert_array_almost_equal(result["values"], [1.0, 2.0, 3.0])

    def test_adjust_damping_does_not_mutate_input(self):
        state = make_state([2.0, 4.0])
        original = state["values"].copy()
        adjust_damping(state, alpha=0.5)
        np.testing.assert_array_equal(state["values"], original)

    def test_reweight_messages(self):
        state = make_state([1.0, 2.0, 3.0])
        result = reweight_messages(state, weight=2.0)
        np.testing.assert_array_almost_equal(result["values"], [2.0, 4.0, 6.0])

    def test_freeze_nodes_zeros_masked(self):
        state = make_state([1.0, 2.0, 3.0, 4.0])
        result = freeze_nodes(state, mask=[True, False, True, False])
        np.testing.assert_array_almost_equal(result["values"], [0.0, 2.0, 0.0, 4.0])

    def test_schedule_update_sequential(self):
        state = make_state([4.0, 2.0, 6.0])
        result = schedule_update(state, mode="sequential")
        # cumulative mean: [4/1, 6/2, 12/3] = [4, 3, 4]
        np.testing.assert_array_almost_equal(result["values"], [4.0, 3.0, 4.0])

    def test_schedule_update_parallel_preserves(self):
        state = make_state([1.0, 2.0, 3.0])
        result = schedule_update(state, mode="parallel")
        np.testing.assert_array_almost_equal(result["values"], [1.0, 2.0, 3.0])

    def test_correction_mode_hard(self):
        state = make_state([-2.0, 0.0, 3.0])
        result = correction_mode(state, mode="hard")
        np.testing.assert_array_almost_equal(result["values"], [-1.0, 0.0, 1.0])

    def test_correction_mode_soft(self):
        state = make_state([0.0])
        result = correction_mode(state, mode="soft")
        np.testing.assert_array_almost_equal(result["values"], [0.0])

    def test_correction_mode_clamp(self):
        state = make_state([-5.0, 0.5, 5.0])
        result = correction_mode(state, mode="clamp")
        np.testing.assert_array_almost_equal(result["values"], [-1.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# STEP 2 — LAW -> ACTION MAPPING
# ---------------------------------------------------------------------------


class TestLawToActionMapping:
    def test_known_action_maps(self):
        law = _make_law("L1", "reduce_oscillation")
        result = map_law_to_action(law)
        assert result is not None
        assert result[0] == "adjust_damping"
        assert result[1] == {"alpha": 0.5}

    def test_stabilize_maps(self):
        law = _make_law("L2", "stabilize")
        result = map_law_to_action(law)
        assert result == ("schedule_update", {"mode": "sequential"})

    def test_unknown_action_returns_none(self):
        law = _make_law("L3", "unknown_action")
        result = map_law_to_action(law)
        assert result is None

    def test_all_actions_in_map_are_dispatchable(self):
        """Every action in ACTION_MAP must map to a known action type."""
        from qec.decoder.law_driven_decoder import _ACTION_DISPATCH

        for action_str, (action_type, _params) in ACTION_MAP.items():
            assert action_type in _ACTION_DISPATCH, (
                f"ACTION_MAP entry {action_str!r} -> {action_type!r} not in dispatch"
            )


# ---------------------------------------------------------------------------
# STEP 3 — APPLICABLE LAWS
# ---------------------------------------------------------------------------


class TestApplicableLaws:
    def test_all_conditions_met(self):
        cond = Condition("variance", "gt", 0.5)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        result = get_applicable_laws([law], {"variance": 1.0})
        assert len(result) == 1
        assert result[0].id == "L1"

    def test_condition_not_met(self):
        cond = Condition("variance", "lt", 0.5)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        result = get_applicable_laws([law], {"variance": 1.0})
        assert len(result) == 0

    def test_missing_metric_not_applicable(self):
        cond = Condition("variance", "gt", 0.5)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        result = get_applicable_laws([law], {"mean": 1.0})
        assert len(result) == 0

    def test_multiple_laws_mixed(self):
        c1 = Condition("variance", "gt", 0.5)
        c2 = Condition("mean", "lt", 0.0)
        law1 = _make_law("L1", "reduce_oscillation", conditions=[c1])
        law2 = _make_law("L2", "stabilize", conditions=[c2])
        result = get_applicable_laws([law1, law2], {"variance": 1.0, "mean": 0.5})
        assert len(result) == 1
        assert result[0].id == "L1"


# ---------------------------------------------------------------------------
# STEP 4 — ACTION AGGREGATION
# ---------------------------------------------------------------------------


class TestActionAggregation:
    def test_groups_by_action_type(self):
        law1 = _make_law("L1", "reduce_oscillation")
        law2 = _make_law("L2", "increase_damping")
        groups = aggregate_actions([law1, law2])
        assert "adjust_damping" in groups
        assert len(groups["adjust_damping"]) == 2

    def test_skips_unmapped_actions(self):
        law = _make_law("L1", "unknown_thing")
        groups = aggregate_actions([law])
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# STEP 5 — CONFLICT RESOLUTION
# ---------------------------------------------------------------------------


class TestConflictResolution:
    def test_highest_confidence_wins(self):
        law1 = _make_law("L1", "reduce_oscillation", confidence=0.3)
        law2 = _make_law("L2", "increase_damping", confidence=0.9)
        groups = aggregate_actions([law1, law2])
        resolved = resolve_conflicts(groups)
        assert resolved["adjust_damping"]["alpha"] == 0.3  # law2 maps to alpha=0.3

    def test_specificity_tiebreaker(self):
        c1 = Condition("variance", "gt", 0.5)
        c2 = Condition("mean", "lt", 1.0)
        law1 = _make_law("L1", "reduce_oscillation", conditions=[c1], confidence=0.5)
        law2 = _make_law("L2", "increase_damping", conditions=[c1, c2], confidence=0.5)
        groups = aggregate_actions([law1, law2])
        resolved = resolve_conflicts(groups)
        # law2 has more conditions (specificity=2 vs 1), so wins
        assert resolved["adjust_damping"]["alpha"] == 0.3

    def test_lexicographic_final_tiebreaker(self):
        law1 = _make_law("B_law", "reduce_oscillation", confidence=0.5)
        law2 = _make_law("A_law", "increase_damping", confidence=0.5)
        groups = aggregate_actions([law1, law2])
        resolved = resolve_conflicts(groups)
        # Same confidence, same specificity (0), A_law < B_law lexicographically
        assert resolved["adjust_damping"]["alpha"] == 0.3  # A_law -> increase_damping


# ---------------------------------------------------------------------------
# STEP 6 — STRATEGY OBJECT
# ---------------------------------------------------------------------------


class TestDecoderStrategy:
    def test_apply_single_action(self):
        strategy = DecoderStrategy({"adjust_damping": {"alpha": 0.5}})
        state = make_state([2.0, 4.0])
        result = strategy.apply(state)
        np.testing.assert_array_almost_equal(result["values"], [1.0, 2.0])

    def test_apply_multiple_actions_deterministic_order(self):
        strategy = DecoderStrategy({
            "adjust_damping": {"alpha": 2.0},
            "correction_mode": {"mode": "clamp"},
        })
        state = make_state([0.3, -0.8])
        result = strategy.apply(state)
        # adjust_damping first (a < c): [0.6, -1.6]
        # then correction_mode clamp: [0.6, -1.0]
        np.testing.assert_array_almost_equal(result["values"], [0.6, -1.0])

    def test_to_dict_serializable(self):
        strategy = DecoderStrategy({"adjust_damping": {"alpha": 0.5}})
        d = strategy.to_dict()
        assert "actions" in d
        assert "adjust_damping" in d["actions"]

    def test_empty_strategy_no_change(self):
        strategy = DecoderStrategy({})
        state = make_state([1.0, 2.0])
        result = strategy.apply(state)
        np.testing.assert_array_almost_equal(result["values"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# STEP 9 — METRIC EXTRACTION
# ---------------------------------------------------------------------------


class TestMetricExtraction:
    def test_basic_metrics(self):
        state = make_state([1.0, 3.0])
        metrics = extract_metrics(state)
        assert metrics["mean"] == pytest.approx(2.0)
        assert metrics["variance"] == pytest.approx(1.0)
        assert metrics["delta"] == 0.0

    def test_delta_with_prev_state(self):
        prev = make_state([0.0, 0.0])
        curr = make_state([3.0, 4.0])
        metrics = extract_metrics(curr, prev)
        assert metrics["delta"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# STEP 10 — EVALUATION
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_convergence_detected(self):
        trajectory = [make_state([0.0, 0.0])]
        result = evaluate_run(trajectory)
        assert result["converged"] is True
        assert result["steps_to_convergence"] == 0

    def test_no_convergence(self):
        trajectory = [make_state([1.0, -1.0, 2.0])]
        result = evaluate_run(trajectory)
        assert result["converged"] is False

    def test_oscillation_detection(self):
        # Create alternating variance pattern
        trajectory = []
        for i in range(6):
            if i % 2 == 0:
                trajectory.append(make_state([1.0, -1.0]))  # var=1
            else:
                trajectory.append(make_state([0.5, -0.5]))  # var=0.25
        assert detect_oscillation(trajectory, window=5) is True

    def test_empty_trajectory(self):
        result = evaluate_run([])
        assert result["converged"] is False
        assert result["final_variance"] == float("inf")


# ---------------------------------------------------------------------------
# STEP 8 — DECODER LOOP
# ---------------------------------------------------------------------------


class TestDecoderLoop:
    def test_basic_run(self):
        cond = Condition("variance", "gt", 0.001)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond], confidence=0.8)
        state = make_state([10.0, -10.0, 5.0, -5.0])
        result = run_decoder([law], state, max_steps=20)
        assert "final_state" in result
        assert "trajectory" in result
        assert "strategies" in result
        assert "evaluation" in result
        # Damping should reduce variance over steps
        initial_var = float(np.var(state["values"]))
        final_var = float(np.var(result["final_state"]["values"]))
        assert final_var < initial_var

    def test_no_applicable_laws_stops_immediately(self):
        cond = Condition("variance", "lt", -1.0)  # never true
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        state = make_state([1.0, 2.0])
        result = run_decoder([law], state, max_steps=10)
        # Should stop at step 0 (no actions)
        assert len(result["trajectory"]) == 1

    def test_deterministic_repeated_runs(self):
        cond = Condition("variance", "gt", 0.001)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        state = make_state([5.0, -3.0, 7.0])
        r1 = run_decoder([law], state, max_steps=10)
        r2 = run_decoder([law], state, max_steps=10)
        np.testing.assert_array_equal(
            r1["final_state"]["values"], r2["final_state"]["values"]
        )
        assert len(r1["trajectory"]) == len(r2["trajectory"])

    def test_convergence_stops_loop(self):
        cond = Condition("variance", "gt", 0.0)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        state = make_state([0.001, -0.001])
        result = run_decoder([law], state, max_steps=100)
        # Should converge quickly, not run all 100 steps
        assert len(result["trajectory"]) < 100

    def test_multiple_laws_compose(self):
        c1 = Condition("variance", "gt", 0.01)
        c2 = Condition("mean", "gt", -100.0)
        law1 = _make_law("L1", "reduce_oscillation", conditions=[c1], confidence=0.9)
        law2 = _make_law("L2", "soft_correction", conditions=[c2], confidence=0.8)
        state = make_state([10.0, -10.0, 5.0])
        result = run_decoder([law1, law2], state, max_steps=10)
        # Both actions should have been applied
        assert len(result["strategies"]) > 0
        first_strat = result["strategies"][0]
        assert "adjust_damping" in first_strat["actions"]
        assert "correction_mode" in first_strat["actions"]

    def test_build_strategy_integration(self):
        cond = Condition("variance", "gt", 0.5)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        metrics = {"variance": 1.0, "mean": 0.0, "delta": 0.0}
        strategy = build_strategy([law], metrics)
        assert "adjust_damping" in strategy.actions


# ---------------------------------------------------------------------------
# DETERMINISM TESTS
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_strategy_action_order_is_deterministic(self):
        """Actions are applied in sorted order regardless of insertion."""
        s1 = DecoderStrategy({"correction_mode": {"mode": "hard"}, "adjust_damping": {"alpha": 0.5}})
        s2 = DecoderStrategy({"adjust_damping": {"alpha": 0.5}, "correction_mode": {"mode": "hard"}})
        state = make_state([2.0, -3.0])
        r1 = s1.apply(state)
        r2 = s2.apply(state)
        np.testing.assert_array_equal(r1["values"], r2["values"])

    def test_resolve_conflicts_is_deterministic(self):
        """Same inputs always produce same resolution."""
        law1 = _make_law("L1", "reduce_oscillation", confidence=0.5)
        law2 = _make_law("L2", "increase_damping", confidence=0.5)
        for _ in range(10):
            groups = aggregate_actions([law1, law2])
            resolved = resolve_conflicts(groups)
            # L1 < L2 lexicographically, so L1 wins -> reduce_oscillation -> alpha=0.5
            assert resolved["adjust_damping"]["alpha"] == 0.5

    def test_full_run_bitwise_deterministic(self):
        """Two identical runs produce byte-identical outputs."""
        cond = Condition("variance", "gt", 0.001)
        law = _make_law("L1", "reduce_oscillation", conditions=[cond])
        state = make_state([8.0, -4.0, 2.0, -1.0])
        r1 = run_decoder([law], state, max_steps=30)
        r2 = run_decoder([law], state, max_steps=30)
        assert len(r1["trajectory"]) == len(r2["trajectory"])
        for s1, s2 in zip(r1["trajectory"], r2["trajectory"]):
            np.testing.assert_array_equal(s1["values"], s2["values"])
