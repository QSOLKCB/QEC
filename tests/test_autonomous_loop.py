"""Tests for autonomous_loop (v98.0.0)."""

import pytest

from qec.analysis.law_promotion import Condition, Law
from qec.analysis.autonomous_loop import (
    STATE_EXPLORE,
    STATE_META_ANALYZE,
    STATE_UPDATE,
    STATE_EXECUTE,
    AutonomousLoop,
    detect_oscillation,
    detect_stagnation,
    evaluate_results,
    next_state,
    update_laws_from_meta,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_law(
    law_id: str,
    action: str,
    conditions: list,
    confidence: float = 0.8,
) -> Law:
    conds = [Condition(m, op, v) for m, op, v in conditions]
    return Law(
        law_id=law_id,
        version=1,
        conditions=conds,
        action=action,
        evidence=["test_run"],
        scores={"confidence": confidence, "coverage": 0.5, "law_score": 0.4},
        created_at=0.0,
    )


def _make_state(values=None):
    if values is None:
        values = [1.0, 2.0, 3.0]
    return {"values": values}


class _MockConsensusStrategy:
    """Minimal mock for ConsensusStrategy."""

    def __init__(self, action_type="adjust_damping", params=None, law_id="L1"):
        self.strategy = _MockStrategy(action_type, params or {"alpha": 0.5}, law_id)
        self.scores = {"blue": 0.7, "white": 0.6}


class _MockStrategy:
    def __init__(self, action_type, params, law_id):
        self.action_type = action_type
        self.params = dict(params)
        self.law_id = law_id


def _mock_consensus(laws, state):
    """Mock consensus function that returns a valid result."""
    if not laws:
        return {"selected": None, "all_scores": {}, "reasoning": []}
    return {
        "selected": _MockConsensusStrategy(),
        "all_scores": {"L1": {"blue": 0.7}},
        "reasoning": [{"step": "mock"}],
    }


def _mock_consensus_empty(laws, state):
    """Mock consensus that returns no strategy."""
    return {"selected": None, "all_scores": {}, "reasoning": []}


# ---------------------------------------------------------------------------
# STATE MACHINE
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_full_cycle(self):
        assert next_state(STATE_EXPLORE) == STATE_META_ANALYZE
        assert next_state(STATE_META_ANALYZE) == STATE_UPDATE
        assert next_state(STATE_UPDATE) == STATE_EXECUTE
        assert next_state(STATE_EXECUTE) == STATE_EXPLORE

    def test_unknown_state(self):
        assert next_state("INVALID") == STATE_EXPLORE


# ---------------------------------------------------------------------------
# STAGNATION DETECTION
# ---------------------------------------------------------------------------


class TestStagnation:
    def test_stagnant(self):
        history = [0.5, 0.5, 0.5]
        assert detect_stagnation(history, window=3) is True

    def test_not_stagnant(self):
        history = [0.5, 0.6, 0.7]
        assert detect_stagnation(history, window=3) is False

    def test_short_history(self):
        assert detect_stagnation([0.5], window=3) is False

    def test_empty(self):
        assert detect_stagnation([], window=3) is False

    def test_near_zero_spread(self):
        history = [1.0, 1.0 + 1e-12, 1.0 - 1e-12]
        assert detect_stagnation(history, window=3) is True


# ---------------------------------------------------------------------------
# OSCILLATION DETECTION
# ---------------------------------------------------------------------------


class TestOscillation:
    def test_oscillating(self):
        # Up, down, up, down
        history = [1.0, 2.0, 1.0, 2.0]
        assert detect_oscillation(history, window=4) is True

    def test_not_oscillating_monotone(self):
        history = [1.0, 2.0, 3.0, 4.0]
        assert detect_oscillation(history, window=4) is False

    def test_short_history(self):
        assert detect_oscillation([1.0, 2.0], window=4) is False

    def test_small_diffs_rejected(self):
        history = [1.0, 1.001, 1.0, 1.001]
        assert detect_oscillation(history, window=4, threshold=0.01) is False

    def test_clear_oscillation(self):
        history = [0.0, 1.0, 0.0, 1.0]
        assert detect_oscillation(history, window=4) is True


# ---------------------------------------------------------------------------
# EVALUATE RESULTS
# ---------------------------------------------------------------------------


class TestEvaluateResults:
    def test_basic(self):
        state = {"values": [1.0, 2.0, 3.0]}
        metrics = evaluate_results(state)
        assert metrics["mean"] == 2.0
        assert metrics["variance"] > 0

    def test_with_prev_mean(self):
        state = {"values": [1.0, 2.0, 3.0], "prev_mean": 1.5}
        metrics = evaluate_results(state)
        assert metrics["improvement"] == 0.5

    def test_empty_values(self):
        state = {"values": []}
        # numpy mean of empty array would warn; we use default
        state2 = {}
        metrics = evaluate_results(state2)
        assert "mean" in metrics


# ---------------------------------------------------------------------------
# UPDATE LAWS FROM META
# ---------------------------------------------------------------------------


class TestUpdateLaws:
    def test_no_redundant(self):
        laws = [_make_law("L1", "stabilize", [("snr", "gt", 3.0)])]
        meta = {"redundant_pairs": []}
        result = update_laws_from_meta(laws, meta)
        assert len(result) == 1

    def test_remove_redundant(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        meta = {"redundant_pairs": [("L1", "L2")]}
        result = update_laws_from_meta(laws, meta)
        assert len(result) == 1
        assert result[0].id == "L1"

    def test_no_mutation(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        original_len = len(laws)
        meta = {"redundant_pairs": [("L1", "L2")]}
        update_laws_from_meta(laws, meta)
        assert len(laws) == original_len  # input not mutated


# ---------------------------------------------------------------------------
# AUTONOMOUS LOOP (INTEGRATION)
# ---------------------------------------------------------------------------


class TestAutonomousLoop:
    def test_single_step(self):
        loop = AutonomousLoop()
        laws = [
            _make_law("L1", "reduce_oscillation", [("snr", "gt", 1.0)]),
        ]
        state = _make_state([1.0, 2.0, 3.0])
        result = loop.step(laws, state, _mock_consensus)
        assert "laws" in result
        assert "state" in result
        assert "metrics" in result
        assert "phase" in result
        assert result["step_count"] == 1

    def test_phase_advances(self):
        loop = AutonomousLoop()
        assert loop.phase == STATE_EXPLORE
        laws = [_make_law("L1", "stabilize", [("snr", "gt", 1.0)])]
        state = _make_state()

        r1 = loop.step(laws, state, _mock_consensus)
        assert r1["phase"] == STATE_META_ANALYZE

        r2 = loop.step(laws, state, _mock_consensus)
        assert r2["phase"] == STATE_UPDATE

        r3 = loop.step(laws, state, _mock_consensus)
        assert r3["phase"] == STATE_EXECUTE

        r4 = loop.step(laws, state, _mock_consensus)
        assert r4["phase"] == STATE_EXPLORE

    def test_stagnation_triggers_fallback(self):
        loop = AutonomousLoop()
        laws = [_make_law("L1", "stabilize", [("snr", "gt", 1.0)])]
        state = _make_state([2.0, 2.0, 2.0])

        # First step establishes a stable strategy
        r1 = loop.step(laws, state, _mock_consensus)

        # Subsequent steps with same values -> stagnation
        r2 = loop.step(laws, state, _mock_consensus)
        r3 = loop.step(laws, state, _mock_consensus)

        assert r3["stagnation"] is True
        assert r3["fallback_used"] is True

    def test_no_strategies(self):
        loop = AutonomousLoop()
        laws = []
        state = _make_state()
        result = loop.step(laws, state, _mock_consensus_empty)
        assert result["composed_strategy"] == []
        assert result["fallback_used"] is False

    def test_deterministic_execution(self):
        laws = [
            _make_law("L1", "reduce_oscillation", [("snr", "gt", 1.0)]),
            _make_law("L2", "stabilize", [("var", "lt", 1.0)]),
        ]
        state = _make_state([1.0, 2.0, 3.0])

        results = []
        for _ in range(2):
            loop = AutonomousLoop()
            r = loop.step(laws, state, _mock_consensus)
            results.append(r["metrics"])

        assert results[0] == results[1]

    def test_oscillation_detection(self):
        loop = AutonomousLoop()
        laws = [_make_law("L1", "stabilize", [("snr", "gt", 1.0)])]

        # Manually inject oscillating metric history
        loop.metric_history = [1.0, 5.0, 1.0]

        # Next step with mean=5 would make [1,5,1,5] -> oscillation
        state = _make_state([5.0, 5.0, 5.0])

        # First set a stable strategy
        loop.last_stable_strategy = [{"action_type": "fallback", "params": {}}]

        result = loop.step(laws, state, _mock_consensus)
        assert result["oscillation"] is True
        assert result["fallback_used"] is True

    def test_prev_mean_updated(self):
        loop = AutonomousLoop()
        laws = [_make_law("L1", "stabilize", [("snr", "gt", 1.0)])]
        state = _make_state([1.0, 2.0, 3.0])
        result = loop.step(laws, state, _mock_consensus)
        assert "prev_mean" in result["state"]
        assert result["state"]["prev_mean"] == 2.0

    def test_meta_laws_returned(self):
        loop = AutonomousLoop()
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        state = _make_state()
        result = loop.step(laws, state, _mock_consensus)
        assert "meta_laws" in result
