"""
Tests for QEC FSM Controller (v80.1.0).

Covers:
- determinism (identical runs produce identical results)
- state transition correctness
- termination conditions
- no mutation of input data
- max_steps enforcement
- accept/reject logic
- history structure
- configuration handling
- edge cases
"""

from __future__ import annotations

import copy

import pytest

from qec.controller.qec_fsm import (
    QECFSM,
    INIT,
    ANALYZE,
    PERTURB,
    INVARIANT,
    EVALUATE,
    ACCEPT,
    REJECT,
    TERMINATE,
    _VALID_STATES,
    _DEFAULT_CONFIG,
    _adapt_thresholds,
    _build_analysis_result,
    _has_converged,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data() -> dict:
    """Stable sample input that should lead to ACCEPT."""
    return {
        "rms_energy": 0.01,
        "spectral_centroid_hz": 500.0,
        "spectral_spread_hz": 200.0,
        "zero_crossing_rate": 0.05,
        "fft_top_peaks": [
            {"frequency_hz": 100.0, "magnitude": 0.5},
            {"frequency_hz": 200.0, "magnitude": 0.3},
        ],
    }


@pytest.fixture
def default_fsm() -> QECFSM:
    return QECFSM()


@pytest.fixture
def strict_fsm() -> QECFSM:
    """FSM with very strict thresholds that should reject easily."""
    return QECFSM(config={
        "stability_threshold": 0.0001,
        "boundary_crossing_threshold": 1,
        "max_reject_cycles": 1,
    })


# ---------------------------------------------------------------------------
# 1. Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Identical inputs must produce identical outputs."""

    def test_run_deterministic(self, sample_data: dict) -> None:
        """Two FSM runs with identical input produce identical results."""
        fsm1 = QECFSM()
        fsm2 = QECFSM()
        r1 = fsm1.run(copy.deepcopy(sample_data), max_steps=20)
        r2 = fsm2.run(copy.deepcopy(sample_data), max_steps=20)
        assert r1 == r2

    def test_step_deterministic(self, sample_data: dict) -> None:
        """Two FSM instances stepping identically produce same transitions."""
        fsm1 = QECFSM()
        fsm2 = QECFSM()
        s1, d1 = fsm1.step(INIT, copy.deepcopy(sample_data))
        s2, d2 = fsm2.step(INIT, copy.deepcopy(sample_data))
        assert s1 == s2
        assert d1 == d2

    def test_repeated_runs_deterministic(self, sample_data: dict) -> None:
        """The same FSM instance run twice produces identical results."""
        fsm = QECFSM()
        r1 = fsm.run(copy.deepcopy(sample_data), max_steps=20)
        r2 = fsm.run(copy.deepcopy(sample_data), max_steps=20)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 2. State transition tests
# ---------------------------------------------------------------------------

class TestStateTransitions:
    """Verify each state handler routes to the correct next state."""

    def test_init_to_analyze(self, default_fsm: QECFSM, sample_data: dict) -> None:
        next_state, data = default_fsm.step(INIT, sample_data)
        assert next_state == ANALYZE

    def test_analyze_to_perturb(self, default_fsm: QECFSM, sample_data: dict) -> None:
        _, data = default_fsm.step(INIT, sample_data)
        next_state, _ = default_fsm.step(ANALYZE, data)
        assert next_state == PERTURB

    def test_perturb_to_invariant(self, default_fsm: QECFSM, sample_data: dict) -> None:
        _, data = default_fsm.step(INIT, sample_data)
        _, data = default_fsm.step(ANALYZE, data)
        next_state, _ = default_fsm.step(PERTURB, data)
        assert next_state == INVARIANT

    def test_invariant_to_evaluate(self, default_fsm: QECFSM, sample_data: dict) -> None:
        _, data = default_fsm.step(INIT, sample_data)
        _, data = default_fsm.step(ANALYZE, data)
        _, data = default_fsm.step(PERTURB, data)
        next_state, _ = default_fsm.step(INVARIANT, data)
        assert next_state == EVALUATE

    def test_accept_to_terminate(self, default_fsm: QECFSM, sample_data: dict) -> None:
        next_state, data = default_fsm.step(ACCEPT, sample_data)
        assert next_state == TERMINATE
        assert data["verdict"] == ACCEPT

    def test_terminate_stays_terminate(self, default_fsm: QECFSM, sample_data: dict) -> None:
        next_state, _ = default_fsm.step(TERMINATE, sample_data)
        assert next_state == TERMINATE

    def test_invalid_state_raises(self, default_fsm: QECFSM, sample_data: dict) -> None:
        with pytest.raises(ValueError, match="Invalid FSM state"):
            default_fsm.step("BOGUS", sample_data)


# ---------------------------------------------------------------------------
# 3. Termination tests
# ---------------------------------------------------------------------------

class TestTermination:
    """Verify the FSM terminates correctly."""

    def test_max_steps_enforced(self, sample_data: dict) -> None:
        """FSM terminates when max_steps is reached."""
        fsm = QECFSM(config={"stability_threshold": -1.0})  # never accept
        result = fsm.run(sample_data, max_steps=5)
        assert result["steps"] <= 5
        assert fsm.state == TERMINATE

    def test_terminates_on_accept(self, sample_data: dict) -> None:
        """FSM terminates when ACCEPT is reached."""
        fsm = QECFSM(config={"stability_threshold": 100.0})
        result = fsm.run(sample_data, max_steps=20)
        assert result["final_state"] == ACCEPT

    def test_terminates_on_reject_exhaustion(self, sample_data: dict) -> None:
        """FSM terminates after max_reject_cycles rejections."""
        fsm = QECFSM(config={
            "stability_threshold": 0.0001,
            "boundary_crossing_threshold": 0,
            "max_reject_cycles": 2,
        })
        result = fsm.run(sample_data, max_steps=50)
        assert result["final_state"] == REJECT


# ---------------------------------------------------------------------------
# 4. No-mutation tests
# ---------------------------------------------------------------------------

class TestNoMutation:
    """Input data must never be mutated by the FSM."""

    def test_run_does_not_mutate_input(self, sample_data: dict) -> None:
        original = copy.deepcopy(sample_data)
        fsm = QECFSM()
        fsm.run(sample_data, max_steps=20)
        assert sample_data == original

    def test_step_does_not_mutate_input(self, sample_data: dict) -> None:
        original = copy.deepcopy(sample_data)
        fsm = QECFSM()
        fsm.step(INIT, sample_data)
        assert sample_data == original


# ---------------------------------------------------------------------------
# 5. History structure tests
# ---------------------------------------------------------------------------

class TestHistory:
    """Verify the history trace is complete and well-formed."""

    def test_history_records_all_transitions(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        assert len(result["history"]) >= result["steps"]

    def test_history_entry_has_required_keys(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        for entry in result["history"]:
            assert "from_state" in entry
            assert "to_state" in entry

    def test_history_starts_from_init(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        assert result["history"][0]["from_state"] == INIT

    def test_history_ends_at_terminate(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        last = result["history"][-1]
        assert last["to_state"] == TERMINATE


# ---------------------------------------------------------------------------
# 6. Accept / Reject logic tests
# ---------------------------------------------------------------------------

class TestEvaluationLogic:
    """Verify EVALUATE transitions are based on thresholds."""

    def test_high_threshold_accepts_easily(self, sample_data: dict) -> None:
        fsm = QECFSM(config={"stability_threshold": 100.0})
        result = fsm.run(sample_data, max_steps=20)
        assert result["final_state"] == ACCEPT

    def test_zero_threshold_rejects(self, sample_data: dict) -> None:
        fsm = QECFSM(config={
            "stability_threshold": 0.0,
            "boundary_crossing_threshold": 0,
            "max_reject_cycles": 1,
        })
        result = fsm.run(sample_data, max_steps=50)
        assert result["final_state"] == REJECT

    def test_boundary_crossing_triggers_reject(self, sample_data: dict) -> None:
        """Low boundary_crossing_threshold forces REJECT path."""
        fsm = QECFSM(config={
            "stability_threshold": 0.0001,
            "boundary_crossing_threshold": 0,
            "max_reject_cycles": 1,
        })
        result = fsm.run(sample_data, max_steps=50)
        assert result["final_state"] == REJECT


# ---------------------------------------------------------------------------
# 7. Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Verify FSM configuration handling."""

    def test_default_config_used(self) -> None:
        fsm = QECFSM()
        for key in _DEFAULT_CONFIG:
            assert key in fsm._config

    def test_custom_config_overrides(self) -> None:
        fsm = QECFSM(config={"epsilon": 0.1})
        assert fsm._config["epsilon"] == 0.1

    def test_partial_config_keeps_defaults(self) -> None:
        fsm = QECFSM(config={"epsilon": 0.1})
        assert fsm._config["n_perturbations"] == _DEFAULT_CONFIG["n_perturbations"]


# ---------------------------------------------------------------------------
# 8. Output structure tests
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Verify the output dict has the expected shape."""

    def test_result_has_required_keys(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        assert "final_state" in result
        assert "steps" in result
        assert "history" in result

    def test_steps_is_positive_int(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        assert isinstance(result["steps"], int)
        assert result["steps"] > 0

    def test_final_state_is_valid(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        assert result["final_state"] in {ACCEPT, REJECT, TERMINATE}


# ---------------------------------------------------------------------------
# 9. Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Test internal helper functions."""

    def test_build_analysis_result_from_data(self) -> None:
        data = {"rms_energy": 0.5, "spectral_centroid_hz": 100.0}
        result = _build_analysis_result(data)
        assert result["rms_energy"] == 0.5
        assert result["spectral_centroid_hz"] == 100.0

    def test_build_analysis_result_with_explicit_result(self) -> None:
        inner = {"rms_energy": 0.9, "spectral_centroid_hz": 999.0,
                 "spectral_spread_hz": 1.0, "zero_crossing_rate": 0.1}
        data = {"result": inner}
        result = _build_analysis_result(data)
        assert result == inner

    def test_build_analysis_result_does_not_mutate(self) -> None:
        inner = {"rms_energy": 0.9, "spectral_centroid_hz": 999.0,
                 "spectral_spread_hz": 1.0, "zero_crossing_rate": 0.1}
        data = {"result": inner}
        result = _build_analysis_result(data)
        result["rms_energy"] = 0.0
        assert inner["rms_energy"] == 0.9

    def test_build_analysis_result_defaults(self) -> None:
        result = _build_analysis_result({})
        assert "rms_energy" in result
        assert "spectral_centroid_hz" in result
        assert "spectral_spread_hz" in result
        assert "zero_crossing_rate" in result


# ---------------------------------------------------------------------------
# 10. Valid states
# ---------------------------------------------------------------------------

class TestValidStates:
    """Verify the state set is correct."""

    def test_all_states_present(self) -> None:
        expected = {INIT, ANALYZE, PERTURB, INVARIANT,
                    EVALUATE, ACCEPT, REJECT, TERMINATE}
        assert _VALID_STATES == expected

    def test_state_strings(self) -> None:
        assert INIT == "INIT"
        assert TERMINATE == "TERMINATE"


# ---------------------------------------------------------------------------
# 11. Enriched history fields (v80.0.1)
# ---------------------------------------------------------------------------

class TestEnrichedHistory:
    """Verify history entries include epsilon, reject_cycle, decision."""

    def test_history_contains_epsilon(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        for entry in result["history"]:
            assert "epsilon" in entry
            assert isinstance(entry["epsilon"], float)

    def test_history_contains_reject_cycle(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        for entry in result["history"]:
            assert "reject_cycle" in entry
            assert isinstance(entry["reject_cycle"], int)

    def test_history_contains_decision(self, sample_data: dict) -> None:
        fsm = QECFSM()
        result = fsm.run(sample_data, max_steps=20)
        for entry in result["history"]:
            assert "decision" in entry
            assert entry["decision"] in {"ACCEPT", "REJECT", "CONTINUE"}

    def test_accept_decision_logged(self, sample_data: dict) -> None:
        """An accepting run must have exactly one ACCEPT decision entry."""
        fsm = QECFSM(config={"stability_threshold": 100.0})
        result = fsm.run(sample_data, max_steps=20)
        accept_entries = [e for e in result["history"] if e["decision"] == "ACCEPT"]
        assert len(accept_entries) == 1

    def test_reject_cycle_increments(self, sample_data: dict) -> None:
        """reject_cycle must increase after each REJECT transition."""
        fsm = QECFSM(config={
            "stability_threshold": 0.0001,
            "boundary_crossing_threshold": 0,
            "max_reject_cycles": 3,
        })
        result = fsm.run(sample_data, max_steps=50)
        reject_entries = [e for e in result["history"] if e["decision"] == "REJECT"]
        cycles = [e["reject_cycle"] for e in reject_entries]
        # Each reject should have a higher reject_cycle than the previous.
        assert cycles == sorted(cycles)
        assert len(cycles) > 0

    def test_epsilon_tightens_on_reject(self, sample_data: dict) -> None:
        """epsilon should halve after each reject cycle that retries."""
        fsm = QECFSM(config={
            "stability_threshold": 0.0001,
            "boundary_crossing_threshold": 0,
            "max_reject_cycles": 3,
            "epsilon": 1e-3,
        })
        result = fsm.run(sample_data, max_steps=50)
        # Find PERTURB entries (where epsilon is actually used).
        perturb_entries = [
            e for e in result["history"] if e["from_state"] == PERTURB
        ]
        if len(perturb_entries) >= 2:
            # Second PERTURB should use smaller epsilon than first.
            assert perturb_entries[1]["epsilon"] < perturb_entries[0]["epsilon"]


# ---------------------------------------------------------------------------
# 12. Convergence detection (v80.0.1)
# ---------------------------------------------------------------------------

class TestConvergence:
    """Verify _has_converged helper and early termination."""

    def test_converged_identical_scores(self) -> None:
        history = [
            {"stability_score": 0.5},
            {"stability_score": 0.5},
            {"stability_score": 0.5},
        ]
        assert _has_converged(history, window=3, tolerance=1e-6) is True

    def test_not_converged_different_scores(self) -> None:
        history = [
            {"stability_score": 0.5},
            {"stability_score": 1.0},
            {"stability_score": 0.5},
        ]
        assert _has_converged(history, window=3, tolerance=1e-6) is False

    def test_not_converged_too_few_entries(self) -> None:
        history = [{"stability_score": 0.5}]
        assert _has_converged(history, window=3) is False

    def test_converged_within_tolerance(self) -> None:
        history = [
            {"stability_score": 1.0},
            {"stability_score": 1.0 + 1e-8},
            {"stability_score": 1.0 - 1e-8},
        ]
        assert _has_converged(history, window=3, tolerance=1e-6) is True

    def test_none_scores_skipped(self) -> None:
        history = [
            {"stability_score": None},
            {"stability_score": 0.5},
            {"stability_score": None},
            {"stability_score": 0.5},
            {"stability_score": 0.5},
        ]
        assert _has_converged(history, window=3, tolerance=1e-6) is True

    def test_early_termination_on_convergence(self, sample_data: dict) -> None:
        """FSM that loops without accepting should eventually converge."""
        # With stability_threshold=-1 nothing is accepted, so FSM loops.
        # The stability scores will be identical each cycle → convergence.
        fsm = QECFSM(config={
            "stability_threshold": -1.0,
            "boundary_crossing_threshold": 999,
        })
        result = fsm.run(sample_data, max_steps=50)
        converge_entries = [
            e for e in result["history"] if e.get("reason") == "converged"
        ]
        assert len(converge_entries) == 1


# ---------------------------------------------------------------------------
# 13. Adaptive threshold tests (v80.1.0)
# ---------------------------------------------------------------------------

class TestAdaptiveThresholds:
    """Verify deterministic adaptive threshold behaviour."""

    def test_thresholds_update_deterministically(self) -> None:
        """Same history produces same thresholds."""
        history = [
            {"stability_score": 1.0, "decision": "CONTINUE"},
            {"stability_score": 2.0, "decision": "CONTINUE"},
            {"stability_score": 3.0, "decision": "CONTINUE"},
        ]
        r1 = _adapt_thresholds(history, 0.5, 2.0, window=3)
        r2 = _adapt_thresholds(history, 0.5, 2.0, window=3)
        assert r1 == r2

    def test_thresholds_stay_within_bounds(self) -> None:
        """Thresholds must be clamped to [0, 10]."""
        # Very high scores should push stability toward 10.
        history = [
            {"stability_score": 100.0, "decision": "CONTINUE"},
        ] * 5
        st, bt = _adapt_thresholds(history, 9.5, 9.5, window=5)
        assert 0.0 <= st <= 10.0
        assert 0.0 <= bt <= 10.0

        # Very negative scores should push toward 0.
        history_neg = [
            {"stability_score": -100.0, "decision": "CONTINUE"},
        ] * 5
        st2, bt2 = _adapt_thresholds(history_neg, 0.5, 0.5, window=5)
        assert 0.0 <= st2 <= 10.0
        assert 0.0 <= bt2 <= 10.0

    def test_no_mutation_of_inputs(self) -> None:
        """_adapt_thresholds must not mutate the history list."""
        history = [
            {"stability_score": 1.0, "decision": "CONTINUE"},
            {"stability_score": 2.0, "decision": "REJECT"},
        ]
        original = copy.deepcopy(history)
        _adapt_thresholds(history, 0.5, 2.0, window=5)
        assert history == original

    def test_behaviour_with_short_history(self) -> None:
        """When history < window, use all available entries."""
        history = [{"stability_score": 1.0, "decision": "CONTINUE"}]
        st, bt = _adapt_thresholds(history, 0.5, 2.0, window=5)
        # EMA: 0.8 * 0.5 + 0.2 * 1.0 = 0.6
        assert abs(st - 0.6) < 1e-9

    def test_adaptation_only_on_continue_loops(self, sample_data: dict) -> None:
        """Thresholds entry only appears on EVALUATE → ANALYZE transitions."""
        fsm = QECFSM(config={
            "stability_threshold": -1.0,
            "boundary_crossing_threshold": 999,
        })
        result = fsm.run(sample_data, max_steps=50)
        for entry in result["history"]:
            if "thresholds" in entry:
                # Must be an EVALUATE → ANALYZE transition.
                assert entry["from_state"] == EVALUATE
                assert entry["to_state"] == ANALYZE

    def test_convergence_still_works(self, sample_data: dict) -> None:
        """Convergence detection must still trigger with adaptive thresholds."""
        fsm = QECFSM(config={
            "stability_threshold": -1.0,
            "boundary_crossing_threshold": 999,
        })
        result = fsm.run(sample_data, max_steps=50)
        converge_entries = [
            e for e in result["history"] if e.get("reason") == "converged"
        ]
        assert len(converge_entries) == 1

    def test_thresholds_trend_toward_mean_score(self) -> None:
        """Stability threshold should move toward the mean of recent scores."""
        history = [
            {"stability_score": 4.0, "decision": "CONTINUE"},
            {"stability_score": 4.0, "decision": "CONTINUE"},
            {"stability_score": 4.0, "decision": "CONTINUE"},
        ]
        # Starting threshold 0.5, mean is 4.0.
        # new = 0.8 * 0.5 + 0.2 * 4.0 = 1.2
        st, _ = _adapt_thresholds(history, 0.5, 2.0, window=3)
        assert abs(st - 1.2) < 1e-9
        # Apply again: new = 0.8 * 1.2 + 0.2 * 4.0 = 1.76
        st2, _ = _adapt_thresholds(history, st, 2.0, window=3)
        assert abs(st2 - 1.76) < 1e-9
        # Trending toward 4.0.
        assert st2 > st

    def test_reject_pressure_affects_boundary(self) -> None:
        """High reject rate should increase boundary threshold."""
        # All REJECT decisions → reject_rate > 0.5 → increase.
        history = [
            {"stability_score": 1.0, "decision": "REJECT"},
            {"stability_score": 1.0, "decision": "REJECT"},
            {"stability_score": 1.0, "decision": "REJECT"},
        ]
        _, bt = _adapt_thresholds(history, 0.5, 2.0, window=3)
        assert bt > 2.0

        # All CONTINUE decisions → reject_rate < 0.2 → decrease.
        history_continue = [
            {"stability_score": 1.0, "decision": "CONTINUE"},
            {"stability_score": 1.0, "decision": "CONTINUE"},
            {"stability_score": 1.0, "decision": "CONTINUE"},
        ]
        _, bt2 = _adapt_thresholds(history_continue, 0.5, 2.0, window=3)
        assert bt2 < 2.0
