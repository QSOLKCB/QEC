"""
Tests for QEC FSM Controller (v80.0.0).

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
    _build_analysis_result,
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
