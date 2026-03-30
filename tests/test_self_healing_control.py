"""Tests for self-healing control layer."""

from qec.analysis.self_healing_control import (
    compute_escalation_freeze,
    compute_healing_damping,
    compute_healing_mode,
    compute_suppression_intensity,
    run_self_healing_control,
)


def _make_adaptive(damping=0.8, mode="normal", gain=1.0):
    return {
        "adaptive_damping": damping,
        "response_mode": mode,
        "control_gain": gain,
    }


def _make_regime(jump=False, jump_type="none", coherence=5, behavior="locked"):
    return {
        "jump_detected": jump,
        "jump_type": jump_type,
        "coherence_length": coherence,
        "regime_behavior": behavior,
    }


class TestComputeSuppressionIntensity:
    def test_oscillatory_short(self):
        assert compute_suppression_intensity("oscillatory", 1) == 0.05

    def test_oscillatory_long_clamp(self):
        assert compute_suppression_intensity("oscillatory", 5) == 0.2

    def test_transition_scaling(self):
        assert compute_suppression_intensity("transition", 4) == 0.08

    def test_locked(self):
        assert compute_suppression_intensity("locked", 5) == 0.0

    def test_determinism(self):
        results = [compute_suppression_intensity("oscillatory", 3) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestComputeHealingDamping:
    def test_locked(self):
        assert compute_healing_damping(0.8, "locked") == 0.8

    def test_transition_default_coherence(self):
        assert compute_healing_damping(0.8, "transition") == 0.78

    def test_oscillatory_default_coherence(self):
        assert compute_healing_damping(0.8, "oscillatory") == 0.75

    def test_oscillatory_long_coherence(self):
        assert compute_healing_damping(0.8, "oscillatory", coherence_length=5) == 0.6

    def test_oscillatory_clamp(self):
        assert compute_healing_damping(0.3, "oscillatory") == 0.3

    def test_transition_clamp(self):
        assert compute_healing_damping(0.3, "transition") == 0.3


class TestComputeHealingMode:
    def test_locked(self):
        assert compute_healing_mode("locked") == "hold"

    def test_transition(self):
        assert compute_healing_mode("transition") == "stabilize"

    def test_oscillatory(self):
        assert compute_healing_mode("oscillatory") == "suppress"

    def test_unknown(self):
        assert compute_healing_mode("unknown") == "hold"


class TestComputeEscalationFreeze:
    def test_oscillatory(self):
        assert compute_escalation_freeze("oscillatory") is True

    def test_transition(self):
        assert compute_escalation_freeze("transition") is False

    def test_locked(self):
        assert compute_escalation_freeze("locked") is False


class TestRunSelfHealingControl:
    def test_locked(self):
        result = run_self_healing_control(
            _make_adaptive(0.8), _make_regime(behavior="locked")
        )
        assert result["healing_damping"] == 0.8
        assert result["healing_mode"] == "hold"
        assert result["escalation_frozen"] is False
        assert result["suppression_intensity"] == 0.0

    def test_transition(self):
        result = run_self_healing_control(
            _make_adaptive(0.8), _make_regime(behavior="transition", coherence=5)
        )
        assert result["healing_damping"] == 0.7
        assert result["healing_mode"] == "stabilize"
        assert result["escalation_frozen"] is False
        assert result["suppression_intensity"] == 0.1

    def test_oscillatory(self):
        result = run_self_healing_control(
            _make_adaptive(0.8), _make_regime(behavior="oscillatory", coherence=5)
        )
        assert result["healing_damping"] == 0.6
        assert result["healing_mode"] == "suppress"
        assert result["escalation_frozen"] is True
        assert result["suppression_intensity"] == 0.2

    def test_lower_clamp(self):
        result = run_self_healing_control(
            _make_adaptive(0.3), _make_regime(behavior="oscillatory")
        )
        assert result["healing_damping"] == 0.3

    def test_healing_gain_locked(self):
        result = run_self_healing_control(
            _make_adaptive(0.8), _make_regime(behavior="locked")
        )
        assert result["healing_gain"] == 1.0

    def test_determinism(self):
        adaptive = _make_adaptive(0.8)
        regime = _make_regime(behavior="oscillatory")
        results = [run_self_healing_control(adaptive, regime) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_missing_keys(self):
        result = run_self_healing_control({}, {})
        assert isinstance(result["healing_damping"], float)
        assert isinstance(result["healing_mode"], str)
        assert isinstance(result["escalation_frozen"], bool)
        assert isinstance(result["healing_gain"], float)
        assert isinstance(result["suppression_intensity"], float)
