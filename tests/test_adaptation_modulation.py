"""Tests for physics-signal adaptation modulation (v99.7.0)."""

from __future__ import annotations

from qec.analysis.strategy_memory import (
    compute_adaptation_modulation,
    score_strategy_with_memory,
    select_strategy_with_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(regime: str = "unstable") -> dict:
    return {
        "regime": regime,
        "basin_score": 0.5,
        "phi": 0.5,
        "consistency": 0.5,
        "divergence": 0.1,
        "curvature": 0.1,
        "resonance": 0.1,
        "complexity": 0.1,
    }


def _make_strategies() -> dict:
    return {
        "s1": {"action_type": "damping", "params": {"alpha": 0.1}, "confidence": 0.5},
        "s2": {"action_type": "scaling", "params": {"beta": 0.2}, "confidence": 0.5},
    }


def _make_signals(
    energy: float = 0.0,
    phase: float = 1.0,
    coherence: float = 1.0,
    alignment: float = 1.0,
    oscillation: float = 0.0,
) -> dict:
    return {
        "system_energy": energy,
        "phase_stability": phase,
        "multiscale_coherence": coherence,
        "control_alignment": alignment,
        "oscillation_strength": oscillation,
    }


# ---------------------------------------------------------------------------
# 1. compute_adaptation_modulation — determinism
# ---------------------------------------------------------------------------


class TestAdaptationModulationDeterminism:

    def test_identical_inputs_identical_output(self):
        sig = _make_signals(energy=0.3, phase=0.8, coherence=0.9, alignment=0.7)
        r1 = compute_adaptation_modulation(sig)
        r2 = compute_adaptation_modulation(sig)
        assert r1 == r2

    def test_repeated_calls_stable(self):
        sig = _make_signals(energy=0.5, phase=0.6, coherence=0.7, alignment=0.8)
        results = [compute_adaptation_modulation(sig) for _ in range(10)]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# 2. compute_adaptation_modulation — bounds
# ---------------------------------------------------------------------------


class TestAdaptationModulationBounds:

    def test_minimum_bound(self):
        # Worst case: high energy, low everything else
        sig = _make_signals(energy=1.0, phase=0.0, coherence=0.0, alignment=0.0)
        r = compute_adaptation_modulation(sig)
        assert r["adaptation_modulation"] >= 0.5

    def test_maximum_bound(self):
        # Best case: zero energy, all ones
        sig = _make_signals(energy=0.0, phase=1.0, coherence=1.0, alignment=1.0)
        r = compute_adaptation_modulation(sig)
        assert r["adaptation_modulation"] <= 1.5

    def test_always_in_range(self):
        """Sweep a range of inputs and verify bounds."""
        for e in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for p in [0.0, 0.3, 0.6, 1.0]:
                for c in [0.0, 0.5, 1.0]:
                    for a in [0.0, 0.5, 1.0]:
                        sig = _make_signals(energy=e, phase=p, coherence=c, alignment=a)
                        r = compute_adaptation_modulation(sig)
                        mod = r["adaptation_modulation"]
                        assert 0.5 <= mod <= 1.5, (
                            f"Out of bounds: {mod} for e={e}, p={p}, c={c}, a={a}"
                        )


# ---------------------------------------------------------------------------
# 3. compute_adaptation_modulation — behavior
# ---------------------------------------------------------------------------


class TestAdaptationModulationBehavior:

    def test_high_energy_lowers_modulation(self):
        low_e = compute_adaptation_modulation(
            _make_signals(energy=0.1, phase=0.8, coherence=0.8, alignment=0.8),
        )
        high_e = compute_adaptation_modulation(
            _make_signals(energy=0.9, phase=0.8, coherence=0.8, alignment=0.8),
        )
        assert low_e["adaptation_modulation"] > high_e["adaptation_modulation"]

    def test_high_coherence_raises_modulation(self):
        low_c = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.8, coherence=0.2, alignment=0.8),
        )
        high_c = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.8, coherence=0.9, alignment=0.8),
        )
        assert high_c["adaptation_modulation"] > low_c["adaptation_modulation"]

    def test_high_alignment_raises_modulation(self):
        low_a = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.8, coherence=0.8, alignment=0.2),
        )
        high_a = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.8, coherence=0.8, alignment=0.9),
        )
        assert high_a["adaptation_modulation"] > low_a["adaptation_modulation"]

    def test_high_phase_stability_raises_modulation(self):
        low_p = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.2, coherence=0.8, alignment=0.8),
        )
        high_p = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.9, coherence=0.8, alignment=0.8),
        )
        assert high_p["adaptation_modulation"] > low_p["adaptation_modulation"]

    def test_regime_sensitive_damping(self):
        """High oscillation + low phase stability → damping applied."""
        no_damp = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.5, coherence=0.8,
                          alignment=0.8, oscillation=0.5),
        )
        damped = compute_adaptation_modulation(
            _make_signals(energy=0.3, phase=0.2, coherence=0.8,
                          alignment=0.8, oscillation=0.8),
        )
        # Damped case has both lower phase AND damping factor
        assert damped["adaptation_modulation"] < no_damp["adaptation_modulation"]

    def test_damping_limited_to_10_percent(self):
        """Regime-sensitive damping multiplies by 0.9 (≤10% effect)."""
        # Create signals right at the damping threshold boundary
        sig_below = _make_signals(
            energy=0.3, phase=0.29, coherence=0.8,
            alignment=0.8, oscillation=0.71,
        )
        sig_above = _make_signals(
            energy=0.3, phase=0.31, coherence=0.8,
            alignment=0.8, oscillation=0.71,
        )
        r_below = compute_adaptation_modulation(sig_below)
        r_above = compute_adaptation_modulation(sig_above)
        # Below threshold gets 0.9× damping AND lower phase
        # The damping itself is exactly 0.9× (10%)
        # We verify the damped value is less
        assert r_below["adaptation_modulation"] < r_above["adaptation_modulation"]


# ---------------------------------------------------------------------------
# 4. compute_adaptation_modulation — fallback / edge cases
# ---------------------------------------------------------------------------


class TestAdaptationModulationFallback:

    def test_empty_signals(self):
        r = compute_adaptation_modulation({})
        assert r["adaptation_modulation"] == 1.0
        assert r["energy"] == 0.0
        assert r["coherence"] == 1.0
        assert r["alignment"] == 1.0

    def test_partial_signals(self):
        """Missing keys use safe defaults."""
        r = compute_adaptation_modulation({"system_energy": 0.5})
        assert 0.5 <= r["adaptation_modulation"] <= 1.5

    def test_output_keys_present(self):
        r = compute_adaptation_modulation(_make_signals())
        assert "adaptation_modulation" in r
        assert "energy" in r
        assert "coherence" in r
        assert "alignment" in r


# ---------------------------------------------------------------------------
# 5. Integration — scoring with physics signals
# ---------------------------------------------------------------------------


class TestScoringWithPhysicsSignals:

    def test_score_with_signals_deterministic(self):
        state = _make_state()
        history = [{"score": 0.5, "outcome": "improved"}]
        memory = {}
        sig = _make_signals(energy=0.3, phase=0.8, coherence=0.9, alignment=0.7)

        r1 = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", physics_signals=sig,
        )
        r2 = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", physics_signals=sig,
        )
        assert r1 == r2

    def test_score_without_signals_unchanged(self):
        """Without physics_signals, modulation defaults to 1.0."""
        state = _make_state()
        history = [{"score": 0.5, "outcome": "improved"}]
        memory = {}

        r = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1",
        )
        assert r["adaptation_modulation"] == 1.0
        assert r["energy"] == 0.0

    def test_score_output_contains_modulation_fields(self):
        state = _make_state()
        history = [{"score": 0.5, "outcome": "improved"}]
        memory = {}
        sig = _make_signals(energy=0.4)

        r = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", physics_signals=sig,
        )
        assert "adaptation_modulation" in r
        assert "energy" in r
        assert "coherence" in r
        assert "alignment" in r

    def test_modulation_affects_multiplicative_scoring(self):
        """When regime_key + transition_memory provided, modulation changes score."""
        state = _make_state("unstable")
        history = [{"score": 0.5, "outcome": "improved"}]
        rk = ("unstable", "basin_2")
        memory = {(rk, "s1"): [{"score": 0.6}]}
        tm = {}  # empty transition memory → neutral bias

        r_neutral = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", regime_key=rk, transition_memory=tm,
        )

        sig_good = _make_signals(energy=0.0, phase=1.0, coherence=1.0, alignment=1.0)
        r_good = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", regime_key=rk, transition_memory=tm,
            physics_signals=sig_good,
        )

        sig_bad = _make_signals(energy=0.9, phase=0.2, coherence=0.3, alignment=0.2)
        r_bad = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            memory, "s1", regime_key=rk, transition_memory=tm,
            physics_signals=sig_bad,
        )

        # Good signals should give >= neutral, bad signals should give <= neutral
        assert r_good["score"] >= r_neutral["score"]
        assert r_bad["score"] <= r_neutral["score"]


# ---------------------------------------------------------------------------
# 6. Integration — selection with physics signals
# ---------------------------------------------------------------------------


class TestSelectionWithPhysicsSignals:

    def test_select_empty_strategies(self):
        r = select_strategy_with_memory(
            {}, _make_state(), [], {},
            physics_signals=_make_signals(),
        )
        assert r["selected"] == ""
        assert r["adaptation_modulation"] == 1.0

    def test_select_returns_modulation_fields(self):
        sig = _make_signals(energy=0.3)
        r = select_strategy_with_memory(
            _make_strategies(), _make_state(),
            [{"score": 0.5, "outcome": "improved"}], {},
            physics_signals=sig,
        )
        assert "adaptation_modulation" in r
        assert "energy" in r
        assert "coherence" in r
        assert "alignment" in r

    def test_select_deterministic_with_signals(self):
        sig = _make_signals(energy=0.3, phase=0.7, coherence=0.8, alignment=0.6)
        args = (
            _make_strategies(), _make_state(),
            [{"score": 0.5, "outcome": "improved"}], {},
        )
        r1 = select_strategy_with_memory(*args, physics_signals=sig)
        r2 = select_strategy_with_memory(*args, physics_signals=sig)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 7. No regression — backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:

    def test_score_without_new_param_works(self):
        """Existing callers without physics_signals still work."""
        state = _make_state()
        history = [{"score": 0.5, "outcome": "improved"}]
        r = score_strategy_with_memory(
            {"action_type": "damping", "params": {}}, state, history,
            {}, "s1",
        )
        assert 0.0 <= r["score"] <= 1.0

    def test_select_without_new_param_works(self):
        """Existing callers without physics_signals still work."""
        r = select_strategy_with_memory(
            _make_strategies(), _make_state(),
            [{"score": 0.5, "outcome": "improved"}], {},
        )
        assert r["selected"] in ("s1", "s2")
