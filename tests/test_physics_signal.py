"""Tests for physics-informed signal layer (v99.6.0).

Covers: bounds, determinism, monotonic sanity, no NaN/division errors.
"""

from __future__ import annotations

import numpy as np

from qec.analysis.physics_signal import (
    compute_control_alignment,
    compute_geometric_consistency,
    compute_multiscale_coherence,
    compute_oscillation_strength,
    compute_phase_lock_ratio,
    compute_phase_stability,
    compute_physics_signals,
    compute_scale_conflict,
    compute_system_energy,
    map_state_to_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> None:
    assert not np.isnan(value), f"NaN detected: {value}"
    assert not np.isinf(value), f"Inf detected: {value}"
    assert lo <= value <= hi, f"{value} not in [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Task 1 — Oscillation & Phase Metrics
# ---------------------------------------------------------------------------


class TestOscillationStrength:
    def test_monotone_zero(self) -> None:
        assert compute_oscillation_strength([1.0, 2.0, 3.0, 4.0, 5.0]) == 0.0

    def test_alternating_high(self) -> None:
        val = compute_oscillation_strength([1.0, 3.0, 1.0, 3.0, 1.0])
        assert val == 1.0

    def test_bounded(self) -> None:
        for seq in [[1, 2, 3], [3, 1, 4, 1, 5], [0, 0, 0], [1, -1, 1, -1]]:
            _assert_bounded(compute_oscillation_strength(seq))

    def test_short_history(self) -> None:
        assert compute_oscillation_strength([1.0]) == 0.0
        assert compute_oscillation_strength([1.0, 2.0]) == 0.0

    def test_deterministic(self) -> None:
        h = [1.0, 3.0, 2.0, 4.0, 1.0, 5.0]
        assert compute_oscillation_strength(h) == compute_oscillation_strength(h)

    def test_empty(self) -> None:
        assert compute_oscillation_strength([]) == 0.0


class TestPhaseStability:
    def test_constant_perfect(self) -> None:
        assert compute_phase_stability([5.0, 5.0, 5.0, 5.0]) == 1.0

    def test_high_variance_low_stability(self) -> None:
        val = compute_phase_stability([0.0, 100.0, 0.0, 100.0])
        assert val < 0.01

    def test_bounded(self) -> None:
        for seq in [[1, 2, 3], [0, 0, 0], [10, -10, 10], [1.5]]:
            _assert_bounded(compute_phase_stability(seq))

    def test_single_element(self) -> None:
        assert compute_phase_stability([42.0]) == 1.0

    def test_deterministic(self) -> None:
        h = [1.0, 2.5, 0.3, 4.1]
        assert compute_phase_stability(h) == compute_phase_stability(h)


class TestPhaseLockRatio:
    def test_monotone_locked(self) -> None:
        assert compute_phase_lock_ratio([1.0, 2.0, 3.0, 4.0, 5.0]) == 1.0

    def test_alternating_unlocked(self) -> None:
        assert compute_phase_lock_ratio([1.0, 3.0, 1.0, 3.0, 1.0]) == 0.0

    def test_bounded(self) -> None:
        for seq in [[1, 2, 3], [3, 1, 4], [0, 0, 0]]:
            _assert_bounded(compute_phase_lock_ratio(seq))

    def test_short(self) -> None:
        assert compute_phase_lock_ratio([1.0, 2.0]) == 0.0

    def test_deterministic(self) -> None:
        h = [1.0, 3.0, 2.0, 4.0]
        assert compute_phase_lock_ratio(h) == compute_phase_lock_ratio(h)


# ---------------------------------------------------------------------------
# Task 2 — Multi-Scale Coherence
# ---------------------------------------------------------------------------


class TestMultiscaleCoherence:
    def test_uniform_perfect(self) -> None:
        assert compute_multiscale_coherence([5.0, 5.0, 5.0]) == 1.0

    def test_divergent_low(self) -> None:
        val = compute_multiscale_coherence([0.0, 100.0])
        assert val <= 0.5

    def test_single_element(self) -> None:
        assert compute_multiscale_coherence([3.0]) == 1.0

    def test_bounded(self) -> None:
        for scales in [[1, 2, 3], [0, 0, 0], [10, 10, 10.001]]:
            _assert_bounded(compute_multiscale_coherence(scales))

    def test_deterministic(self) -> None:
        s = [1.5, 2.5, 3.5]
        assert compute_multiscale_coherence(s) == compute_multiscale_coherence(s)


class TestScaleConflict:
    def test_no_conflict(self) -> None:
        assert compute_scale_conflict([5.0, 5.0, 5.0]) == 0.0

    def test_high_conflict(self) -> None:
        val = compute_scale_conflict([0.0, 100.0])
        assert val > 0.9

    def test_single_element(self) -> None:
        assert compute_scale_conflict([3.0]) == 0.0

    def test_bounded(self) -> None:
        for scales in [[1, 2, 3], [0, 0], [-5, 5]]:
            _assert_bounded(compute_scale_conflict(scales))

    def test_deterministic(self) -> None:
        s = [1.0, 3.0, 7.0]
        assert compute_scale_conflict(s) == compute_scale_conflict(s)


# ---------------------------------------------------------------------------
# Task 3 — Energy / Effort Metric
# ---------------------------------------------------------------------------


class TestSystemEnergy:
    def test_zero_energy(self) -> None:
        val = compute_system_energy({"variance": 0.0, "oscillation": 0.0}, [])
        assert val == 0.0

    def test_high_energy(self) -> None:
        val = compute_system_energy(
            {"variance": 10.0, "oscillation": 10.0},
            [10.0, 10.0, 10.0],
        )
        assert val > 0.9

    def test_bounded(self) -> None:
        _assert_bounded(compute_system_energy({"variance": 1.0}, [0.5, 0.5]))
        _assert_bounded(compute_system_energy({}, []))
        _assert_bounded(compute_system_energy({"oscillation": 100.0}, [100.0]))

    def test_deterministic(self) -> None:
        m = {"variance": 2.0, "oscillation": 1.5}
        d = [0.3, 0.7, 0.1]
        assert compute_system_energy(m, d) == compute_system_energy(m, d)

    def test_monotonic_with_increasing_variance(self) -> None:
        e1 = compute_system_energy({"variance": 1.0}, [0.0])
        e2 = compute_system_energy({"variance": 5.0}, [0.0])
        assert e2 > e1


# ---------------------------------------------------------------------------
# Task 4 — Local vs Global Alignment
# ---------------------------------------------------------------------------


class TestControlAlignment:
    def test_identical_perfect(self) -> None:
        m = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert compute_control_alignment(m, m) == 1.0

    def test_opposite_low(self) -> None:
        local = {"a": 1.0, "b": 0.0}
        glob = {"a": -1.0, "b": 0.0}
        val = compute_control_alignment(local, glob)
        assert val == 0.0

    def test_no_shared_keys(self) -> None:
        assert compute_control_alignment({"a": 1.0}, {"b": 2.0}) == 0.0

    def test_bounded(self) -> None:
        _assert_bounded(compute_control_alignment({"x": 5.0}, {"x": -3.0}))
        _assert_bounded(compute_control_alignment({"x": 0.0}, {"x": 0.0}))

    def test_deterministic(self) -> None:
        l = {"a": 1.0, "b": 2.0}
        g = {"a": 0.5, "b": -1.0}
        assert compute_control_alignment(l, g) == compute_control_alignment(l, g)


# ---------------------------------------------------------------------------
# Task 5 — Geometric Consistency
# ---------------------------------------------------------------------------


class TestGeometricConsistency:
    def test_uniform_curvature(self) -> None:
        assert compute_geometric_consistency([1.0, 1.0, 1.0]) == 1.0

    def test_scalar_curvature(self) -> None:
        assert compute_geometric_consistency(0.5) == 1.0

    def test_empty_curvature(self) -> None:
        assert compute_geometric_consistency([]) == 1.0

    def test_with_structure(self) -> None:
        val = compute_geometric_consistency(
            [1.0, 1.0],
            {"topology": 0.8, "resonance": 0.9},
        )
        _assert_bounded(val)

    def test_bounded(self) -> None:
        _assert_bounded(compute_geometric_consistency([0, 100, 0, 100]))
        _assert_bounded(compute_geometric_consistency(0.0, {"topology": 0.0, "resonance": 0.0}))

    def test_deterministic(self) -> None:
        c = [1.0, 2.0, 3.0]
        s = {"topology": 0.5, "resonance": 0.7}
        assert compute_geometric_consistency(c, s) == compute_geometric_consistency(c, s)


# ---------------------------------------------------------------------------
# Task 6 — Sonification Bridge
# ---------------------------------------------------------------------------


class TestMapStateToSignal:
    def test_keys_present(self) -> None:
        result = map_state_to_signal({})
        assert set(result.keys()) == {"tension", "coherence", "stability", "intensity"}

    def test_all_bounded(self) -> None:
        result = map_state_to_signal({
            "oscillation": 0.8,
            "variance": 0.3,
            "coherence": 0.6,
            "energy": 0.5,
            "alignment": 0.7,
            "stability": 0.9,
            "phase_stability": 0.4,
        })
        for key, val in result.items():
            _assert_bounded(val)

    def test_empty_metrics(self) -> None:
        result = map_state_to_signal({})
        for val in result.values():
            _assert_bounded(val)

    def test_deterministic(self) -> None:
        m = {"oscillation": 0.5, "variance": 0.2, "energy": 0.3}
        assert map_state_to_signal(m) == map_state_to_signal(m)

    def test_extreme_values_clipped(self) -> None:
        result = map_state_to_signal({"oscillation": 5.0, "variance": -1.0})
        for val in result.values():
            _assert_bounded(val)


# ---------------------------------------------------------------------------
# Task 7 — Integration (compute_physics_signals)
# ---------------------------------------------------------------------------


class TestComputePhysicsSignals:
    def test_all_inputs(self) -> None:
        result = compute_physics_signals(
            history=[1.0, 2.0, 1.0, 3.0, 2.0],
            scales=[1.0, 1.1, 0.9],
            metrics={"variance": 0.5, "oscillation": 0.3},
            deltas=[0.1, 0.2],
            local_metrics={"a": 1.0},
            global_metrics={"a": 0.8},
            curvature=[1.0, 1.0, 1.0],
            structure={"topology": 0.9, "resonance": 0.8},
        )
        assert "oscillation_strength" in result
        assert "phase_stability" in result
        assert "phase_lock_ratio" in result
        assert "multiscale_coherence" in result
        assert "scale_conflict" in result
        assert "system_energy" in result
        assert "control_alignment" in result
        assert "geometric_consistency" in result
        assert "signal_map" in result

    def test_partial_inputs(self) -> None:
        result = compute_physics_signals(history=[1.0, 2.0, 3.0])
        assert "oscillation_strength" in result
        assert "system_energy" not in result

    def test_none_inputs(self) -> None:
        result = compute_physics_signals()
        assert result == {}

    def test_deterministic(self) -> None:
        kwargs = {
            "history": [1.0, 3.0, 2.0, 4.0],
            "scales": [1.0, 2.0, 3.0],
            "metrics": {"variance": 0.5, "oscillation": 0.3},
            "deltas": [0.1, 0.2, 0.3],
        }
        assert compute_physics_signals(**kwargs) == compute_physics_signals(**kwargs)

    def test_no_nans(self) -> None:
        result = compute_physics_signals(
            history=[0.0, 0.0, 0.0],
            scales=[0.0, 0.0],
            metrics={"variance": 0.0, "oscillation": 0.0},
            deltas=[0.0],
            local_metrics={"a": 0.0},
            global_metrics={"a": 0.0},
            curvature=0.0,
        )
        for key, val in result.items():
            if isinstance(val, dict):
                for v in val.values():
                    assert not np.isnan(v), f"NaN in signal_map.{key}"
            else:
                assert not np.isnan(val), f"NaN in {key}"


# ---------------------------------------------------------------------------
# Cross-cutting: numpy array inputs
# ---------------------------------------------------------------------------


class TestNumpyInputs:
    def test_oscillation_numpy(self) -> None:
        arr = np.array([1.0, 3.0, 1.0, 3.0], dtype=np.float64)
        _assert_bounded(compute_oscillation_strength(arr))

    def test_scales_numpy(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _assert_bounded(compute_multiscale_coherence(arr))
        _assert_bounded(compute_scale_conflict(arr))

    def test_deltas_numpy(self) -> None:
        _assert_bounded(
            compute_system_energy(
                {"variance": 1.0},
                np.array([0.5, 0.5], dtype=np.float64),
            )
        )

    def test_curvature_numpy(self) -> None:
        _assert_bounded(
            compute_geometric_consistency(
                np.array([1.0, 2.0, 3.0], dtype=np.float64)
            )
        )
