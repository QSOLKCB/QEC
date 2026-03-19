"""
Tests for BP phase diagram analysis (v4.6.0).

Validates:
  - Determinism (identical outputs on repeated runs)
  - Correct aggregation of synthetic regime trace data
  - Empty input handling
  - Bench integration smoke test
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest

from qec.diagnostics.bp_phase_diagram import (
    compute_bp_phase_diagram,
    DEFAULT_METASTABLE_THRESHOLD,
    RegimeTraceSummary,
    RegimeTraceResult,
    RunResult,
    BpPhaseDiagram,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_regime_trace_result(
    regime_trace: list[str],
    freeze_score: float = 0.3,
    switch_rate: float = 0.1,
    max_dwell: int = 10,
    num_events: int = 1,
) -> dict:
    """Build a synthetic per-trial regime trace result."""
    return {
        "regime_trace": regime_trace,
        "transitions": [],
        "dwell_times": {},
        "transition_counts": {},
        "summary": {
            "freeze_score": float(freeze_score),
            "switch_rate": float(switch_rate),
            "max_dwell": int(max_dwell),
            "num_events": int(num_events),
        },
    }


def _make_run_result(
    distance: int,
    noise: float,
    traces: list[dict],
) -> dict:
    """Build a synthetic run result entry."""
    return {
        "distance": distance,
        "noise": noise,
        "regime_trace_results": traces,
    }


# ── Test: Empty input ────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_list(self):
        result = compute_bp_phase_diagram([])
        assert result["phase_points"] == []
        assert result["distance_levels"] == []
        assert result["noise_levels"] == []
        assert result["phase_statistics"]["total_runs"] == 0
        assert result["phase_statistics"]["total_parameter_points"] == 0

    def test_empty_traces_per_point(self):
        entry = _make_run_result(distance=7, noise=0.02, traces=[])
        result = compute_bp_phase_diagram([entry])
        assert len(result["phase_points"]) == 1
        pt = result["phase_points"][0]
        assert pt["distance"] == 7
        assert pt["noise"] == 0.02
        assert pt["num_runs"] == 0
        assert pt["metastable_probability"] == 0.0
        assert pt["mean_freeze_score"] == 0.0


# ── Test: Determinism ────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_outputs(self):
        """Repeated calls with identical inputs produce byte-identical JSON."""
        traces = [
            _make_regime_trace_result(
                ["stable_convergence"] * 20,
                freeze_score=0.8,
                switch_rate=0.0,
                max_dwell=20,
                num_events=0,
            ),
            _make_regime_trace_result(
                ["metastable_state"] * 15 + ["stable_convergence"] * 5,
                freeze_score=0.6,
                switch_rate=0.05,
                max_dwell=15,
                num_events=1,
            ),
        ]
        run_results = [
            _make_run_result(7, 0.02, traces),
            _make_run_result(15, 0.02, traces),
        ]

        r1 = compute_bp_phase_diagram(run_results)
        r2 = compute_bp_phase_diagram(run_results)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_input_order_invariance(self):
        """Results should be the same regardless of input order."""
        traces = [
            _make_regime_trace_result(
                ["stable_convergence"] * 10,
                freeze_score=0.4,
            ),
        ]
        entry_a = _make_run_result(7, 0.01, traces)
        entry_b = _make_run_result(15, 0.02, traces)

        r1 = compute_bp_phase_diagram([entry_a, entry_b])
        r2 = compute_bp_phase_diagram([entry_b, entry_a])

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2


# ── Test: Correct aggregation ────────────────────────────────────────


class TestAggregation:
    def test_metastable_probability(self):
        """Metastable probability = fraction with freeze_score > threshold."""
        traces = [
            _make_regime_trace_result(["stable_convergence"] * 10, freeze_score=0.3),
            _make_regime_trace_result(["stable_convergence"] * 10, freeze_score=0.6),
            _make_regime_trace_result(["stable_convergence"] * 10, freeze_score=0.8),
            _make_regime_trace_result(["stable_convergence"] * 10, freeze_score=0.4),
        ]
        entry = _make_run_result(7, 0.02, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        # 2 out of 4 have freeze_score > 0.5
        assert pt["metastable_probability"] == pytest.approx(0.5)

    def test_mean_freeze_score(self):
        """Mean freeze score is correct average."""
        traces = [
            _make_regime_trace_result(["s"] * 5, freeze_score=0.2),
            _make_regime_trace_result(["s"] * 5, freeze_score=0.4),
            _make_regime_trace_result(["s"] * 5, freeze_score=0.6),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        assert pt["mean_freeze_score"] == pytest.approx(0.4)

    def test_mean_switch_rate(self):
        """Mean switch rate is correct average."""
        traces = [
            _make_regime_trace_result(["s"] * 5, switch_rate=0.1),
            _make_regime_trace_result(["s"] * 5, switch_rate=0.3),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        assert pt["mean_switch_rate"] == pytest.approx(0.2)

    def test_mean_max_dwell(self):
        """Mean max dwell is correct average."""
        traces = [
            _make_regime_trace_result(["s"] * 5, max_dwell=10),
            _make_regime_trace_result(["s"] * 5, max_dwell=20),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        assert pt["mean_max_dwell"] == pytest.approx(15.0)

    def test_event_rate(self):
        """Event rate = mean num_events."""
        traces = [
            _make_regime_trace_result(["s"] * 5, num_events=2),
            _make_regime_trace_result(["s"] * 5, num_events=4),
            _make_regime_trace_result(["s"] * 5, num_events=0),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        assert pt["event_rate"] == pytest.approx(2.0)

    def test_regime_frequencies(self):
        """Regime frequencies are correctly normalized."""
        traces = [
            _make_regime_trace_result(
                ["stable_convergence"] * 8 + ["metastable_state"] * 2,
            ),
            _make_regime_trace_result(
                ["stable_convergence"] * 6 + ["metastable_state"] * 4,
            ),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        freqs = pt["regime_frequencies"]
        # Total labels: 20, stable=14, metastable=6
        assert freqs["stable_convergence"] == pytest.approx(14.0 / 20.0)
        assert freqs["metastable_state"] == pytest.approx(6.0 / 20.0)
        # Keys are sorted
        keys = list(freqs.keys())
        assert keys == sorted(keys)

    def test_custom_metastable_threshold(self):
        """Custom threshold changes metastable probability."""
        traces = [
            _make_regime_trace_result(["s"] * 5, freeze_score=0.3),
            _make_regime_trace_result(["s"] * 5, freeze_score=0.6),
        ]
        entry = _make_run_result(7, 0.01, traces)

        # Default threshold 0.5: 1/2 metastable
        r1 = compute_bp_phase_diagram([entry])
        assert r1["phase_points"][0]["metastable_probability"] == pytest.approx(0.5)

        # Threshold 0.2: 2/2 metastable
        r2 = compute_bp_phase_diagram([entry], metastable_threshold=0.2)
        assert r2["phase_points"][0]["metastable_probability"] == pytest.approx(1.0)

        # Threshold 0.7: 0/2 metastable
        r3 = compute_bp_phase_diagram([entry], metastable_threshold=0.7)
        assert r3["phase_points"][0]["metastable_probability"] == pytest.approx(0.0)


# ── Test: Multiple parameter points ──────────────────────────────────


class TestMultiplePoints:
    def test_distance_and_noise_levels(self):
        """Phase points are produced for each unique (distance, noise) pair."""
        traces = [_make_regime_trace_result(["s"] * 5)]
        entries = [
            _make_run_result(7, 0.01, traces),
            _make_run_result(7, 0.02, traces),
            _make_run_result(15, 0.01, traces),
            _make_run_result(15, 0.02, traces),
        ]
        result = compute_bp_phase_diagram(entries)

        assert result["distance_levels"] == [7, 15]
        assert result["noise_levels"] == [0.01, 0.02]
        assert len(result["phase_points"]) == 4
        assert result["phase_statistics"]["total_parameter_points"] == 4

    def test_sorted_output_order(self):
        """Phase points are sorted by (distance, noise)."""
        traces = [_make_regime_trace_result(["s"] * 5)]
        entries = [
            _make_run_result(15, 0.02, traces),
            _make_run_result(7, 0.01, traces),
            _make_run_result(15, 0.01, traces),
            _make_run_result(7, 0.02, traces),
        ]
        result = compute_bp_phase_diagram(entries)

        coords = [(pt["distance"], pt["noise"]) for pt in result["phase_points"]]
        assert coords == [(7, 0.01), (7, 0.02), (15, 0.01), (15, 0.02)]

    def test_multiple_entries_same_point(self):
        """Multiple entries for same (distance, noise) are merged."""
        traces_a = [
            _make_regime_trace_result(["s"] * 5, freeze_score=0.2),
        ]
        traces_b = [
            _make_regime_trace_result(["s"] * 5, freeze_score=0.8),
        ]
        entries = [
            _make_run_result(7, 0.01, traces_a),
            _make_run_result(7, 0.01, traces_b),
        ]
        result = compute_bp_phase_diagram(entries)

        assert len(result["phase_points"]) == 1
        pt = result["phase_points"][0]
        assert pt["num_runs"] == 2
        assert pt["mean_freeze_score"] == pytest.approx(0.5)


# ── Test: JSON serialization ─────────────────────────────────────────


class TestSerialization:
    def test_json_serializable(self):
        """Output must be fully JSON-serializable."""
        traces = [
            _make_regime_trace_result(
                ["stable_convergence"] * 10,
                freeze_score=0.7,
                switch_rate=0.05,
                max_dwell=10,
                num_events=1,
            ),
        ]
        entries = [
            _make_run_result(7, 0.02, traces),
            _make_run_result(15, 0.02, traces),
        ]
        result = compute_bp_phase_diagram(entries)
        # Must not raise
        serialized = json.dumps(result, sort_keys=True)
        roundtrip = json.loads(serialized)
        assert roundtrip == result

    def test_stable_key_ordering(self):
        """Regime frequencies must have lexicographically sorted keys."""
        traces = [
            _make_regime_trace_result(
                ["chaotic_behavior"] * 3 + ["stable_convergence"] * 2
                + ["metastable_state"] * 5,
            ),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        freqs = result["phase_points"][0]["regime_frequencies"]
        keys = list(freqs.keys())
        assert keys == sorted(keys)


# ── Test: Bench integration smoke test ───────────────────────────────


# ── Test: Empty regime traces edge case ───────────────────────────────


class TestEmptyRegimeTraces:
    def test_empty_regime_traces(self):
        """Runs with empty regime_trace but valid summary fields."""
        traces = [
            _make_regime_trace_result([], freeze_score=0.2),
            _make_regime_trace_result([], freeze_score=0.4),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        assert pt["regime_frequencies"] == {}
        assert pt["mean_freeze_score"] == pytest.approx(0.3)
        assert pt["num_runs"] == 2
        assert pt["metastable_probability"] == pytest.approx(0.0)

    def test_mixed_empty_and_nonempty_traces(self):
        """Mix of empty and non-empty regime traces."""
        traces = [
            _make_regime_trace_result([], freeze_score=0.2),
            _make_regime_trace_result(
                ["stable_convergence"] * 10, freeze_score=0.6,
            ),
        ]
        entry = _make_run_result(7, 0.01, traces)
        result = compute_bp_phase_diagram([entry])

        pt = result["phase_points"][0]
        # Only 10 labels from the non-empty trace.
        assert pt["regime_frequencies"] == {"stable_convergence": 1.0}
        assert pt["mean_freeze_score"] == pytest.approx(0.4)


# ── Test: Input validation ────────────────────────────────────────────


class TestInputValidation:
    def test_not_a_sequence(self):
        with pytest.raises(TypeError, match="must be a sequence"):
            compute_bp_phase_diagram(42)  # type: ignore[arg-type]

    def test_missing_distance(self):
        with pytest.raises(ValueError, match="missing required key 'distance' or 'noise'"):
            compute_bp_phase_diagram([{"noise": 0.01, "regime_trace_results": []}])

    def test_missing_noise(self):
        with pytest.raises(ValueError, match="missing required key 'distance' or 'noise'"):
            compute_bp_phase_diagram([{"distance": 7, "regime_trace_results": []}])

    def test_missing_regime_trace_results(self):
        with pytest.raises(ValueError, match="missing required key 'regime_trace_results'"):
            compute_bp_phase_diagram([{"distance": 7, "noise": 0.01}])

    def test_missing_summary_in_trace(self):
        entry = {
            "distance": 7,
            "noise": 0.01,
            "regime_trace_results": [{"regime_trace": ["s"]}],
        }
        with pytest.raises(ValueError, match="missing 'summary'"):
            compute_bp_phase_diagram([entry])

    def test_missing_regime_trace_in_trace(self):
        entry = {
            "distance": 7,
            "noise": 0.01,
            "regime_trace_results": [{"summary": {"freeze_score": 0.1, "switch_rate": 0.0, "max_dwell": 1, "num_events": 0}}],
        }
        with pytest.raises(ValueError, match="missing 'regime_trace'"):
            compute_bp_phase_diagram([entry])

    def test_missing_summary_key(self):
        entry = {
            "distance": 7,
            "noise": 0.01,
            "regime_trace_results": [{
                "regime_trace": ["s"],
                "summary": {"freeze_score": 0.1},  # missing switch_rate, max_dwell, num_events
            }],
        }
        with pytest.raises(ValueError, match="missing 'switch_rate'"):
            compute_bp_phase_diagram([entry])


# ── Test: Type exports ────────────────────────────────────────────────


class TestTypeExports:
    def test_typed_dict_exports(self):
        """TypedDict types are importable."""
        assert RegimeTraceSummary is not None
        assert RegimeTraceResult is not None
        assert RunResult is not None
        assert BpPhaseDiagram is not None


# ── Test: Bench integration smoke test ───────────────────────────────


class TestBenchIntegration:
    def test_run_mode_with_phase_diagram_flag(self):
        """Smoke test: run_mode accepts enable_bp_phase_diagram parameter."""
        from bench.dps_v381_eval import run_mode, _pre_generate_instances
        from qec_qldpc_codes import create_code, syndrome, channel_llr

        seed = 42
        distance = 3
        p = 0.01
        trials = 5
        max_iters = 10

        rng = np.random.default_rng(seed)
        code = create_code(name="rate_0.50", lifting_size=distance, seed=seed)
        H = code.H_X
        instances = _pre_generate_instances(H, p, trials, rng)

        result = run_mode(
            "baseline", H, instances,
            max_iters=max_iters,
            enable_bp_phase_diagram=True,
        )

        # Phase diagram flag implies bp_transitions, so regime traces should exist.
        assert "bp_regime_trace" in result
        assert "bp_transition_summary" in result

    def test_compute_phase_diagram_from_bench_results(self):
        """Smoke test: compute_bp_phase_diagram works with bench-style data."""
        from bench.dps_v381_eval import run_mode, _pre_generate_instances
        from qec_qldpc_codes import create_code

        seed = 42
        distance = 3
        p = 0.01
        trials = 5
        max_iters = 10

        rng = np.random.default_rng(seed)
        code = create_code(name="rate_0.50", lifting_size=distance, seed=seed)
        H = code.H_X
        instances = _pre_generate_instances(H, p, trials, rng)

        result = run_mode(
            "baseline", H, instances,
            max_iters=max_iters,
            enable_bp_phase_diagram=True,
        )

        # Build phase diagram input from results.
        run_results = [{
            "distance": distance,
            "noise": p,
            "regime_trace_results": result["bp_regime_trace"],
        }]
        diagram = compute_bp_phase_diagram(run_results)

        assert len(diagram["phase_points"]) == 1
        pt = diagram["phase_points"][0]
        assert pt["distance"] == distance
        assert pt["noise"] == p
        assert pt["num_runs"] == trials
        assert 0.0 <= pt["metastable_probability"] <= 1.0
        assert pt["mean_freeze_score"] >= 0.0

        # JSON serializable
        serialized = json.dumps(diagram, sort_keys=True)
        assert json.loads(serialized) == diagram
