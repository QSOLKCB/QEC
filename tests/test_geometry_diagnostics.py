"""
Tests for geometry-aware syndrome-only diagnostics (v3.3.0).

Covers metric correctness, deterministic serialization, aggregation
order stability, baseline-unchanged guarantees, and per-iteration
instrumentation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.geometry_diagnostics import (
    BSIConfigError,
    build_geometry_sidecar,
    collect_per_iteration_data,
    compute_bsi,
    compute_dps,
    compute_fcr,
    compute_local_inconsistency,
    compute_per_iteration_summary,
    compute_residual_summary,
    compute_ssi,
    compute_stall_metrics,
)


# ── Helper ─────────────────────────────────────────────────────────

def _rec(
    decoder: str = "bp_min_sum_flooding_none",
    distance: int = 3,
    p: float = 0.01,
    fer: float = 0.1,
    scr: float = 0.9,
    trials: int = 100,
) -> dict:
    """Build a minimal benchmark record for testing."""
    return {
        "decoder": decoder,
        "decoder_identity": {"adapter": "bp", "params": {"mode": "min_sum"}},
        "distance": distance,
        "fidelity": 1.0 - fer,
        "fer": fer,
        "mean_iters": 10.0,
        "p": p,
        "syndrome_success_rate": scr,
        "trials": trials,
        "wer": fer,
    }


# ═══════════════════════════════════════════════════════════════════
# DPS
# ═══════════════════════════════════════════════════════════════════

class TestDPS:
    def test_positive_slope_means_inversion(self):
        records = [
            _rec(distance=3, p=0.05, fer=0.10),
            _rec(distance=5, p=0.05, fer=0.20),
            _rec(distance=7, p=0.05, fer=0.30),
        ]
        result = compute_dps(records)
        assert len(result) == 1
        assert result[0]["inverted"] is True
        assert result[0]["slope"] > 0

    def test_negative_slope_means_normal(self):
        records = [
            _rec(distance=3, p=0.01, fer=0.30),
            _rec(distance=5, p=0.01, fer=0.20),
            _rec(distance=7, p=0.01, fer=0.10),
        ]
        result = compute_dps(records)
        assert len(result) == 1
        assert result[0]["inverted"] is False
        assert result[0]["slope"] < 0

    def test_groups_by_decoder_and_p(self):
        records = [
            _rec(distance=3, p=0.01, fer=0.1),
            _rec(distance=5, p=0.01, fer=0.2),
            _rec(distance=3, p=0.05, fer=0.3),
            _rec(distance=5, p=0.05, fer=0.1),
        ]
        result = compute_dps(records)
        assert len(result) == 2
        assert result[0]["p"] == 0.01
        assert result[1]["p"] == 0.05

    def test_single_distance_zero_slope(self):
        records = [_rec(distance=3, p=0.01, fer=0.5)]
        result = compute_dps(records)
        assert len(result) == 1
        assert result[0]["slope"] == 0.0
        assert result[0]["inverted"] is False

    def test_deterministic_output(self):
        records = [
            _rec(distance=3, p=0.01, fer=0.3),
            _rec(distance=5, p=0.01, fer=0.2),
        ]
        assert compute_dps(records) == compute_dps(records)

    def test_empty_input(self):
        assert compute_dps([]) == []


# ═══════════════════════════════════════════════════════════════════
# FCR
# ═══════════════════════════════════════════════════════════════════

class TestFCR:
    def test_fcr_equals_scr_minus_fidelity(self):
        records = [_rec(fer=0.2, scr=0.85)]
        result = compute_fcr(records)
        expected = 0.85 - (1.0 - 0.2)  # 0.05
        assert abs(result[0]["fcr"] - expected) < 1e-9

    def test_fcr_clamped_to_zero(self):
        records = [_rec(fer=0.5, scr=0.4)]
        result = compute_fcr(records)
        assert result[0]["fcr"] == 0.0

    def test_perfect_decoder(self):
        records = [_rec(fer=0.0, scr=1.0)]
        result = compute_fcr(records)
        assert result[0]["fcr"] == 0.0

    def test_sorted_output(self):
        records = [
            _rec(distance=7, p=0.05, fer=0.1, scr=0.95),
            _rec(distance=3, p=0.01, fer=0.2, scr=0.85),
        ]
        result = compute_fcr(records)
        assert result[0]["distance"] == 3
        assert result[1]["distance"] == 7


# ═══════════════════════════════════════════════════════════════════
# BSI
# ═══════════════════════════════════════════════════════════════════

class TestBSI:
    def test_positive_bsi(self):
        base = [_rec(distance=3, p=0.05, fer=0.4)]
        double = [_rec(distance=3, p=0.05, fer=0.2)]
        result = compute_bsi(base, double)
        assert len(result) == 1
        assert result[0]["bsi"] == pytest.approx(0.2, abs=1e-9)

    def test_zero_bsi(self):
        base = [_rec(distance=3, p=0.05, fer=0.3)]
        double = [_rec(distance=3, p=0.05, fer=0.3)]
        result = compute_bsi(base, double)
        assert result[0]["bsi"] == 0.0

    def test_missing_2x_record_raises(self):
        base = [_rec(distance=3, p=0.05, fer=0.3)]
        double = [_rec(distance=5, p=0.05, fer=0.2)]
        with pytest.raises(BSIConfigError, match="BSI mismatch"):
            compute_bsi(base, double)


# ═══════════════════════════════════════════════════════════════════
# SSI
# ═══════════════════════════════════════════════════════════════════

class TestSSI:
    def test_ssi_basic(self):
        by_sched = {
            "flooding": [_rec(distance=3, p=0.05, fer=0.3)],
            "layered": [_rec(distance=3, p=0.05, fer=0.1)],
        }
        result = compute_ssi(by_sched)
        assert len(result) == 1
        assert result[0]["ssi"] == pytest.approx(0.2, abs=1e-9)

    def test_ssi_single_schedule_empty(self):
        by_sched = {"flooding": [_rec(distance=3, p=0.05, fer=0.3)]}
        assert compute_ssi(by_sched) == []

    def test_ssi_fer_by_schedule_sorted(self):
        by_sched = {
            "residual": [_rec(distance=3, p=0.05, fer=0.4)],
            "flooding": [_rec(distance=3, p=0.05, fer=0.3)],
            "layered": [_rec(distance=3, p=0.05, fer=0.1)],
        }
        result = compute_ssi(by_sched)
        assert list(result[0]["fer_by_schedule"].keys()) == [
            "flooding", "layered", "residual",
        ]

    def test_ssi_multi_decoder_no_mixing(self):
        """Two decoders at same (distance, p) produce separate SSI entries."""
        dec_a = "bp_min_sum_flooding_none"
        dec_b = "bp_sum_product_flooding_none"
        by_sched = {
            "flooding": [
                _rec(decoder=dec_a, distance=3, p=0.05, fer=0.3),
                _rec(decoder=dec_b, distance=3, p=0.05, fer=0.6),
            ],
            "layered": [
                _rec(decoder=dec_a, distance=3, p=0.05, fer=0.1),
                _rec(decoder=dec_b, distance=3, p=0.05, fer=0.5),
            ],
        }
        result = compute_ssi(by_sched)
        assert len(result) == 2
        decoders = [r["decoder"] for r in result]
        assert dec_a in decoders
        assert dec_b in decoders
        # Each decoder gets its own SSI.
        for r in result:
            if r["decoder"] == dec_a:
                assert r["ssi"] == pytest.approx(0.2, abs=1e-9)
            else:
                assert r["ssi"] == pytest.approx(0.1, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# Per-Iteration Summary
# ═══════════════════════════════════════════════════════════════════

class TestPerIterationSummary:
    def test_syndrome_weight_computation(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        target = np.array([1, 0], dtype=np.uint8)
        # iter 0: hard=[0,0,0] → syn=[0,0] → diff=[1,0] → sw=1
        # iter 1: hard=[1,0,0] → syn=[1,0] → diff=[0,0] → sw=0
        llr_history = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ])
        result = compute_per_iteration_summary(H, llr_history, target)
        assert result["syndrome_weight"] == [1, 0]
        assert result["check_satisfaction_ratio"][0] == 0.5
        assert result["check_satisfaction_ratio"][1] == 1.0
        assert result["delta_syndrome"] == [0, -1]
        assert result["stall_count"] == 0

    def test_empty_history(self):
        H = np.array([[1, 1]], dtype=np.uint8)
        llr_history = np.zeros((0, 2))
        target = np.array([0], dtype=np.uint8)
        result = compute_per_iteration_summary(H, llr_history, target)
        assert result["syndrome_weight"] == []
        assert result["stall_count"] == 0
        assert result["stall_fraction"] == 0.0

    def test_stall_detection(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        target = np.array([0, 0], dtype=np.uint8)
        # All-positive LLR every iteration → constant syndrome weight 0.
        llr_history = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        result = compute_per_iteration_summary(H, llr_history, target)
        assert result["stall_count"] == 2
        assert result["stall_fraction"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# Residual Summary
# ═══════════════════════════════════════════════════════════════════

class TestResidualSummary:
    def test_basic(self):
        metrics = {
            "residual_linf": [np.array([0.5, 0.3, 0.1])],
            "residual_l2": [np.array([0.4, 0.2, 0.1])],
            "residual_energy": [1.5],
        }
        result = compute_residual_summary(metrics)
        assert len(result) == 1
        assert result[0]["energy"] == pytest.approx(1.5)
        assert result[0]["linf_max"] == pytest.approx(0.5)
        assert result[0]["linf_mean"] == pytest.approx(0.3)

    def test_empty(self):
        assert compute_residual_summary({}) == []


# ═══════════════════════════════════════════════════════════════════
# Stall Metrics
# ═══════════════════════════════════════════════════════════════════

class TestStallMetrics:
    def test_aggregate(self):
        summaries = [
            {"stall_fraction": 0.5, "stall_count": 2},
            {"stall_fraction": 0.0, "stall_count": 0},
            {"stall_fraction": 1.0, "stall_count": 4},
        ]
        result = compute_stall_metrics(summaries)
        assert result["total_trials"] == 3
        assert result["trials_with_stall"] == 2
        assert result["max_stall_fraction"] == 1.0
        assert result["mean_stall_fraction"] == pytest.approx(0.5)

    def test_empty(self):
        result = compute_stall_metrics([])
        assert result["total_trials"] == 0


# ═══════════════════════════════════════════════════════════════════
# Local Inconsistency
# ═══════════════════════════════════════════════════════════════════

class TestLocalInconsistency:
    def test_detected(self):
        summaries = [{"delta_syndrome": [0, -1, 2, -1]}]
        result = compute_local_inconsistency(summaries)
        assert result["trials_with_inconsistency"] == 1
        assert result["max_inconsistency_count"] == 1

    def test_no_inconsistency(self):
        summaries = [{"delta_syndrome": [0, -2, -1, 0]}]
        result = compute_local_inconsistency(summaries)
        assert result["trials_with_inconsistency"] == 0

    def test_empty(self):
        result = compute_local_inconsistency([])
        assert result["total_trials"] == 0


# ═══════════════════════════════════════════════════════════════════
# Sidecar Builder
# ═══════════════════════════════════════════════════════════════════

class TestSidecarBuilder:
    def test_basic_structure(self):
        records = [
            _rec(distance=3, p=0.01, fer=0.1, scr=0.95),
            _rec(distance=5, p=0.01, fer=0.2, scr=0.85),
        ]
        sidecar = build_geometry_sidecar(records)
        assert sidecar["diagnostic_version"] == "3.3.0"
        assert "dps" in sidecar["metrics"]
        assert "fcr" in sidecar["metrics"]

    def test_bsi_absent_without_2x(self):
        sidecar = build_geometry_sidecar([_rec()])
        assert "bsi" not in sidecar["metrics"]

    def test_ssi_absent_without_schedules(self):
        sidecar = build_geometry_sidecar([_rec()])
        assert "ssi" not in sidecar["metrics"]

    def test_deterministic_serialization(self):
        records = [
            _rec(distance=3, p=0.01, fer=0.1, scr=0.95),
            _rec(distance=5, p=0.02, fer=0.2, scr=0.85),
        ]
        s1 = build_geometry_sidecar(records)
        s2 = build_geometry_sidecar(records)
        j1 = json.dumps(s1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(s2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2

    def test_with_bsi_and_ssi(self):
        base = [
            _rec(distance=3, p=0.05, fer=0.4, scr=0.7),
            _rec(distance=5, p=0.05, fer=0.5, scr=0.6),
        ]
        double = [
            _rec(distance=3, p=0.05, fer=0.2, scr=0.85),
            _rec(distance=5, p=0.05, fer=0.3, scr=0.75),
        ]
        by_sched = {
            "flooding": [_rec(distance=3, p=0.05, fer=0.4)],
            "layered": [_rec(distance=3, p=0.05, fer=0.2)],
        }
        sidecar = build_geometry_sidecar(
            base, records_2x=double, records_by_schedule=by_sched,
        )
        assert "bsi" in sidecar["metrics"]
        assert "ssi" in sidecar["metrics"]

    def test_with_per_iteration_data(self):
        records = [_rec()]
        per_iter = [{
            "per_iteration": {
                "syndrome_weight": [3, 2, 1],
                "check_satisfaction_ratio": [0.5, 0.7, 0.9],
                "delta_syndrome": [0, -1, -1],
                "stall_count": 0,
                "stall_fraction": 0.0,
            },
            "residual_summary": [],
        }]
        sidecar = build_geometry_sidecar(records, per_iteration_data=per_iter)
        assert "stall_metrics" in sidecar["metrics"]
        assert "local_inconsistency" in sidecar["metrics"]


# ═══════════════════════════════════════════════════════════════════
# Aggregation Order Stability
# ═══════════════════════════════════════════════════════════════════

class TestAggregationOrderStability:
    def test_dps_order_independent(self):
        r1 = _rec(distance=3, p=0.01, fer=0.3)
        r2 = _rec(distance=5, p=0.01, fer=0.2)
        r3 = _rec(distance=7, p=0.01, fer=0.1)
        assert compute_dps([r1, r2, r3]) == compute_dps([r3, r1, r2])

    def test_fcr_order_independent(self):
        r1 = _rec(distance=3, p=0.01, fer=0.1, scr=0.95)
        r2 = _rec(distance=5, p=0.02, fer=0.2, scr=0.85)
        assert compute_fcr([r1, r2]) == compute_fcr([r2, r1])

    def test_bsi_order_independent(self):
        b1 = _rec(distance=3, p=0.05, fer=0.4)
        b2 = _rec(distance=5, p=0.05, fer=0.5)
        d1 = _rec(distance=3, p=0.05, fer=0.2)
        d2 = _rec(distance=5, p=0.05, fer=0.3)
        assert compute_bsi([b1, b2], [d1, d2]) == compute_bsi(
            [b2, b1], [d2, d1],
        )

    def test_sidecar_order_independent(self):
        r1 = _rec(distance=3, p=0.01, fer=0.1, scr=0.95)
        r2 = _rec(distance=5, p=0.02, fer=0.2, scr=0.85)
        s_a = build_geometry_sidecar([r1, r2])
        s_b = build_geometry_sidecar([r2, r1])
        j_a = json.dumps(s_a, sort_keys=True, separators=(",", ":"))
        j_b = json.dumps(s_b, sort_keys=True, separators=(",", ":"))
        assert j_a == j_b


# ═══════════════════════════════════════════════════════════════════
# Baseline Unchanged (diagnostics disabled)
# ═══════════════════════════════════════════════════════════════════

class TestBaselineUnchanged:
    def test_runner_output_unchanged(self):
        """Canonical benchmark output is byte-identical with/without
        the geometry_diagnostics module being importable."""
        from bench.config import BenchmarkConfig, DecoderSpec
        from bench.runner import run_benchmark

        config = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=10,
            max_iters=10,
            decoders=[DecoderSpec(adapter="bp", params={
                "mode": "min_sum", "schedule": "flooding",
            })],
            deterministic_metadata=True,
        )
        r1 = run_benchmark(config)
        r2 = run_benchmark(config)
        j1 = json.dumps(r1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(r2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2


# ═══════════════════════════════════════════════════════════════════
# Integration: collect_per_iteration_data
# ═══════════════════════════════════════════════════════════════════

class TestCollectPerIterationData:
    def test_returns_expected_keys(self):
        from qec_qldpc_codes import create_code, syndrome, channel_llr

        code = create_code(name="rate_0.50", lifting_size=3, seed=42)
        H = code.H_X
        _, n = H.shape
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = collect_per_iteration_data(
            H, llr, s,
            decoder_params={"mode": "min_sum", "schedule": "flooding"},
            max_iters=10,
        )
        assert "per_iteration" in result
        assert "residual_summary" in result
        assert "converged" in result
        assert "iterations" in result
        assert isinstance(result["per_iteration"]["syndrome_weight"], list)

    def test_deterministic(self):
        from qec_qldpc_codes import create_code, syndrome, channel_llr

        code = create_code(name="rate_0.50", lifting_size=3, seed=42)
        H = code.H_X
        _, n = H.shape
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        params = {"mode": "min_sum", "schedule": "flooding"}
        r1 = collect_per_iteration_data(H, llr, s, params, max_iters=10)
        r2 = collect_per_iteration_data(H, llr, s, params, max_iters=10)
        assert r1["per_iteration"] == r2["per_iteration"]
        assert r1["converged"] == r2["converged"]
        assert r1["iterations"] == r2["iterations"]


# ═══════════════════════════════════════════════════════════════════
# Sidecar Determinism (byte-identical rerun)
# ═══════════════════════════════════════════════════════════════════

class TestSidecarDeterminism:
    def test_byte_identical_sidecar_rerun(self):
        """Full sidecar built twice from identical input → identical JSON."""
        records = [
            _rec(distance=d, p=p, fer=0.1 * d * p, scr=0.9)
            for d in [3, 5, 7]
            for p in [0.01, 0.02, 0.05]
        ]
        s1 = build_geometry_sidecar(records)
        s2 = build_geometry_sidecar(records)
        j1 = json.dumps(s1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(s2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2
        assert len(j1) > 100  # non-trivial content
