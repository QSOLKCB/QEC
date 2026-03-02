"""
Tests for v3.8.1 DPS evaluation harness.

Verifies:
- Harness runs without modifying decoder
- Modes dispatch correctly (schedule and RPC configuration)
- RPC row counts change when enabled
- Determinism holds (identical inputs → identical outputs)

Uses reduced grid for fast test execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec_qldpc_codes import bp_decode, syndrome, create_code
from src.qec.channel import get_channel_model
from src.qec.decoder.rpc import RPCConfig, StructuralConfig, build_rpc_augmented_system

# Import harness functions
from bench.dps_v381_eval import (
    MODES,
    SEED,
    MODE,
    MAX_ITERS,
    CHANNEL,
    pregenerate_instances,
    evaluate_mode,
    run_determinism_check,
)


# ── Reduced test grid ─────────────────────────────────────────────

TEST_DISTANCES = [8, 12]
TEST_P_VALUES = [0.01, 0.02]
TEST_TRIALS = 30


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_instances():
    """Pre-generate instances for the reduced test grid."""
    return pregenerate_instances(
        TEST_DISTANCES, TEST_P_VALUES, TEST_TRIALS, SEED,
    )


# ══════════════════════════════════════════════════════════════════
#  1. Harness does not modify decoder
# ══════════════════════════════════════════════════════════════════

class TestHarnessNoDecoderModification:
    """Confirm the harness uses bp_decode as-is with no modification."""

    def test_baseline_uses_bp_decode_directly(self, test_instances):
        """Baseline evaluation calls bp_decode and returns valid records."""
        records, audit = evaluate_mode(
            "baseline", MODES["baseline"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        # Should have one record per (distance, p) pair
        assert len(records) == len(TEST_DISTANCES) * len(TEST_P_VALUES)
        for rec in records:
            assert "fer" in rec
            assert "mean_iters" in rec
            assert 0.0 <= rec["fer"] <= 1.0

    def test_bp_decode_returns_same_before_and_after_harness(self):
        """bp_decode gives identical results before/after harness runs."""
        code = create_code("rate_0.50", lifting_size=8, seed=SEED)
        H = code.H_X
        n = H.shape[1]

        rng = np.random.default_rng(12345)
        e = (rng.random(n) < 0.02).astype(np.uint8)
        s = syndrome(H, e)
        channel = get_channel_model(CHANNEL)
        llr = channel.compute_llr(p=0.02, n=n, error_vector=e)

        # Run before
        r_before = bp_decode(
            H, llr, max_iters=MAX_ITERS, mode=MODE,
            schedule="flooding", syndrome_vec=s,
        )

        # Run a mini harness evaluation
        instances = pregenerate_instances([8], [0.02], 10, SEED)
        evaluate_mode(
            "baseline", MODES["baseline"],
            distances=[8], p_values=[0.02],
            trials=10, instances=instances, seed=SEED,
        )

        # Run after
        r_after = bp_decode(
            H, llr, max_iters=MAX_ITERS, mode=MODE,
            schedule="flooding", syndrome_vec=s,
        )

        np.testing.assert_array_equal(r_before[0], r_after[0])
        assert r_before[1] == r_after[1]


# ══════════════════════════════════════════════════════════════════
#  2. Modes dispatch correctly
# ══════════════════════════════════════════════════════════════════

class TestModeDispatch:
    """Verify each mode uses the correct schedule and structural config."""

    def test_baseline_schedule_is_flooding(self, test_instances):
        """Baseline audit reports schedule='flooding', rpc disabled."""
        _, audit = evaluate_mode(
            "baseline", MODES["baseline"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["schedule"] == "flooding"
            assert entry["rpc_enabled"] is False

    def test_rpc_only_schedule_is_flooding_rpc_enabled(self, test_instances):
        """rpc_only audit reports schedule='flooding', rpc enabled."""
        _, audit = evaluate_mode(
            "rpc_only", MODES["rpc_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["schedule"] == "flooding"
            assert entry["rpc_enabled"] is True

    def test_geom_v1_only_schedule(self, test_instances):
        """geom_v1_only audit reports schedule='geom_v1', rpc disabled."""
        _, audit = evaluate_mode(
            "geom_v1_only", MODES["geom_v1_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["schedule"] == "geom_v1"
            assert entry["rpc_enabled"] is False

    def test_rpc_geom_schedule(self, test_instances):
        """rpc_geom audit reports schedule='geom_v1', rpc enabled."""
        _, audit = evaluate_mode(
            "rpc_geom", MODES["rpc_geom"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["schedule"] == "geom_v1"
            assert entry["rpc_enabled"] is True

    def test_exactly_four_modes(self):
        """Harness defines exactly four modes."""
        assert len(MODES) == 4
        assert set(MODES.keys()) == {
            "baseline", "rpc_only", "geom_v1_only", "rpc_geom",
        }


# ══════════════════════════════════════════════════════════════════
#  3. RPC row counts change when enabled
# ══════════════════════════════════════════════════════════════════

class TestRPCRowCounts:
    """Verify RPC augmentation produces additional rows."""

    def test_baseline_has_zero_added_rows(self, test_instances):
        """Baseline mode: added_rows == 0 for all entries."""
        _, audit = evaluate_mode(
            "baseline", MODES["baseline"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["added_rows"] == 0

    def test_rpc_has_positive_added_rows(self, test_instances):
        """RPC-enabled modes: added_rows > 0 for all entries."""
        _, audit = evaluate_mode(
            "rpc_only", MODES["rpc_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["added_rows"] > 0, (
                f"Expected added_rows > 0 for rpc_only at "
                f"d={entry['distance']}, p={entry['p']}"
            )

    def test_augmented_rows_gt_original(self, test_instances):
        """When RPC enabled, augmented_rows > original_rows."""
        _, audit = evaluate_mode(
            "rpc_geom", MODES["rpc_geom"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for entry in audit:
            assert entry["augmented_rows"] > entry["original_rows"]

    def test_rpc_augmentation_deterministic(self):
        """RPC augmentation is deterministic for same inputs."""
        code = create_code("rate_0.50", lifting_size=8, seed=SEED)
        H = code.H_X
        rng = np.random.default_rng(999)
        n = H.shape[1]
        e = (rng.random(n) < 0.02).astype(np.uint8)
        s = syndrome(H, e)

        rpc_cfg = RPCConfig(enabled=True, max_rows=64, w_min=2, w_max=32)

        H1, s1 = build_rpc_augmented_system(H, s, rpc_cfg)
        H2, s2 = build_rpc_augmented_system(H, s, rpc_cfg)

        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(s1, s2)


# ══════════════════════════════════════════════════════════════════
#  4. Determinism holds
# ══════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Two identical evaluations produce identical results."""

    def test_baseline_determinism(self, test_instances):
        """Baseline mode produces identical FER across two runs."""
        rec_a, _ = evaluate_mode(
            "baseline", MODES["baseline"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        rec_b, _ = evaluate_mode(
            "baseline", MODES["baseline"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for ra, rb in zip(rec_a, rec_b):
            assert ra["fer"] == rb["fer"], (
                f"FER mismatch at d={ra['distance']}, p={ra['p']}"
            )
            assert ra["mean_iters"] == rb["mean_iters"]

    def test_rpc_only_determinism(self, test_instances):
        """rpc_only mode produces identical FER across two runs."""
        rec_a, _ = evaluate_mode(
            "rpc_only", MODES["rpc_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        rec_b, _ = evaluate_mode(
            "rpc_only", MODES["rpc_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for ra, rb in zip(rec_a, rec_b):
            assert ra["fer"] == rb["fer"]
            assert ra["mean_iters"] == rb["mean_iters"]

    def test_geom_v1_only_determinism(self, test_instances):
        """geom_v1_only mode produces identical FER across two runs."""
        rec_a, _ = evaluate_mode(
            "geom_v1_only", MODES["geom_v1_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        rec_b, _ = evaluate_mode(
            "geom_v1_only", MODES["geom_v1_only"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for ra, rb in zip(rec_a, rec_b):
            assert ra["fer"] == rb["fer"]
            assert ra["mean_iters"] == rb["mean_iters"]

    def test_rpc_geom_determinism(self, test_instances):
        """rpc_geom mode produces identical FER across two runs."""
        rec_a, _ = evaluate_mode(
            "rpc_geom", MODES["rpc_geom"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        rec_b, _ = evaluate_mode(
            "rpc_geom", MODES["rpc_geom"],
            distances=TEST_DISTANCES, p_values=TEST_P_VALUES,
            trials=TEST_TRIALS, instances=test_instances, seed=SEED,
        )
        for ra, rb in zip(rec_a, rec_b):
            assert ra["fer"] == rb["fer"]
            assert ra["mean_iters"] == rb["mean_iters"]

    def test_instances_reuse_determinism(self):
        """Pre-generated instances produce identical results when reused."""
        inst_a = pregenerate_instances([8], [0.01], 10, SEED)
        inst_b = pregenerate_instances([8], [0.01], 10, SEED)

        for (e_a, s_a, llr_a), (e_b, s_b, llr_b) in zip(
            inst_a[(8, 0.01)], inst_b[(8, 0.01)]
        ):
            np.testing.assert_array_equal(e_a, e_b)
            np.testing.assert_array_equal(s_a, s_b)
            np.testing.assert_array_equal(llr_a, llr_b)
