"""
Tests for v3.8.0 — Structural Geometry Intervention.

Covers:
- Baseline identity (structural disabled → identical outputs)
- RPC determinism (two runs → identical H_aug)
- RPC feasible-set integrity (augmented system preserves solutions)
- Degree scaling determinism (two runs → identical LLR outputs)
- Four-mode harness compatibility (baseline, RPC only, degree only, RPC + degree)
"""

import numpy as np
import pytest

from src.qec_qldpc_codes import bp_decode, syndrome, create_code, channel_llr
from src.qec.decoder.rpc import (
    RPCConfig,
    StructuralConfig,
    build_rpc_augmented_system,
)
from src.qec.decoder.geom import compute_check_degree_scale


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code("rate_0.50", lifting_size=8, seed=42)


@pytest.fixture
def noisy_setup(small_code):
    """Code + low-noise error + syndrome + LLR."""
    rng = np.random.default_rng(42)
    e = (rng.random(small_code.n) < 0.05).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.05)
    return small_code, e, s, llr


@pytest.fixture
def rpc_config_enabled():
    """RPC config with augmentation enabled."""
    return StructuralConfig(
        rpc=RPCConfig(enabled=True, max_rows=5, w_min=2, w_max=50),
    )


@pytest.fixture
def rpc_config_disabled():
    """RPC config with augmentation disabled."""
    return StructuralConfig(
        rpc=RPCConfig(enabled=False),
    )


# ───────────────────────────────────────────────────────────────────
# Baseline Identity Tests
# ───────────────────────────────────────────────────────────────────

class TestBaselineIdentity:
    """Structural disabled → identical outputs to baseline."""

    def test_disabled_rpc_returns_same_H_s(self, small_code):
        """build_rpc_augmented_system with disabled config returns H, s unchanged."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)
        config = StructuralConfig(rpc=RPCConfig(enabled=False))

        H_aug, s_aug = build_rpc_augmented_system(H, s, config)

        assert H_aug is H
        assert s_aug is s

    def test_none_config_returns_same_H_s(self, small_code):
        """build_rpc_augmented_system with None config returns H, s unchanged."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)

        H_aug, s_aug = build_rpc_augmented_system(H, s, None)

        assert H_aug is H
        assert s_aug is s

    def test_flooding_baseline_unchanged(self, noisy_setup):
        """Flooding schedule with structural disabled matches baseline exactly."""
        code, e, s, llr = noisy_setup
        H = code.H_X

        # Baseline run
        corr_base, iters_base = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )

        # Second run — identical call
        corr_again, iters_again = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )

        np.testing.assert_array_equal(corr_base, corr_again)
        assert iters_base == iters_again

    def test_geom_v1_accepted_as_schedule(self, noisy_setup):
        """geom_v1 is accepted as a valid schedule."""
        code, e, s, llr = noisy_setup
        H = code.H_X

        corr, iters = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s,
        )
        assert corr.shape == (H.shape[1],)
        assert iters <= 10


# ───────────────────────────────────────────────────────────────────
# RPC Determinism Tests
# ───────────────────────────────────────────────────────────────────

class TestRPCDeterminism:
    """Two runs of RPC augmentation → identical H_aug."""

    def test_two_runs_identical(self, small_code, rpc_config_enabled):
        """RPC augmentation is deterministic across two calls."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)

        H_aug1, s_aug1 = build_rpc_augmented_system(H, s, rpc_config_enabled)
        H_aug2, s_aug2 = build_rpc_augmented_system(H, s, rpc_config_enabled)

        np.testing.assert_array_equal(H_aug1, H_aug2)
        np.testing.assert_array_equal(s_aug1, s_aug2)

    def test_augmented_has_more_rows(self, small_code, rpc_config_enabled):
        """Augmented H has more rows than original (when rows are accepted)."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)

        H_aug, s_aug = build_rpc_augmented_system(H, s, rpc_config_enabled)

        assert H_aug.shape[0] >= H.shape[0]
        assert H_aug.shape[1] == H.shape[1]
        assert s_aug.shape[0] == H_aug.shape[0]

    def test_original_H_not_mutated(self, small_code, rpc_config_enabled):
        """Original H is not modified in place."""
        H = small_code.H_X.astype(np.uint8)
        H_copy = H.copy()
        s = np.zeros(H.shape[0], dtype=np.uint8)

        build_rpc_augmented_system(H, s, rpc_config_enabled)

        np.testing.assert_array_equal(H, H_copy)

    def test_max_rows_respected(self, small_code):
        """Number of added rows does not exceed max_rows."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)
        max_rows = 3
        config = StructuralConfig(
            rpc=RPCConfig(enabled=True, max_rows=max_rows, w_min=1, w_max=1000),
        )

        H_aug, s_aug = build_rpc_augmented_system(H, s, config)

        added = H_aug.shape[0] - H.shape[0]
        assert added <= max_rows

    def test_weight_filter(self, small_code):
        """Redundant rows outside weight range are rejected."""
        H = small_code.H_X.astype(np.uint8)
        s = np.zeros(H.shape[0], dtype=np.uint8)

        # Very narrow weight range that likely rejects most rows.
        config = StructuralConfig(
            rpc=RPCConfig(enabled=True, max_rows=100, w_min=999, w_max=1000),
        )

        H_aug, s_aug = build_rpc_augmented_system(H, s, config)

        # Should have no added rows (weight filter too strict).
        assert H_aug.shape[0] == H.shape[0]

    def test_with_nonzero_syndrome(self, noisy_setup, rpc_config_enabled):
        """RPC augmentation works with a non-zero syndrome vector."""
        code, e, s, llr = noisy_setup
        H = code.H_X.astype(np.uint8)

        H_aug1, s_aug1 = build_rpc_augmented_system(H, s, rpc_config_enabled)
        H_aug2, s_aug2 = build_rpc_augmented_system(H, s, rpc_config_enabled)

        np.testing.assert_array_equal(H_aug1, H_aug2)
        np.testing.assert_array_equal(s_aug1, s_aug2)


# ───────────────────────────────────────────────────────────────────
# RPC Feasible-Set Integrity Tests
# ───────────────────────────────────────────────────────────────────

class TestRPCFeasibleSet:
    """Augmented system preserves feasible set."""

    def test_valid_error_satisfies_augmented(self, small_code, rpc_config_enabled):
        """An error e satisfying He=s also satisfies H_aug @ e = s_aug."""
        H = small_code.H_X.astype(np.uint8)
        rng = np.random.default_rng(123)

        for _ in range(5):
            e = (rng.random(H.shape[1]) < 0.05).astype(np.uint8)
            s = syndrome(H, e)

            H_aug, s_aug = build_rpc_augmented_system(H, s, rpc_config_enabled)

            # Verify augmented system is also satisfied.
            s_check = syndrome(H_aug, e)
            np.testing.assert_array_equal(s_check, s_aug)

    def test_zero_error_satisfies_augmented(self, small_code, rpc_config_enabled):
        """Zero error vector satisfies augmented system with zero syndrome."""
        H = small_code.H_X.astype(np.uint8)
        e = np.zeros(H.shape[1], dtype=np.uint8)
        s = syndrome(H, e)

        H_aug, s_aug = build_rpc_augmented_system(H, s, rpc_config_enabled)

        s_check = syndrome(H_aug, e)
        np.testing.assert_array_equal(s_check, s_aug)


# ───────────────────────────────────────────────────────────────────
# Degree Scaling Determinism Tests
# ───────────────────────────────────────────────────────────────────

class TestDegreeScalingDeterminism:
    """Degree-aware scaling is deterministic."""

    def test_scale_factors_deterministic(self, small_code):
        """compute_check_degree_scale returns identical results across calls."""
        from src.qec_qldpc_codes import _tanner_graph

        H = small_code.H_X
        c2v, v2c = _tanner_graph(H)

        alpha1 = compute_check_degree_scale(c2v)
        alpha2 = compute_check_degree_scale(c2v)

        np.testing.assert_array_equal(alpha1, alpha2)

    def test_scale_factors_values(self):
        """Scale factors match 1/sqrt(d) for known degrees."""
        # Manual test: checks with degree 4, 6, 1.
        c2v = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0]]
        alpha = compute_check_degree_scale(c2v)

        np.testing.assert_almost_equal(alpha[0], 1.0 / np.sqrt(4.0))
        np.testing.assert_almost_equal(alpha[1], 1.0 / np.sqrt(6.0))
        np.testing.assert_almost_equal(alpha[2], 1.0 / np.sqrt(1.0))

    def test_zero_degree_gives_one(self):
        """Degree-0 check gets scale factor 1.0 (no-op)."""
        c2v = [[], [0, 1]]
        alpha = compute_check_degree_scale(c2v)

        assert alpha[0] == 1.0
        np.testing.assert_almost_equal(alpha[1], 1.0 / np.sqrt(2.0))

    def test_geom_v1_llr_determinism(self, noisy_setup):
        """Two geom_v1 BP runs produce identical LLR outputs."""
        code, e, s, llr = noisy_setup
        H = code.H_X

        corr1, iters1 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s,
        )
        corr2, iters2 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s,
        )

        np.testing.assert_array_equal(corr1, corr2)
        assert iters1 == iters2

    def test_geom_v1_with_llr_history_determinism(self, noisy_setup):
        """geom_v1 with llr_history returns deterministic history."""
        code, e, s, llr = noisy_setup
        H = code.H_X

        corr1, iters1, hist1 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s, llr_history=3,
        )
        corr2, iters2, hist2 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s, llr_history=3,
        )

        np.testing.assert_array_equal(corr1, corr2)
        assert iters1 == iters2
        np.testing.assert_array_equal(hist1, hist2)


# ───────────────────────────────────────────────────────────────────
# Four-Mode Harness Compatibility
# ───────────────────────────────────────────────────────────────────

class TestFourModeCompatibility:
    """All four modes run without error and produce valid output shapes."""

    def _run_decode(self, H, llr, s, schedule, structural_config=None):
        """Run bp_decode with optional RPC augmentation."""
        H_used, s_used = H, s
        if structural_config is not None and structural_config.rpc.enabled:
            H_used, s_used = build_rpc_augmented_system(H, s, structural_config)

        corr, iters = bp_decode(
            H_used, llr, max_iters=15, mode="min_sum",
            schedule=schedule, syndrome_vec=s_used,
        )
        return corr, iters

    def test_mode_baseline(self, noisy_setup):
        """Mode 1: baseline (no RPC, flooding)."""
        code, e, s, llr = noisy_setup
        corr, iters = self._run_decode(code.H_X, llr, s, "flooding")
        assert corr.shape == (code.H_X.shape[1],)
        assert iters <= 15

    def test_mode_rpc_only(self, noisy_setup, rpc_config_enabled):
        """Mode 2: RPC only (flooding schedule)."""
        code, e, s, llr = noisy_setup
        corr, iters = self._run_decode(
            code.H_X, llr, s, "flooding", rpc_config_enabled,
        )
        # Correction length matches original n (not augmented m).
        assert corr.shape[0] == code.H_X.shape[1]
        assert iters <= 15

    def test_mode_degree_only(self, noisy_setup):
        """Mode 3: degree rebalancing only (geom_v1 schedule, no RPC)."""
        code, e, s, llr = noisy_setup
        corr, iters = self._run_decode(code.H_X, llr, s, "geom_v1")
        assert corr.shape == (code.H_X.shape[1],)
        assert iters <= 15

    def test_mode_rpc_plus_degree(self, noisy_setup, rpc_config_enabled):
        """Mode 4: RPC + degree rebalancing (geom_v1 schedule)."""
        code, e, s, llr = noisy_setup
        corr, iters = self._run_decode(
            code.H_X, llr, s, "geom_v1", rpc_config_enabled,
        )
        assert corr.shape[0] == code.H_X.shape[1]
        assert iters <= 15


# ───────────────────────────────────────────────────────────────────
# StructuralConfig / RPCConfig construction
# ───────────────────────────────────────────────────────────────────

class TestConfigConstruction:
    """Config dataclass construction and from_dict."""

    def test_default_structural_config(self):
        """Default StructuralConfig has RPC disabled."""
        cfg = StructuralConfig()
        assert cfg.rpc.enabled is False

    def test_from_dict_none(self):
        """from_dict(None) returns defaults."""
        cfg = StructuralConfig.from_dict(None)
        assert cfg.rpc.enabled is False

    def test_from_dict_rpc_enabled(self):
        """from_dict with RPC settings."""
        cfg = StructuralConfig.from_dict({
            "rpc": {"enabled": True, "max_rows": 7, "w_min": 3, "w_max": 40},
        })
        assert cfg.rpc.enabled is True
        assert cfg.rpc.max_rows == 7
        assert cfg.rpc.w_min == 3
        assert cfg.rpc.w_max == 40

    def test_from_dict_empty_rpc(self):
        """from_dict with empty rpc dict uses defaults."""
        cfg = StructuralConfig.from_dict({"rpc": {}})
        assert cfg.rpc.enabled is False
        assert cfg.rpc.max_rows == 10

    def test_rpc_config_frozen(self):
        """RPCConfig is frozen (immutable)."""
        cfg = RPCConfig(enabled=True)
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]
