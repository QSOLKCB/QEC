"""Tests for conceptual branch module — v137.0.0.

Covers:
1.  branch-state creation
2.  hash-chain determinism
3.  export equality
4.  frozen immutability
5.  no-regression against v136.9.x governed bundles
6.  no-regression against v136.10.0 quantization
7.  100-run determinism
8.  canonical branch epoch field
9.  stable ledger hashing
10. roadmap / doc consistency
"""

from __future__ import annotations

import json
import os

import pytest

from qec.analysis.conceptual_branch_v137 import (
    CONCEPTUAL_BRANCH_VERSION,
    ConceptualBranchLedger,
    ConceptualBranchState,
    build_branch_ledger,
    export_conceptual_branch_bundle,
    export_conceptual_branch_ledger,
    unify_control_and_quantization,
    _compute_replay_hash_chain,
)


# ---------------------------------------------------------------------------
# Fixtures — deterministic upstream hashes (simulated stable_hash values)
# ---------------------------------------------------------------------------

GOVERNED_HASH = "a" * 64  # simulated SHA-256 from GovernedSteeringBundle
QUANTIZATION_HASH = "b" * 64  # simulated SHA-256 from QuantizationDecision
POLICY_HASH = "c" * 64  # simulated SHA-256 from PolicyMemoryLedger


def _make_state(
    governed_route: str = "PRIMARY",
    quantized_risk_band: str = "LOW",
    policy_cycle_index: int = 5,
    oscillation_count: int = 0,
    phase_bin_index: tuple = (3, -2),
    quantization_domain: str = "control_risk_band",
    governed_bundle_hash: str = GOVERNED_HASH,
    quantization_decision_hash: str = QUANTIZATION_HASH,
    policy_memory_hash: str | None = POLICY_HASH,
) -> ConceptualBranchState:
    """Helper to create a branch state with canonical defaults."""
    return unify_control_and_quantization(
        governed_route=governed_route,
        quantized_risk_band=quantized_risk_band,
        policy_cycle_index=policy_cycle_index,
        oscillation_count=oscillation_count,
        phase_bin_index=phase_bin_index,
        quantization_domain=quantization_domain,
        governed_bundle_hash=governed_bundle_hash,
        quantization_decision_hash=quantization_decision_hash,
        policy_memory_hash=policy_memory_hash,
    )


# -----------------------------------------------------------------------
# 1. Branch-state creation
# -----------------------------------------------------------------------

class TestBranchStateCreation:
    """ConceptualBranchState construction and field correctness."""

    def test_basic_creation(self):
        state = _make_state()
        assert state.governed_route == "PRIMARY"
        assert state.quantized_risk_band == "LOW"
        assert state.policy_cycle_index == 5
        assert state.oscillation_count == 0
        assert state.phase_bin_index == (3, -2)
        assert state.quantization_domain == "control_risk_band"
        assert state.branch_epoch == "v137.0.0"
        assert len(state.replay_hash_chain) == 64

    def test_all_routes(self):
        for route in ("PRIMARY", "RECOVERY", "ALTERNATE", "EMERGENCY"):
            state = _make_state(governed_route=route)
            assert state.governed_route == route

    def test_all_risk_bands(self):
        for band in ("LOW", "WATCH", "WARNING", "CRITICAL", "COLLAPSE_IMMINENT"):
            state = _make_state(quantized_risk_band=band)
            assert state.quantized_risk_band == band

    def test_without_policy_memory(self):
        state = _make_state(policy_memory_hash=None)
        assert len(state.replay_hash_chain) == 64

    def test_hash_differs_with_and_without_policy(self):
        s1 = _make_state(policy_memory_hash=POLICY_HASH)
        s2 = _make_state(policy_memory_hash=None)
        assert s1.replay_hash_chain != s2.replay_hash_chain


# -----------------------------------------------------------------------
# 2. Hash-chain determinism
# -----------------------------------------------------------------------

class TestHashChainDeterminism:
    """Replay hash chain must be deterministic."""

    def test_same_inputs_same_hash(self):
        h1 = _compute_replay_hash_chain(GOVERNED_HASH, QUANTIZATION_HASH, POLICY_HASH)
        h2 = _compute_replay_hash_chain(GOVERNED_HASH, QUANTIZATION_HASH, POLICY_HASH)
        assert h1 == h2

    def test_different_governed_hash_different_chain(self):
        h1 = _compute_replay_hash_chain(GOVERNED_HASH, QUANTIZATION_HASH)
        h2 = _compute_replay_hash_chain("d" * 64, QUANTIZATION_HASH)
        assert h1 != h2

    def test_different_quantization_hash_different_chain(self):
        h1 = _compute_replay_hash_chain(GOVERNED_HASH, QUANTIZATION_HASH)
        h2 = _compute_replay_hash_chain(GOVERNED_HASH, "e" * 64)
        assert h1 != h2

    def test_hash_is_sha256_hex(self):
        h = _compute_replay_hash_chain(GOVERNED_HASH, QUANTIZATION_HASH)
        assert len(h) == 64
        int(h, 16)  # must be valid hex


# -----------------------------------------------------------------------
# 3. Export equality
# -----------------------------------------------------------------------

class TestExportEquality:
    """Export must be deterministic and byte-identical."""

    def test_state_export_determinism(self):
        state = _make_state()
        e1 = export_conceptual_branch_bundle(state)
        e2 = export_conceptual_branch_bundle(state)
        assert e1 == e2

    def test_state_export_json_safe(self):
        state = _make_state()
        exported = export_conceptual_branch_bundle(state)
        # Must be JSON-serializable
        serialized = json.dumps(exported, sort_keys=True, separators=(",", ":"))
        assert isinstance(serialized, str)
        # Must round-trip
        recovered = json.loads(serialized)
        assert recovered == exported

    def test_state_export_contains_metadata(self):
        state = _make_state()
        exported = export_conceptual_branch_bundle(state)
        assert exported["layer"] == "conceptual_branch_v137"
        assert exported["version"] == CONCEPTUAL_BRANCH_VERSION
        assert exported["branch_epoch"] == "v137.0.0"
        assert exported["replay_hash_chain"] == state.replay_hash_chain

    def test_ledger_export_determinism(self):
        s1 = _make_state(policy_cycle_index=0)
        s2 = _make_state(policy_cycle_index=1)
        ledger = build_branch_ledger((s1, s2))
        e1 = export_conceptual_branch_ledger(ledger)
        e2 = export_conceptual_branch_ledger(ledger)
        assert e1 == e2

    def test_ledger_export_json_parseable(self):
        state = _make_state()
        ledger = build_branch_ledger((state,))
        exported = export_conceptual_branch_ledger(ledger)
        parsed = json.loads(exported)
        assert parsed["version"] == CONCEPTUAL_BRANCH_VERSION
        assert parsed["state_count"] == 1


# -----------------------------------------------------------------------
# 4. Frozen immutability
# -----------------------------------------------------------------------

class TestFrozenImmutability:
    """All dataclasses must be frozen (immutable)."""

    def test_state_immutable(self):
        state = _make_state()
        with pytest.raises(AttributeError):
            state.governed_route = "RECOVERY"

    def test_state_epoch_immutable(self):
        state = _make_state()
        with pytest.raises(AttributeError):
            state.branch_epoch = "v999.0.0"

    def test_state_hash_immutable(self):
        state = _make_state()
        with pytest.raises(AttributeError):
            state.replay_hash_chain = "tampered"

    def test_ledger_immutable(self):
        state = _make_state()
        ledger = build_branch_ledger((state,))
        with pytest.raises(AttributeError):
            ledger.stable_hash = "tampered"

    def test_ledger_states_immutable(self):
        state = _make_state()
        ledger = build_branch_ledger((state,))
        with pytest.raises(AttributeError):
            ledger.states = ()


# -----------------------------------------------------------------------
# 5. No-regression against v136.9.x governed bundles
# -----------------------------------------------------------------------

class TestNoRegressionV136_9:
    """Conceptual branch must consume v136.9.x vocabulary without error."""

    def test_governed_routes_accepted(self):
        """All v136.9.x route labels must be valid."""
        for route in ("PRIMARY", "RECOVERY", "ALTERNATE", "EMERGENCY"):
            state = _make_state(governed_route=route)
            exported = export_conceptual_branch_bundle(state)
            assert exported["governed_route"] == route

    def test_oscillation_count_preserved(self):
        state = _make_state(oscillation_count=7)
        assert state.oscillation_count == 7

    def test_policy_cycle_index_preserved(self):
        state = _make_state(policy_cycle_index=42)
        assert state.policy_cycle_index == 42

    def test_governed_bundle_hash_flows_into_chain(self):
        """Different governed hashes must produce different chains."""
        s1 = _make_state(governed_bundle_hash="a" * 64)
        s2 = _make_state(governed_bundle_hash="f" * 64)
        assert s1.replay_hash_chain != s2.replay_hash_chain


# -----------------------------------------------------------------------
# 6. No-regression against v136.10.0 quantization
# -----------------------------------------------------------------------

class TestNoRegressionV136_10:
    """Conceptual branch must consume v136.10.0 vocabulary without error."""

    def test_risk_bands_accepted(self):
        for band in ("LOW", "WATCH", "WARNING", "CRITICAL", "COLLAPSE_IMMINENT"):
            state = _make_state(quantized_risk_band=band)
            assert state.quantized_risk_band == band

    def test_quantization_domains_accepted(self):
        for domain in ("audio_sample_rate", "audio_bit_depth", "ai_weight",
                        "control_risk_band", "phase_space"):
            state = _make_state(quantization_domain=domain)
            assert state.quantization_domain == domain

    def test_phase_bin_index_preserved(self):
        state = _make_state(phase_bin_index=(10, -5))
        assert state.phase_bin_index == (10, -5)

    def test_quantization_hash_flows_into_chain(self):
        s1 = _make_state(quantization_decision_hash="b" * 64)
        s2 = _make_state(quantization_decision_hash="f" * 64)
        assert s1.replay_hash_chain != s2.replay_hash_chain


# -----------------------------------------------------------------------
# 7. 100-run determinism
# -----------------------------------------------------------------------

class TestDeterminism100Run:
    """All outputs must be byte-identical across 100 runs."""

    def test_state_hash_100_runs(self):
        hashes = set()
        for _ in range(100):
            state = _make_state()
            hashes.add(state.replay_hash_chain)
        assert len(hashes) == 1

    def test_export_100_runs(self):
        exports = set()
        for _ in range(100):
            state = _make_state()
            exported = json.dumps(
                export_conceptual_branch_bundle(state),
                sort_keys=True, separators=(",", ":"),
            )
            exports.add(exported)
        assert len(exports) == 1

    def test_ledger_hash_100_runs(self):
        hashes = set()
        for _ in range(100):
            s1 = _make_state(policy_cycle_index=0)
            s2 = _make_state(policy_cycle_index=1)
            ledger = build_branch_ledger((s1, s2))
            hashes.add(ledger.stable_hash)
        assert len(hashes) == 1

    def test_ledger_export_100_runs(self):
        exports = set()
        for _ in range(100):
            state = _make_state()
            ledger = build_branch_ledger((state,))
            exports.add(export_conceptual_branch_ledger(ledger))
        assert len(exports) == 1


# -----------------------------------------------------------------------
# 8. Canonical branch epoch field
# -----------------------------------------------------------------------

class TestBranchEpoch:
    """Branch epoch must always be v137.0.0."""

    def test_default_epoch(self):
        state = _make_state()
        assert state.branch_epoch == "v137.0.0"

    def test_epoch_in_export(self):
        state = _make_state()
        exported = export_conceptual_branch_bundle(state)
        assert exported["branch_epoch"] == "v137.0.0"

    def test_version_constant(self):
        assert CONCEPTUAL_BRANCH_VERSION == "v137.0.0"


# -----------------------------------------------------------------------
# 9. Stable ledger hashing
# -----------------------------------------------------------------------

class TestLedgerHashing:
    """Ledger hashing must be stable and order-sensitive."""

    def test_empty_ledger(self):
        ledger = build_branch_ledger(())
        assert ledger.state_count == 0
        assert len(ledger.stable_hash) == 64

    def test_single_state_ledger(self):
        state = _make_state()
        ledger = build_branch_ledger((state,))
        assert ledger.state_count == 1
        assert len(ledger.stable_hash) == 64

    def test_order_matters(self):
        s1 = _make_state(policy_cycle_index=0)
        s2 = _make_state(policy_cycle_index=1)
        l1 = build_branch_ledger((s1, s2))
        l2 = build_branch_ledger((s2, s1))
        assert l1.stable_hash != l2.stable_hash

    def test_different_states_different_hash(self):
        s1 = _make_state(governed_route="PRIMARY")
        s2 = _make_state(governed_route="RECOVERY")
        l1 = build_branch_ledger((s1,))
        l2 = build_branch_ledger((s2,))
        assert l1.stable_hash != l2.stable_hash

    def test_ledger_hash_is_sha256_hex(self):
        state = _make_state()
        ledger = build_branch_ledger((state,))
        assert len(ledger.stable_hash) == 64
        int(ledger.stable_hash, 16)

    def test_append_changes_hash(self):
        s1 = _make_state(policy_cycle_index=0)
        s2 = _make_state(policy_cycle_index=1)
        l1 = build_branch_ledger((s1,))
        l2 = build_branch_ledger((s1, s2))
        assert l1.stable_hash != l2.stable_hash


# -----------------------------------------------------------------------
# 10. Roadmap / doc consistency
# -----------------------------------------------------------------------

class TestDocConsistency:
    """Roadmap and docs must reflect canonical v137 roadmap semantics."""

    def test_roadmap_contains_v137(self):
        roadmap_path = os.path.join(
            os.path.dirname(__file__), "..", "ROADMAP.md",
        )
        with open(roadmap_path, encoding="utf-8") as f:
            content = f.read()
        assert "stable tip" in content.lower()
        assert "v137." in content

    def test_roadmap_contains_governed_quantized(self):
        roadmap_path = os.path.join(
            os.path.dirname(__file__), "..", "ROADMAP.md",
        )
        with open(roadmap_path, encoding="utf-8") as f:
            content = f.read()
        lowered = content.lower()
        assert "determin" in lowered
        assert "replay" in lowered

    def test_roadmap_contains_replay(self):
        roadmap_path = os.path.join(
            os.path.dirname(__file__), "..", "ROADMAP.md",
        )
        with open(roadmap_path, encoding="utf-8") as f:
            content = f.read()
        assert "replay" in content.lower()
