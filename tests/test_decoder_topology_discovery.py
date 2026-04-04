"""
Tests for decoder_topology_discovery — v137.0.2

Covers:
- deterministic recommendations
- stable ordering
- export equality
- frozen immutability
- 100-run replay
- no decoder contamination
"""

from __future__ import annotations

import json
import sys

import pytest

from qec.analysis.decoder_topology_discovery import (
    KNOWN_DECODER_FAMILIES,
    SIMILARITY_CLASSES,
    TOPOLOGY_DISCOVERY_VERSION,
    TopologyDiscoveryDecision,
    TopologyDiscoveryLedger,
    build_topology_ledger,
    discover_decoder_topology,
    export_topology_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_decision(**overrides):
    """Helper to create a decision with sensible defaults."""
    defaults = dict(
        observation_signature="surface_obs_alpha",
        symbolic_risk_lattice="LOW",
        phase_bin_index=(1, 1),
        decoder_family="surface",
    )
    defaults.update(overrides)
    return discover_decoder_topology(**defaults)


# ---------------------------------------------------------------------------
# Deterministic recommendations
# ---------------------------------------------------------------------------


class TestDeterministicRecommendations:
    """Verify that identical inputs always produce identical outputs."""

    def test_basic_recommendation(self):
        d = _make_decision()
        assert d.recommended_decoder_family in KNOWN_DECODER_FAMILIES
        assert d.similarity_class in SIMILARITY_CLASSES
        assert isinstance(d.topology_pairing_score, float)
        assert 0.0 <= d.topology_pairing_score <= 1.0
        assert len(d.stable_hash) == 64

    def test_exact_family_match_scores_highest(self):
        d_surface = _make_decision(decoder_family="surface")
        assert d_surface.recommended_decoder_family == "surface"

    def test_known_portfolio_restricts_candidates(self):
        d = _make_decision(
            decoder_family=None,
            known_portfolio=("toric", "qldpc"),
        )
        assert d.recommended_decoder_family in ("toric", "qldpc")

    def test_empty_portfolio_falls_back_to_known(self):
        d = _make_decision(known_portfolio=())
        assert d.recommended_decoder_family in KNOWN_DECODER_FAMILIES

    def test_no_family_hint_still_recommends(self):
        d = discover_decoder_topology(
            observation_signature="some_signal",
            symbolic_risk_lattice="WARNING",
            phase_bin_index=(3, 5),
        )
        assert d.recommended_decoder_family in KNOWN_DECODER_FAMILIES

    def test_recovery_suggestions_are_tuples(self):
        d = _make_decision()
        assert isinstance(d.recovery_topology_suggestions, tuple)
        assert all(isinstance(s, str) for s in d.recovery_topology_suggestions)

    def test_all_families_produce_valid_decisions(self):
        for family in KNOWN_DECODER_FAMILIES:
            d = discover_decoder_topology(
                observation_signature=f"{family}_test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=(0, 0),
                decoder_family=family,
            )
            assert d.recommended_decoder_family == family


# ---------------------------------------------------------------------------
# Stable ordering
# ---------------------------------------------------------------------------


class TestStableOrdering:
    """Verify deterministic tie-breaking and ordering."""

    def test_alphabetical_tiebreak(self):
        """When no family hint, tie-breaking is alphabetical."""
        d1 = discover_decoder_topology(
            observation_signature="xyz",
            symbolic_risk_lattice="LOW",
            phase_bin_index=(100, 100),
        )
        d2 = discover_decoder_topology(
            observation_signature="xyz",
            symbolic_risk_lattice="LOW",
            phase_bin_index=(100, 100),
        )
        assert d1.recommended_decoder_family == d2.recommended_decoder_family

    def test_ledger_preserves_insertion_order(self):
        decisions = tuple(
            _make_decision(decoder_family=f)
            for f in KNOWN_DECODER_FAMILIES
        )
        ledger = build_topology_ledger(decisions)
        for i, family in enumerate(KNOWN_DECODER_FAMILIES):
            assert ledger.decisions[i].recommended_decoder_family == family


# ---------------------------------------------------------------------------
# Export equality
# ---------------------------------------------------------------------------


class TestExportEquality:
    """Verify that export is byte-identical for identical inputs."""

    def test_export_deterministic(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        e1 = export_topology_bundle(ledger)
        e2 = export_topology_bundle(ledger)
        assert e1 == e2

    def test_export_is_valid_json(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        exported = export_topology_bundle(ledger)
        parsed = json.loads(exported)
        assert "decisions" in parsed
        assert "ledger_hash" in parsed
        assert "version" in parsed

    def test_export_contains_all_decision_fields(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        exported = json.loads(export_topology_bundle(ledger))
        dec = exported["decisions"][0]
        assert "recommended_decoder_family" in dec
        assert "topology_pairing_score" in dec
        assert "recovery_topology_suggestions" in dec
        assert "similarity_class" in dec
        assert "stable_hash" in dec
        assert "observation_signature" in dec
        assert "symbolic_risk_lattice" in dec
        assert "phase_bin_index" in dec

    def test_rebuilt_ledger_matches_export(self):
        """Two independently built ledgers with same inputs export identically."""
        args = dict(
            observation_signature="toric_alpha",
            symbolic_risk_lattice="WATCH",
            phase_bin_index=(2, 3),
            decoder_family="toric",
        )
        d1 = discover_decoder_topology(**args)
        d2 = discover_decoder_topology(**args)
        l1 = build_topology_ledger((d1,))
        l2 = build_topology_ledger((d2,))
        assert export_topology_bundle(l1) == export_topology_bundle(l2)
        assert l1.stable_hash == l2.stable_hash


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestFrozenImmutability:
    """Verify that dataclasses are truly frozen."""

    def test_decision_is_frozen(self):
        d = _make_decision()
        with pytest.raises(AttributeError):
            d.recommended_decoder_family = "hacked"  # type: ignore[misc]

    def test_decision_hash_is_frozen(self):
        d = _make_decision()
        with pytest.raises(AttributeError):
            d.stable_hash = "0" * 64  # type: ignore[misc]

    def test_ledger_is_frozen(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        with pytest.raises(AttributeError):
            ledger.stable_hash = "0" * 64  # type: ignore[misc]

    def test_ledger_decisions_is_frozen(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        with pytest.raises(AttributeError):
            ledger.decisions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 100-run replay
# ---------------------------------------------------------------------------


class TestReplay:
    """Verify byte-identical replay over 100 runs."""

    def test_100_run_decision_replay(self):
        reference = _make_decision()
        for _ in range(100):
            d = _make_decision()
            assert d.recommended_decoder_family == reference.recommended_decoder_family
            assert d.topology_pairing_score == reference.topology_pairing_score
            assert d.recovery_topology_suggestions == reference.recovery_topology_suggestions
            assert d.similarity_class == reference.similarity_class
            assert d.stable_hash == reference.stable_hash

    def test_100_run_export_replay(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        reference_export = export_topology_bundle(ledger)
        for _ in range(100):
            d2 = _make_decision()
            l2 = build_topology_ledger((d2,))
            assert export_topology_bundle(l2) == reference_export

    def test_100_run_ledger_hash_replay(self):
        decisions = tuple(
            _make_decision(decoder_family=f)
            for f in KNOWN_DECODER_FAMILIES
        )
        reference_ledger = build_topology_ledger(decisions)
        for _ in range(100):
            new_decisions = tuple(
                _make_decision(decoder_family=f)
                for f in KNOWN_DECODER_FAMILIES
            )
            new_ledger = build_topology_ledger(new_decisions)
            assert new_ledger.stable_hash == reference_ledger.stable_hash


# ---------------------------------------------------------------------------
# No decoder contamination
# ---------------------------------------------------------------------------


class TestNoDecoderContamination:
    """Verify that this module never imports from qec.decoder."""

    def test_no_decoder_import_in_module(self):
        import qec.analysis.decoder_topology_discovery as mod
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        # Must not import from qec.decoder
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_decoder_in_sys_modules_after_import(self):
        """Importing this module must not pull in decoder modules."""
        decoder_modules_before = {
            k for k in sys.modules if k.startswith("qec.decoder")
        }
        # Re-import to ensure
        import importlib
        importlib.reload(
            importlib.import_module("qec.analysis.decoder_topology_discovery")
        )
        decoder_modules_after = {
            k for k in sys.modules if k.startswith("qec.decoder")
        }
        # No new decoder modules should appear
        new_decoder = decoder_modules_after - decoder_modules_before
        assert not new_decoder, f"Decoder modules loaded: {new_decoder}"


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersion:
    """Verify version string is present and correct."""

    def test_version(self):
        assert TOPOLOGY_DISCOVERY_VERSION == "v137.0.2"

    def test_ledger_version(self):
        d = _make_decision()
        ledger = build_topology_ledger((d,))
        assert ledger.ledger_version == "v137.0.2"


# ---------------------------------------------------------------------------
# Input validation (hardening pass)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify API boundary validation for phase_bin_index."""

    def test_negative_bin_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            discover_decoder_topology(
                observation_signature="test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=(-1, 0),
            )

    def test_negative_second_bin_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            discover_decoder_topology(
                observation_signature="test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=(0, -3),
            )

    def test_non_int_bin_raises(self):
        with pytest.raises(ValueError, match="two ints"):
            discover_decoder_topology(
                observation_signature="test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=(1.5, 2),  # type: ignore[arg-type]
            )

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="two ints"):
            discover_decoder_topology(
                observation_signature="test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=(1, 2, 3),  # type: ignore[arg-type]
            )

    def test_list_instead_of_tuple_raises(self):
        with pytest.raises(ValueError, match="two ints"):
            discover_decoder_topology(
                observation_signature="test",
                symbolic_risk_lattice="LOW",
                phase_bin_index=[1, 2],  # type: ignore[arg-type]
            )

    def test_ledger_rejects_non_decision(self):
        with pytest.raises(TypeError, match="TopologyDiscoveryDecision"):
            build_topology_ledger(("not_a_decision",))  # type: ignore[arg-type]

    def test_ledger_accepts_list_input(self):
        """build_topology_ledger normalizes list to tuple."""
        d = _make_decision()
        ledger = build_topology_ledger([d])  # type: ignore[arg-type]
        assert isinstance(ledger.decisions, tuple)
        assert len(ledger.decisions) == 1


# ---------------------------------------------------------------------------
# Integration-style test with realistic portfolio
# ---------------------------------------------------------------------------


class TestRealisticPortfolioIntegration:
    """Small integration test with a realistic known_portfolio."""

    def test_realistic_portfolio_deterministic(self):
        portfolio = ("surface", "toric", "qldpc")
        d = discover_decoder_topology(
            observation_signature="surface_syndrome_drift_high",
            symbolic_risk_lattice="WARNING",
            phase_bin_index=(1, 1),
            decoder_family="surface",
            known_portfolio=portfolio,
        )
        assert d.recommended_decoder_family == "surface"
        assert d.recovery_topology_suggestions
        assert len(d.recovery_topology_suggestions) > 0

        # Stable export
        ledger = build_topology_ledger((d,))
        e1 = export_topology_bundle(ledger)
        e2 = export_topology_bundle(build_topology_ledger((
            discover_decoder_topology(
                observation_signature="surface_syndrome_drift_high",
                symbolic_risk_lattice="WARNING",
                phase_bin_index=(1, 1),
                decoder_family="surface",
                known_portfolio=portfolio,
            ),
        )))
        assert e1 == e2
