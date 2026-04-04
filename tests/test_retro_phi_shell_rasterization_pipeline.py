"""Tests for v137.0.13 — Retro Phi-Shell Rasterization Pipeline.

Theory-coupled tests verifying:
  - PHI_SCALE_NODE: phi shell quantization correctness
  - E8_TRIALITY_LOCK: visibility classification boundaries
  - OUROBOROS_FEEDBACK_LOOP: UFF restore term determinism
  - SIS2_STABILITY_RING: ledger replay identity
"""

import hashlib
import json
import math
import sys

import pytest

from qec.analysis.retro_phi_shell_rasterization_pipeline import (
    DEFAULT_PHASE_OFFSET,
    FLOAT_PRECISION,
    MID_SHELL,
    NEAR_SHELL,
    OUTER_SHELL,
    PHI,
    PHI_SHELLS,
    RESONANCE_NODE,
    RETRO_PHI_SHELL_VERSION,
    VALID_VISIBILITY_CLASSES,
    WIGGLE_ZONE,
    RetroPhiShell,
    RetroRasterDecision,
    RetroRasterLedger,
    RetroRasterSpan,
    _canonical_bound_value,
    _compute_phi_midpoints,
    build_phi_raster_decision,
    build_phi_raster_ledger,
    build_phi_scanline_spans,
    build_phi_shell_descriptors,
    build_symbolic_trace,
    classify_shell_visibility,
    compute_phi_restore_term,
    export_phi_raster_bundle,
    export_phi_raster_ledger,
    quantize_depth_phi_shell,
)


# -----------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------


class TestVersion:
    def test_version_string(self):
        assert RETRO_PHI_SHELL_VERSION == "v137.0.13"

    def test_phi_constant(self):
        assert PHI == 1.618


# -----------------------------------------------------------------------
# Frozen dataclasses
# -----------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_raster_span_frozen(self):
        s = RetroRasterSpan(0, 1.0, 1.0, NEAR_SHELL, 0.5, "abc")
        with pytest.raises(AttributeError):
            s.span_index = 1

    def test_phi_shell_frozen(self):
        s = RetroPhiShell(0, 1.0, 0.0, 1.309, "abc")
        with pytest.raises(AttributeError):
            s.shell_index = 1

    def test_raster_decision_frozen(self):
        d = build_phi_raster_decision(4, 5.0)
        with pytest.raises(AttributeError):
            d.width = 10

    def test_raster_ledger_frozen(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        with pytest.raises(AttributeError):
            ledger.decision_count = 99

    def test_span_default_version(self):
        s = RetroRasterSpan(0, 1.0, 1.0, NEAR_SHELL, 0.5, "h")
        assert s.version == RETRO_PHI_SHELL_VERSION

    def test_shell_default_version(self):
        s = RetroPhiShell(0, 1.0, 0.0, 1.309, "h")
        assert s.version == RETRO_PHI_SHELL_VERSION

    def test_decision_default_version(self):
        d = build_phi_raster_decision(2, 3.0)
        assert d.version == RETRO_PHI_SHELL_VERSION

    def test_ledger_default_version(self):
        d = build_phi_raster_decision(2, 3.0)
        ledger = build_phi_raster_ledger((d,))
        assert ledger.version == RETRO_PHI_SHELL_VERSION


# -----------------------------------------------------------------------
# PHI shell progression (PHI_SCALE_NODE invariant)
# -----------------------------------------------------------------------


class TestPhiShellProgression:
    def test_shell_count(self):
        assert len(PHI_SHELLS) == 5

    def test_shell_values(self):
        assert PHI_SHELLS == (1.0, 1.618, 2.618, 4.236, 6.854)

    def test_monotonic_increasing(self):
        for i in range(1, len(PHI_SHELLS)):
            assert PHI_SHELLS[i] > PHI_SHELLS[i - 1]

    def test_golden_recurrence(self):
        """Each shell (from index 2+) approximately equals sum of two preceding."""
        for i in range(2, len(PHI_SHELLS)):
            expected = PHI_SHELLS[i - 2] + PHI_SHELLS[i - 1]
            assert abs(PHI_SHELLS[i] - expected) < 0.001

    def test_first_shell_is_unity(self):
        assert PHI_SHELLS[0] == 1.0

    def test_second_shell_is_phi(self):
        assert PHI_SHELLS[1] == 1.618


# -----------------------------------------------------------------------
# Phi shell quantization
# -----------------------------------------------------------------------


class TestQuantizeDepthPhiShell:
    def test_exact_shell_values(self):
        for sv in PHI_SHELLS:
            assert quantize_depth_phi_shell(sv) == sv

    def test_zero_maps_to_first_shell(self):
        assert quantize_depth_phi_shell(0.0) == 1.0

    def test_large_depth_maps_to_last_shell(self):
        assert quantize_depth_phi_shell(100.0) == 6.854

    def test_midpoint_near_first(self):
        # Midpoint of 1.0 and 1.618 is 1.309; at midpoint, smaller wins
        assert quantize_depth_phi_shell(1.309) == 1.0

    def test_just_above_midpoint(self):
        assert quantize_depth_phi_shell(1.31) == 1.618

    def test_between_second_and_third(self):
        assert quantize_depth_phi_shell(2.0) == 1.618

    def test_between_third_and_fourth(self):
        # 3.5 is closer to 4.236 (dist 0.736) than 2.618 (dist 0.882)
        assert quantize_depth_phi_shell(3.5) == 4.236

    def test_between_fourth_and_fifth(self):
        assert quantize_depth_phi_shell(5.5) == 4.236

    def test_negative_depth_rejected(self):
        with pytest.raises(ValueError):
            quantize_depth_phi_shell(-1.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            quantize_depth_phi_shell(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            quantize_depth_phi_shell(float("inf"))

    def test_string_rejected(self):
        with pytest.raises(TypeError):
            quantize_depth_phi_shell("1.0")

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            quantize_depth_phi_shell(True)

    def test_deterministic(self):
        """Same input always returns same output."""
        for _ in range(100):
            assert quantize_depth_phi_shell(2.5) == 2.618


# -----------------------------------------------------------------------
# UFF restore term (OUROBOROS_FEEDBACK_LOOP invariant)
# -----------------------------------------------------------------------


class TestComputePhiRestoreTerm:
    def test_zero_energy(self):
        result = compute_phi_restore_term(0.0)
        expected = round(0.0 + ((1.618 + math.pi / 2) ** 2) * 0.01, FLOAT_PRECISION)
        assert result == expected

    def test_unit_energy(self):
        result = compute_phi_restore_term(1.0)
        expected = round(1.0 + ((1.618 + math.pi / 2) ** 2) * 0.01, FLOAT_PRECISION)
        assert result == expected

    def test_custom_phase_offset(self):
        result = compute_phi_restore_term(0.5, phase_offset=0.0)
        expected = round(0.5 + ((1.618 + 0.0) ** 2) * 0.01, FLOAT_PRECISION)
        assert result == expected

    def test_default_phase_offset(self):
        assert DEFAULT_PHASE_OFFSET == math.pi / 2

    def test_restore_always_ge_energy(self):
        """Restore term is always >= span_energy (additive correction)."""
        for e in [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]:
            assert compute_phi_restore_term(e) >= e

    def test_deterministic_100_runs(self):
        results = [compute_phi_restore_term(0.42) for _ in range(100)]
        assert len(set(results)) == 1

    def test_negative_energy(self):
        # Negative span_energy is valid (correction still applies)
        result = compute_phi_restore_term(-1.0)
        expected = round(-1.0 + ((1.618 + math.pi / 2) ** 2) * 0.01, FLOAT_PRECISION)
        assert result == expected

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_restore_term(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_restore_term(float("inf"))

    def test_string_rejected(self):
        with pytest.raises(TypeError):
            compute_phi_restore_term("0.5")

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            compute_phi_restore_term(True)


# -----------------------------------------------------------------------
# Visibility classification (E8_TRIALITY_LOCK invariant)
# -----------------------------------------------------------------------


class TestClassifyShellVisibility:
    def test_near_shell(self):
        assert classify_shell_visibility(0.0) == NEAR_SHELL
        assert classify_shell_visibility(1.0) == NEAR_SHELL
        assert classify_shell_visibility(1.309) == NEAR_SHELL

    def test_mid_shell(self):
        assert classify_shell_visibility(1.31) == MID_SHELL
        assert classify_shell_visibility(1.618) == MID_SHELL
        assert classify_shell_visibility(2.118) == MID_SHELL

    def test_outer_shell(self):
        assert classify_shell_visibility(2.119) == OUTER_SHELL
        assert classify_shell_visibility(2.618) == OUTER_SHELL
        assert classify_shell_visibility(3.427) == OUTER_SHELL

    def test_resonance_node(self):
        assert classify_shell_visibility(3.428) == RESONANCE_NODE
        assert classify_shell_visibility(4.236) == RESONANCE_NODE
        assert classify_shell_visibility(5.545) == RESONANCE_NODE

    def test_wiggle_zone(self):
        assert classify_shell_visibility(5.546) == WIGGLE_ZONE
        assert classify_shell_visibility(6.854) == WIGGLE_ZONE
        assert classify_shell_visibility(100.0) == WIGGLE_ZONE

    def test_all_classes_reachable(self):
        classes = set()
        for d in [0.5, 1.5, 3.0, 4.5, 7.0]:
            classes.add(classify_shell_visibility(d))
        assert classes == set(VALID_VISIBILITY_CLASSES)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            classify_shell_visibility(-0.1)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            classify_shell_visibility(float("nan"))

    def test_string_rejected(self):
        with pytest.raises(TypeError):
            classify_shell_visibility("1.0")

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            classify_shell_visibility(True)

    def test_deterministic(self):
        for _ in range(100):
            assert classify_shell_visibility(2.0) == MID_SHELL


# -----------------------------------------------------------------------
# Scanline spans
# -----------------------------------------------------------------------


class TestBuildPhiScanlineSpans:
    def test_single_column(self):
        spans = build_phi_scanline_spans(1, 5.0)
        assert len(spans) == 1
        assert spans[0].span_index == 0
        assert spans[0].depth == 0.0

    def test_multiple_columns(self):
        spans = build_phi_scanline_spans(10, 5.0)
        assert len(spans) == 10

    def test_all_spans_have_phi_shells(self):
        spans = build_phi_scanline_spans(8, 6.0)
        for s in spans:
            assert s.phi_shell in PHI_SHELLS

    def test_all_spans_have_valid_visibility(self):
        spans = build_phi_scanline_spans(8, 6.0)
        for s in spans:
            assert s.visibility_class in VALID_VISIBILITY_CLASSES

    def test_all_spans_have_hashes(self):
        spans = build_phi_scanline_spans(5, 3.0)
        for s in spans:
            assert len(s.stable_hash) == 64

    def test_span_indices_sequential(self):
        spans = build_phi_scanline_spans(5, 3.0)
        for i, s in enumerate(spans):
            assert s.span_index == i

    def test_depth_monotonic(self):
        spans = build_phi_scanline_spans(10, 5.0)
        for i in range(1, len(spans)):
            assert spans[i].depth >= spans[i - 1].depth

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError):
            build_phi_scanline_spans(0, 5.0)

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError):
            build_phi_scanline_spans(-1, 5.0)

    def test_zero_max_depth_rejected(self):
        with pytest.raises(ValueError):
            build_phi_scanline_spans(5, 0.0)

    def test_bool_width_rejected(self):
        with pytest.raises(TypeError):
            build_phi_scanline_spans(True, 5.0)

    def test_string_max_depth_rejected(self):
        with pytest.raises(TypeError):
            build_phi_scanline_spans(5, "3.0")

    def test_deterministic(self):
        a = build_phi_scanline_spans(8, 5.0)
        b = build_phi_scanline_spans(8, 5.0)
        assert a == b


# -----------------------------------------------------------------------
# Shell descriptors
# -----------------------------------------------------------------------


class TestBuildPhiShellDescriptors:
    def test_count(self):
        shells = build_phi_shell_descriptors()
        assert len(shells) == 5

    def test_shell_values_match(self):
        shells = build_phi_shell_descriptors()
        for i, s in enumerate(shells):
            assert s.shell_value == PHI_SHELLS[i]

    def test_first_lower_bound_zero(self):
        shells = build_phi_shell_descriptors()
        assert shells[0].lower_bound == 0.0

    def test_last_upper_bound_inf(self):
        shells = build_phi_shell_descriptors()
        assert shells[-1].upper_bound == float("inf")

    def test_monotonic_shell_ordering(self):
        shells = build_phi_shell_descriptors()
        for i in range(1, len(shells)):
            assert shells[i].shell_value > shells[i - 1].shell_value

    def test_boundaries_contiguous(self):
        """Each shell's lower_bound matches previous shell's upper_bound.

        Interval contract: [lower_bound, upper_bound) for all shells.
        Final shell: [lower_bound, INF).
        """
        shells = build_phi_shell_descriptors()
        for i in range(1, len(shells)):
            assert shells[i].lower_bound == shells[i - 1].upper_bound

    def test_all_hashes_unique(self):
        shells = build_phi_shell_descriptors()
        hashes = [s.stable_hash for s in shells]
        assert len(set(hashes)) == len(hashes)


# -----------------------------------------------------------------------
# Symbolic trace
# -----------------------------------------------------------------------


class TestBuildSymbolicTrace:
    def test_empty_spans(self):
        assert build_symbolic_trace(()) == ""

    def test_single_span(self):
        s = RetroRasterSpan(0, 1.0, 1.0, NEAR_SHELL, 0.5, "h")
        trace = build_symbolic_trace((s,))
        assert trace == "NEAR_SHELL -> PHI_SCALE_NODE"

    def test_deduplication(self):
        s1 = RetroRasterSpan(0, 1.0, 1.0, NEAR_SHELL, 0.5, "h1")
        s2 = RetroRasterSpan(1, 1.0, 1.0, NEAR_SHELL, 0.5, "h2")
        s3 = RetroRasterSpan(2, 2.0, 1.618, MID_SHELL, 0.6, "h3")
        trace = build_symbolic_trace((s1, s2, s3))
        assert trace == "NEAR_SHELL -> MID_SHELL -> PHI_SCALE_NODE"

    def test_all_classes_transition(self):
        spans = []
        classes = [NEAR_SHELL, MID_SHELL, OUTER_SHELL, RESONANCE_NODE, WIGGLE_ZONE]
        for i, c in enumerate(classes):
            spans.append(RetroRasterSpan(i, float(i), 1.0, c, 0.5, f"h{i}"))
        trace = build_symbolic_trace(tuple(spans))
        parts = trace.split(" -> ")
        assert parts[-1] == "PHI_SCALE_NODE"
        assert len(parts) == 6

    def test_terminal_phi_scale_node(self):
        """Every non-empty trace ends with PHI_SCALE_NODE."""
        spans = build_phi_scanline_spans(5, 3.0)
        trace = build_symbolic_trace(spans)
        assert trace.endswith("PHI_SCALE_NODE")


# -----------------------------------------------------------------------
# Raster decision
# -----------------------------------------------------------------------


class TestBuildPhiRasterDecision:
    def test_basic_construction(self):
        d = build_phi_raster_decision(4, 5.0)
        assert d.width == 4
        assert d.max_depth == 5.0
        assert d.span_count == 4
        assert d.shell_count == 5

    def test_has_stable_hash(self):
        d = build_phi_raster_decision(4, 5.0)
        assert len(d.stable_hash) == 64

    def test_has_symbolic_trace(self):
        d = build_phi_raster_decision(10, 6.0)
        assert "PHI_SCALE_NODE" in d.symbolic_trace

    def test_invalid_width_rejected(self):
        with pytest.raises(ValueError):
            build_phi_raster_decision(0, 5.0)

    def test_invalid_max_depth_rejected(self):
        with pytest.raises(ValueError):
            build_phi_raster_decision(4, -1.0)


# -----------------------------------------------------------------------
# Raster ledger
# -----------------------------------------------------------------------


class TestBuildPhiRasterLedger:
    def test_single_decision(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        assert ledger.decision_count == 1

    def test_multiple_decisions(self):
        d1 = build_phi_raster_decision(4, 5.0)
        d2 = build_phi_raster_decision(8, 3.0)
        ledger = build_phi_raster_ledger((d1, d2))
        assert ledger.decision_count == 2

    def test_has_stable_hash(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        assert len(ledger.stable_hash) == 64

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            build_phi_raster_ledger(())

    def test_non_tuple_rejected(self):
        with pytest.raises(TypeError):
            build_phi_raster_ledger([])

    def test_wrong_type_in_tuple_rejected(self):
        with pytest.raises(TypeError):
            build_phi_raster_ledger(("not_a_decision",))


# -----------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------


class TestExport:
    def test_export_ledger_dict(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        exported = export_phi_raster_ledger(ledger)
        assert isinstance(exported, dict)
        assert "decisions" in exported
        assert "stable_hash" in exported
        assert exported["decision_count"] == 1

    def test_export_bundle_is_json(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        bundle = export_phi_raster_bundle(ledger)
        parsed = json.loads(bundle)
        assert "sha256" in parsed
        assert "data" in parsed

    def test_export_bundle_sha256_valid(self):
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        bundle = export_phi_raster_bundle(ledger)
        parsed = json.loads(bundle)
        data_json = json.dumps(
            parsed["data"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=True,
        )
        expected_hash = hashlib.sha256(data_json.encode("utf-8")).hexdigest()
        assert parsed["sha256"] == expected_hash


# -----------------------------------------------------------------------
# 100-run replay determinism (SIS2_STABILITY_RING invariant)
# -----------------------------------------------------------------------


class TestReplayDeterminism:
    def test_span_100_run_replay(self):
        results = [build_phi_scanline_spans(8, 5.0) for _ in range(100)]
        first = results[0]
        for r in results[1:]:
            assert r == first

    def test_decision_100_run_replay(self):
        results = [build_phi_raster_decision(8, 5.0) for _ in range(100)]
        hashes = [r.stable_hash for r in results]
        assert len(set(hashes)) == 1

    def test_ledger_100_run_replay(self):
        decisions = tuple(
            build_phi_raster_decision(w, 5.0) for w in [4, 8]
        )
        results = [build_phi_raster_ledger(decisions) for _ in range(100)]
        hashes = [r.stable_hash for r in results]
        assert len(set(hashes)) == 1

    def test_export_byte_equality_100_runs(self):
        d = build_phi_raster_decision(6, 4.0)
        ledger = build_phi_raster_ledger((d,))
        bundles = [export_phi_raster_bundle(ledger) for _ in range(100)]
        assert len(set(bundles)) == 1

    def test_restore_term_100_run_replay(self):
        results = [compute_phi_restore_term(0.42, 1.23) for _ in range(100)]
        assert len(set(results)) == 1


# -----------------------------------------------------------------------
# No decoder contamination
# -----------------------------------------------------------------------


class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """Module must not import anything from qec.decoder."""
        import qec.analysis.retro_phi_shell_rasterization_pipeline as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_no_decoder_in_sys_modules(self):
        """Importing the module must not pull in decoder modules."""
        decoder_modules_before = {
            k for k in sys.modules if "qec.decoder" in k
        }
        import importlib
        importlib.reload(
            sys.modules["qec.analysis.retro_phi_shell_rasterization_pipeline"]
        )
        decoder_modules_after = {
            k for k in sys.modules if "qec.decoder" in k
        }
        assert decoder_modules_after == decoder_modules_before


# -----------------------------------------------------------------------
# sys.modules side-effect proof
# -----------------------------------------------------------------------


class TestSysModulesSideEffect:
    def test_no_unexpected_modules(self):
        """Import must not introduce unexpected sys.modules entries."""
        baseline = set(sys.modules.keys())
        import importlib
        importlib.reload(
            sys.modules["qec.analysis.retro_phi_shell_rasterization_pipeline"]
        )
        after = set(sys.modules.keys())
        new_modules = after - baseline
        # Only stdlib / already-present modules should appear
        for m in new_modules:
            assert not m.startswith("qec.decoder")
            assert not m.startswith("qec.experiments")
            assert not m.startswith("qec.sims")


# -----------------------------------------------------------------------
# Monotonic shell ordering
# -----------------------------------------------------------------------


class TestMonotonicShellOrdering:
    def test_phi_shells_strictly_increasing(self):
        for i in range(1, len(PHI_SHELLS)):
            assert PHI_SHELLS[i] > PHI_SHELLS[i - 1]

    def test_shell_descriptors_strictly_increasing(self):
        shells = build_phi_shell_descriptors()
        for i in range(1, len(shells)):
            assert shells[i].shell_value > shells[i - 1].shell_value

    def test_shell_lower_bounds_increasing(self):
        shells = build_phi_shell_descriptors()
        for i in range(1, len(shells)):
            assert shells[i].lower_bound > shells[i - 1].lower_bound


# -----------------------------------------------------------------------
# Visibility class boundaries (edge cases)
# -----------------------------------------------------------------------


class TestVisibilityBoundaries:
    def test_exact_boundary_1_309(self):
        assert classify_shell_visibility(1.309) == NEAR_SHELL

    def test_just_above_1_309(self):
        assert classify_shell_visibility(1.3091) == MID_SHELL

    def test_exact_boundary_2_118(self):
        assert classify_shell_visibility(2.118) == MID_SHELL

    def test_just_above_2_118(self):
        assert classify_shell_visibility(2.1181) == OUTER_SHELL

    def test_exact_boundary_3_427(self):
        assert classify_shell_visibility(3.427) == OUTER_SHELL

    def test_just_above_3_427(self):
        assert classify_shell_visibility(3.4271) == RESONANCE_NODE

    def test_exact_boundary_5_545(self):
        assert classify_shell_visibility(5.545) == RESONANCE_NODE

    def test_just_above_5_545(self):
        assert classify_shell_visibility(5.5451) == WIGGLE_ZONE

    def test_zero_is_near_shell(self):
        assert classify_shell_visibility(0.0) == NEAR_SHELL


# -----------------------------------------------------------------------
# Hash uniqueness
# -----------------------------------------------------------------------


class TestHashUniqueness:
    def test_different_widths_produce_different_hashes(self):
        d1 = build_phi_raster_decision(4, 5.0)
        d2 = build_phi_raster_decision(8, 5.0)
        assert d1.stable_hash != d2.stable_hash

    def test_different_depths_produce_different_hashes(self):
        d1 = build_phi_raster_decision(4, 5.0)
        d2 = build_phi_raster_decision(4, 3.0)
        assert d1.stable_hash != d2.stable_hash

    def test_span_hashes_unique_across_indices(self):
        spans = build_phi_scanline_spans(10, 6.0)
        hashes = [s.stable_hash for s in spans]
        # Most should be unique (some may collide if depths match)
        assert len(set(hashes)) >= len(hashes) // 2


# -----------------------------------------------------------------------
# HARDENING: sentinel semantics (PATCH 1 verification)
# -----------------------------------------------------------------------


class TestSentinelSemantics:
    def test_export_last_shell_upper_bound_is_none(self):
        """Exported last shell upper_bound must be None (JSON null)."""
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        exported = export_phi_raster_ledger(ledger)
        last_shell = exported["decisions"][0]["shells"][-1]
        assert last_shell["upper_bound"] is None

    def test_export_non_final_shells_have_float_upper_bound(self):
        """Non-final shells must have finite float upper_bound in export."""
        d = build_phi_raster_decision(4, 5.0)
        ledger = build_phi_raster_ledger((d,))
        exported = export_phi_raster_ledger(ledger)
        shells = exported["decisions"][0]["shells"]
        for s in shells[:-1]:
            assert isinstance(s["upper_bound"], float)

    def test_canonical_bound_value_finite(self):
        assert _canonical_bound_value(1.5) == round(1.5, FLOAT_PRECISION)

    def test_canonical_bound_value_inf(self):
        assert _canonical_bound_value(float("inf")) == "INF"

    def test_canonical_bound_value_neg_inf(self):
        assert _canonical_bound_value(float("-inf")) == "INF"

    def test_hash_uses_inf_string_not_9999(self):
        """Shell hash must use 'INF', never 9999.0."""
        import qec.analysis.retro_phi_shell_rasterization_pipeline as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        assert "9999" not in source

    def test_sentinel_replay_deterministic(self):
        """100-run replay with INF sentinel must be stable."""
        results = [build_phi_shell_descriptors() for _ in range(100)]
        hashes = [tuple(s.stable_hash for s in r) for r in results]
        assert len(set(hashes)) == 1


# -----------------------------------------------------------------------
# HARDENING: midpoint alignment (PATCH 2 + 3 verification)
# -----------------------------------------------------------------------


class TestMidpointAlignment:
    def test_midpoints_count(self):
        mps = _compute_phi_midpoints()
        assert len(mps) == len(PHI_SHELLS) - 1

    def test_midpoints_match_shell_averages(self):
        mps = _compute_phi_midpoints()
        for i, mp in enumerate(mps):
            expected = round((PHI_SHELLS[i] + PHI_SHELLS[i + 1]) / 2.0, FLOAT_PRECISION)
            assert mp == expected

    def test_exact_midpoint_ties_go_to_lower_shell(self):
        """At midpoint boundary, depth <= midpoint -> lower (nearer) class."""
        mps = _compute_phi_midpoints()
        classes = [NEAR_SHELL, MID_SHELL, OUTER_SHELL, RESONANCE_NODE]
        for mp, expected_class in zip(mps, classes):
            assert classify_shell_visibility(mp) == expected_class

    def test_half_open_interval_upper_exclusive(self):
        """Just above midpoint falls into next class."""
        mps = _compute_phi_midpoints()
        next_classes = [MID_SHELL, OUTER_SHELL, RESONANCE_NODE, WIGGLE_ZONE]
        for mp, expected_class in zip(mps, next_classes):
            assert classify_shell_visibility(mp + 0.0001) == expected_class

    def test_final_shell_inclusive_infinity(self):
        """Very large depth always maps to WIGGLE_ZONE."""
        assert classify_shell_visibility(1e6) == WIGGLE_ZONE

    def test_classifier_matches_computed_midpoints(self):
        """Classifier boundaries are exactly the computed midpoints."""
        mps = _compute_phi_midpoints()
        # Each midpoint is the transition point
        for mp in mps:
            below = classify_shell_visibility(mp)
            above = classify_shell_visibility(mp + 0.0001)
            assert below != above


# -----------------------------------------------------------------------
# HARDENING: derived boundaries (PATCH 3 verification)
# -----------------------------------------------------------------------


class TestDerivedBoundaries:
    def test_no_hardcoded_constants_in_classifier(self):
        """classify_shell_visibility must not contain hardcoded boundary values."""
        import inspect
        source = inspect.getsource(classify_shell_visibility)
        # Must not contain any of the old hardcoded values
        assert "1.309" not in source
        assert "2.118" not in source
        assert "3.427" not in source
        assert "5.545" not in source

    def test_shell_descriptors_use_derived_midpoints(self):
        """Shell descriptor boundaries must match _compute_phi_midpoints."""
        mps = _compute_phi_midpoints()
        shells = build_phi_shell_descriptors()
        for i in range(1, len(shells)):
            assert shells[i].lower_bound == mps[i - 1]
        for i in range(len(shells) - 1):
            assert shells[i].upper_bound == mps[i]


# -----------------------------------------------------------------------
# HARDENING: width==1 semantics (PATCH 4 verification)
# -----------------------------------------------------------------------


class TestWidthOneSemantics:
    def test_width_one_depth_is_zero(self):
        """Single-column span must have depth 0.0 (near-plane origin)."""
        spans = build_phi_scanline_spans(1, 10.0)
        assert spans[0].depth == 0.0

    def test_width_one_visibility_is_near_shell(self):
        """Single-column at depth 0.0 must classify as NEAR_SHELL."""
        spans = build_phi_scanline_spans(1, 10.0)
        assert spans[0].visibility_class == NEAR_SHELL

    def test_width_one_phi_shell_is_first(self):
        """Depth 0.0 quantizes to first phi shell (1.0)."""
        spans = build_phi_scanline_spans(1, 10.0)
        assert spans[0].phi_shell == PHI_SHELLS[0]


# -----------------------------------------------------------------------
# HARDENING: file handle hygiene (PATCH 5 verification)
# -----------------------------------------------------------------------


class TestFileHandleHygiene:
    def test_no_bare_open_in_tests(self):
        """This test file must not contain bare open() without context manager."""
        import re
        import pathlib
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        # Match lines that assign result of open() directly (bare open)
        # e.g. "source = open(..." but NOT "with open(..."
        bare_pattern = re.compile(r"^\s+\w+\s*=\s*open\(")
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if bare_pattern.match(line):
                raise AssertionError(
                    f"Bare open() found at line {i + 1}: {line.strip()}"
                )
