"""Tests for v137.0.12 — Retro Lighting + Shading Model."""

import json
import sys

import pytest

from qec.analysis.retro_lighting_shading_model import (
    AMBIENT_RETRO,
    BRIGHT,
    DARK,
    DEPTH_SHADED,
    DIM,
    FULL_SHADOW,
    GIGACOLOR_RETRO,
    HARD_SHADOW,
    HYBRID_LUMINANCE,
    MID,
    NO_SHADOW,
    QUAD_BLEND,
    RETRO_LIGHTING_SHADING_VERSION,
    SECTOR_BANDED,
    SOFT_SHADOW,
    STANDARD_RETRO,
    TRIPLE_HEAD_RETRO,
    VOODOO_FSAA,
    AA_EDGE,
    AA_FULL,
    AA_NONE,
    AA_SUPER,
    CENTER_HEAD,
    DEFAULT_LANE_COUNT,
    LEFT_HEAD,
    RIGHT_HEAD,
    RetroLightZone,
    RetroLightingLedger,
    RetroLuminanceViewport,
    RetroShadingDecision,
    RetroShadingLane,
    build_light_zones,
    build_luminance_symbolic_trace,
    build_luminance_viewports,
    build_retro_lighting_ledger,
    build_retro_shading_decision,
    build_shading_lanes,
    classify_antialias_band,
    classify_gigacolor,
    classify_light,
    classify_lighting_mode,
    classify_shadow,
    compute_contrast_ratio,
    export_retro_lighting_ledger,
    export_retro_shading_bundle,
)


# -----------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------


class TestVersion:
    def test_version_string(self):
        assert RETRO_LIGHTING_SHADING_VERSION == "v137.0.12"


# -----------------------------------------------------------------------
# Frozen dataclasses
# -----------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_light_zone_frozen(self):
        z = RetroLightZone(0, 0.5, BRIGHT, NO_SHADOW, "abc")
        with pytest.raises(AttributeError):
            z.zone_index = 1

    def test_shading_lane_frozen(self):
        l = RetroShadingLane(0, 0.5, AA_NONE, "abc")
        with pytest.raises(AttributeError):
            l.lane_index = 1

    def test_luminance_viewport_frozen(self):
        v = RetroLuminanceViewport(0, LEFT_HEAD, 0.5, "abc")
        with pytest.raises(AttributeError):
            v.viewport_index = 1

    def test_shading_decision_frozen(self):
        d = build_retro_shading_decision([0.5, 0.3, 0.8], False, False)
        with pytest.raises(AttributeError):
            d.contrast_ratio = 0.0

    def test_lighting_ledger_frozen(self):
        d = build_retro_shading_decision([0.5], False, False)
        ledger = build_retro_lighting_ledger([d])
        with pytest.raises(AttributeError):
            ledger.decision_count = 0


# -----------------------------------------------------------------------
# Light classification
# -----------------------------------------------------------------------


class TestClassifyLight:
    def test_bright(self):
        assert classify_light(0.9) == BRIGHT

    def test_mid(self):
        assert classify_light(0.6) == MID

    def test_dim(self):
        assert classify_light(0.3) == DIM

    def test_dark(self):
        assert classify_light(0.1) == DARK

    def test_boundary_bright(self):
        assert classify_light(0.75) == BRIGHT

    def test_boundary_mid(self):
        assert classify_light(0.5) == MID

    def test_boundary_dim(self):
        assert classify_light(0.25) == DIM


# -----------------------------------------------------------------------
# Shadow classification
# -----------------------------------------------------------------------


class TestClassifyShadow:
    def test_no_shadow(self):
        assert classify_shadow(0.9) == NO_SHADOW

    def test_soft(self):
        assert classify_shadow(0.6) == SOFT_SHADOW

    def test_hard(self):
        assert classify_shadow(0.3) == HARD_SHADOW

    def test_full(self):
        assert classify_shadow(0.1) == FULL_SHADOW


# -----------------------------------------------------------------------
# Lighting mode classification
# -----------------------------------------------------------------------


class TestClassifyLightingMode:
    def test_ambient(self):
        assert classify_lighting_mode(1, False) == AMBIENT_RETRO

    def test_sector_banded(self):
        assert classify_lighting_mode(3, False) == SECTOR_BANDED

    def test_depth_shaded(self):
        assert classify_lighting_mode(1, True) == DEPTH_SHADED

    def test_hybrid(self):
        assert classify_lighting_mode(3, True) == HYBRID_LUMINANCE


# -----------------------------------------------------------------------
# Antialias band classification
# -----------------------------------------------------------------------


class TestClassifyAntialiasBand:
    def test_single_lane(self):
        assert classify_antialias_band(0, 1) == AA_NONE

    def test_first_lane(self):
        assert classify_antialias_band(0, 4) == AA_NONE

    def test_second_lane(self):
        assert classify_antialias_band(1, 4) == AA_EDGE

    def test_middle_lane(self):
        assert classify_antialias_band(2, 4) == AA_FULL

    def test_last_lane(self):
        assert classify_antialias_band(3, 4) == AA_SUPER


# -----------------------------------------------------------------------
# Gigacolor classification
# -----------------------------------------------------------------------


class TestClassifyGigacolor:
    def test_standard(self):
        assert classify_gigacolor(2) == STANDARD_RETRO

    def test_gigacolor(self):
        assert classify_gigacolor(3) == GIGACOLOR_RETRO


# -----------------------------------------------------------------------
# Lane count
# -----------------------------------------------------------------------


class TestLaneCount:
    def test_default_lane_count(self):
        assert DEFAULT_LANE_COUNT == 4

    def test_decision_default_lanes(self):
        d = build_retro_shading_decision([0.5], False, False)
        assert len(d.shading_lanes) == 4

    def test_custom_lane_count(self):
        d = build_retro_shading_decision([0.5], False, False, lane_count=6)
        assert len(d.shading_lanes) == 6


# -----------------------------------------------------------------------
# Viewport count
# -----------------------------------------------------------------------


class TestViewportCount:
    def test_always_three_viewports(self):
        d = build_retro_shading_decision([0.5, 0.3, 0.8], False, False)
        assert len(d.viewports) == 3

    def test_viewport_classes(self):
        d = build_retro_shading_decision([0.5, 0.3, 0.8], False, False)
        classes = [v.viewport_class for v in d.viewports]
        assert classes == [LEFT_HEAD, CENTER_HEAD, RIGHT_HEAD]


# -----------------------------------------------------------------------
# Contrast bounds
# -----------------------------------------------------------------------


class TestContrastBounds:
    def test_contrast_zero(self):
        assert compute_contrast_ratio((0.5, 0.5, 0.5)) == 0.0

    def test_contrast_full(self):
        assert compute_contrast_ratio((0.0, 1.0)) == 1.0

    def test_contrast_clamped(self):
        c = compute_contrast_ratio((0.2, 0.7))
        assert 0.0 <= c <= 1.0

    def test_contrast_in_decision(self):
        d = build_retro_shading_decision([0.1, 0.9], False, False)
        assert 0.0 <= d.contrast_ratio <= 1.0
        assert d.contrast_ratio == pytest.approx(0.8)


# -----------------------------------------------------------------------
# Blend modes
# -----------------------------------------------------------------------


class TestBlendModes:
    def test_quad_blend_no_smoothing(self):
        d = build_retro_shading_decision([0.5], False, False)
        assert d.blend_mode == QUAD_BLEND

    def test_voodoo_fsaa_smoothing(self):
        d = build_retro_shading_decision([0.5], True, False)
        assert d.blend_mode == VOODOO_FSAA


# -----------------------------------------------------------------------
# Symbolic trace
# -----------------------------------------------------------------------


class TestSymbolicTrace:
    def test_trace_format(self):
        trace = build_luminance_symbolic_trace(QUAD_BLEND, GIGACOLOR_RETRO)
        assert trace == "QUAD_BLEND -> TRIPLE_HEAD_RETRO -> GIGACOLOR_RETRO"

    def test_trace_in_decision(self):
        d = build_retro_shading_decision([0.5, 0.3, 0.8], True, False)
        assert VOODOO_FSAA in d.luminance_symbolic_trace
        assert TRIPLE_HEAD_RETRO in d.luminance_symbolic_trace
        assert GIGACOLOR_RETRO in d.luminance_symbolic_trace


# -----------------------------------------------------------------------
# Hashing determinism
# -----------------------------------------------------------------------


class TestHashingDeterminism:
    def test_zone_hash_stable(self):
        zones_a = build_light_zones((0.5, 0.3))
        zones_b = build_light_zones((0.5, 0.3))
        assert zones_a[0].stable_hash == zones_b[0].stable_hash
        assert zones_a[1].stable_hash == zones_b[1].stable_hash

    def test_lane_hash_stable(self):
        lanes_a = build_shading_lanes(4, False)
        lanes_b = build_shading_lanes(4, False)
        for a, b in zip(lanes_a, lanes_b):
            assert a.stable_hash == b.stable_hash

    def test_viewport_hash_stable(self):
        vp_a = build_luminance_viewports((0.5, 0.3, 0.8))
        vp_b = build_luminance_viewports((0.5, 0.3, 0.8))
        for a, b in zip(vp_a, vp_b):
            assert a.stable_hash == b.stable_hash

    def test_decision_hash_stable(self):
        d1 = build_retro_shading_decision([0.5, 0.3], False, True)
        d2 = build_retro_shading_decision([0.5, 0.3], False, True)
        assert d1.stable_hash == d2.stable_hash

    def test_different_inputs_different_hash(self):
        d1 = build_retro_shading_decision([0.5], False, False)
        d2 = build_retro_shading_decision([0.9], False, False)
        assert d1.stable_hash != d2.stable_hash


# -----------------------------------------------------------------------
# Export equality
# -----------------------------------------------------------------------


class TestExportEquality:
    def test_export_deterministic(self):
        d = build_retro_shading_decision([0.5, 0.3, 0.8], True, True)
        b1 = export_retro_shading_bundle(d)
        b2 = export_retro_shading_bundle(d)
        assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)

    def test_ledger_export_deterministic(self):
        d = build_retro_shading_decision([0.5], False, False)
        ledger = build_retro_lighting_ledger([d])
        e1 = export_retro_lighting_ledger(ledger)
        e2 = export_retro_lighting_ledger(ledger)
        assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)


# -----------------------------------------------------------------------
# 25-run replay
# -----------------------------------------------------------------------


class TestReplay:
    def test_25_run_replay(self):
        levels = [0.1, 0.4, 0.7, 0.9]
        hashes = set()
        for _ in range(25):
            d = build_retro_shading_decision(levels, True, True)
            hashes.add(d.stable_hash)
        assert len(hashes) == 1

    def test_25_run_ledger_replay(self):
        d = build_retro_shading_decision([0.5, 0.8], False, False)
        hashes = set()
        for _ in range(25):
            ledger = build_retro_lighting_ledger([d])
            hashes.add(ledger.stable_hash)
        assert len(hashes) == 1


# -----------------------------------------------------------------------
# Invalid input rejection
# -----------------------------------------------------------------------


class TestInvalidInput:
    def test_string_luminance(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision("bad", False, False)

    def test_non_iterable_luminance(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision(42, False, False)

    def test_empty_luminance(self):
        with pytest.raises(ValueError):
            build_retro_shading_decision([], False, False)

    def test_out_of_range_luminance(self):
        with pytest.raises(ValueError):
            build_retro_shading_decision([1.5], False, False)

    def test_negative_luminance(self):
        with pytest.raises(ValueError):
            build_retro_shading_decision([-0.1], False, False)

    def test_bool_luminance_element(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision([True], False, False)

    def test_non_bool_smoothing(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision([0.5], 1, False)

    def test_non_bool_has_depth(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision([0.5], False, 1)

    def test_invalid_lane_count(self):
        with pytest.raises(TypeError):
            build_retro_shading_decision([0.5], False, False, lane_count="4")

    def test_zero_lane_count(self):
        with pytest.raises(ValueError):
            build_retro_shading_decision([0.5], False, False, lane_count=0)

    def test_ledger_string_input(self):
        with pytest.raises(TypeError):
            build_retro_lighting_ledger("bad")

    def test_ledger_wrong_type(self):
        with pytest.raises(TypeError):
            build_retro_lighting_ledger([42])

    def test_export_wrong_type(self):
        with pytest.raises(TypeError):
            export_retro_shading_bundle("bad")

    def test_ledger_export_wrong_type(self):
        with pytest.raises(TypeError):
            export_retro_lighting_ledger("bad")


# -----------------------------------------------------------------------
# No decoder contamination
# -----------------------------------------------------------------------


class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        import qec.analysis.retro_lighting_shading_model as mod
        source = open(mod.__file__).read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_no_decoder_in_sys_modules(self):
        # Ensure the module doesn't pull in decoder
        decoder_modules = [
            k for k in sys.modules
            if k.startswith("qec.decoder")
        ]
        # This is a soft check — decoder may be loaded by other tests
        # but this module should not import it
        import qec.analysis.retro_lighting_shading_model as mod
        source = open(mod.__file__).read()
        assert "import qec.decoder" not in source
