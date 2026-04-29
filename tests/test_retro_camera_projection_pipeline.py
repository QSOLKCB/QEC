"""Tests for v137.0.11 — Retro Camera + Projection Pipeline.

110–135 tests covering:
  frozen dataclasses, bool rejection, FOV boundary tests,
  projection mode classification, span generation, visibility classes,
  horizon classification, stable hashing, canonical export equality,
  100-run replay, invalid input rejection, iterable acceptance,
  no decoder contamination, TRUE_3D vs PSEUDO_3D, symbolic trace.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from qec.analysis.retro_3d_world_modelling import (
    RetroWorldModel,
    build_primitive,
    build_retro_world_model,
    build_sector,
)
from qec.analysis.retro_camera_projection_pipeline import (
    CENTER_HORIZON,
    COMPACT_DEPTH,
    DENSE_DEPTH,
    FIXED_CAMERA,
    HIGH_HORIZON,
    HYBRID_CAMERA,
    LOW_HORIZON,
    MINIMAL_DEPTH,
    ORBITAL_CAMERA,
    ORTHOGRAPHIC_RETRO,
    PERSPECTIVE_RETRO,
    PSEUDO_DOOM_VIEW,
    RETRO_CAMERA_PROJECTION_VERSION,
    STRUCTURED_DEPTH,
    TILTED_HORIZON,
    TRACKING_CAMERA,
    RetroCameraProjectionDecision,
    build_projection_spans,
    build_projection_symbolic_trace,
    build_retro_camera_projection,
    build_retro_camera_projection_ledger,
    classify_camera,
    classify_depth_complexity,
    classify_horizon_line,
    classify_projection_mode,
    compute_visible_sector_count,
    export_retro_camera_projection_bundle,
    export_retro_camera_projection_ledger,
)


# ---------------------------------------------------------------------------
# Helpers — build test world models
# ---------------------------------------------------------------------------


def _make_pseudo_3d_world(sector_count: int = 3) -> RetroWorldModel:
    """Build a PSEUDO_3D world model with sectors."""
    sectors = [
        build_sector(f"s{i}", 0.0, 5.0, 4, 1)
        for i in range(sector_count)
    ]
    return build_retro_world_model(
        "PSEUDO_3D", [], sectors, (0.0, 0.0, 0.0),
    )


def _make_true_3d_world(
    primitive_count: int = 2,
    sector_count: int = 0,
) -> RetroWorldModel:
    """Build a TRUE_3D world model with primitives."""
    prims = [
        build_primitive(f"p{i}", "WALL", (float(i), 0.0, 0.0),
                        (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "BASIC")
        for i in range(primitive_count)
    ]
    sectors = [
        build_sector(f"s{i}", 0.0, 5.0, 4, 1)
        for i in range(sector_count)
    ]
    return build_retro_world_model(
        "TRUE_3D", prims, sectors, (0.0, 0.0, 0.0),
    )


def _make_empty_world(mode: str = "TRUE_3D") -> RetroWorldModel:
    """Build a world model with no primitives or sectors."""
    return build_retro_world_model(mode, [], [], (0.0, 0.0, 0.0))


def _make_decision(**kwargs: Any) -> RetroCameraProjectionDecision:
    """Build a projection decision with defaults."""
    world = kwargs.pop("world_model", _make_pseudo_3d_world())
    pos = kwargs.pop("position", (0.0, 0.0, 0.0))
    rot = kwargs.pop("rotation", (0.0, 0.0, 0.0))
    fov = kwargs.pop("fov_degrees", 90.0)
    near = kwargs.pop("near_plane", 0.1)
    far = kwargs.pop("far_plane", 100.0)
    return build_retro_camera_projection(
        world, pos, rot, fov, near, far,
    )


# ===================================================================
# 1. Frozen dataclass tests
# ===================================================================


class TestFrozenDataclasses:
    """All dataclasses must be frozen (immutable)."""

    def test_camera_state_frozen(self) -> None:
        d = _make_decision()
        with pytest.raises(FrozenInstanceError):
            d.camera_state.fov_degrees = 999.0  # type: ignore[misc]

    def test_camera_state_position_frozen(self) -> None:
        d = _make_decision()
        with pytest.raises(FrozenInstanceError):
            d.camera_state.position = (1.0, 1.0, 1.0)  # type: ignore[misc]

    def test_projection_span_frozen(self) -> None:
        d = _make_decision()
        if d.projection_spans:
            with pytest.raises(FrozenInstanceError):
                d.projection_spans[0].span_index = 99  # type: ignore[misc]

    def test_decision_frozen(self) -> None:
        d = _make_decision()
        with pytest.raises(FrozenInstanceError):
            d.projection_mode = "INVALID"  # type: ignore[misc]

    def test_ledger_frozen(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger([d])
        with pytest.raises(FrozenInstanceError):
            ledger.decision_count = 999  # type: ignore[misc]


# ===================================================================
# 2. Bool rejection tests
# ===================================================================


class TestBoolRejection:
    """Bool values must be rejected in numeric triples."""

    def test_position_bool_x(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(position=(True, 0.0, 0.0))

    def test_position_bool_y(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(position=(0.0, False, 0.0))

    def test_position_bool_z(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(position=(0.0, 0.0, True))

    def test_rotation_bool_pitch(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(rotation=(True, 0.0, 0.0))

    def test_rotation_bool_yaw(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(rotation=(0.0, True, 0.0))

    def test_rotation_bool_roll(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(rotation=(0.0, 0.0, False))

    def test_fov_bool(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(fov_degrees=True)

    def test_near_plane_bool(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(near_plane=True)

    def test_far_plane_bool(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            _make_decision(far_plane=True)


# ===================================================================
# 3. FOV boundary tests
# ===================================================================


class TestFOVBoundaries:
    """FOV must be in (0, 180]."""

    def test_fov_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="fov_degrees"):
            _make_decision(fov_degrees=0.0)

    def test_fov_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="fov_degrees"):
            _make_decision(fov_degrees=-10.0)

    def test_fov_over_180_rejected(self) -> None:
        with pytest.raises(ValueError, match="fov_degrees"):
            _make_decision(fov_degrees=181.0)

    def test_fov_180_accepted(self) -> None:
        d = _make_decision(fov_degrees=180.0)
        assert d.camera_state.fov_degrees == 180.0

    def test_fov_small_accepted(self) -> None:
        d = _make_decision(fov_degrees=1.0)
        assert d.camera_state.fov_degrees == 1.0

    def test_fov_90_accepted(self) -> None:
        d = _make_decision(fov_degrees=90.0)
        assert d.camera_state.fov_degrees == 90.0

    def test_fov_integer_accepted(self) -> None:
        d = _make_decision(fov_degrees=60)
        assert d.camera_state.fov_degrees == 60.0


# ===================================================================
# 4. Near/far plane validation
# ===================================================================


class TestNearFarPlanes:
    """Near and far plane constraints."""

    def test_negative_near_rejected(self) -> None:
        with pytest.raises(ValueError, match="near_plane"):
            _make_decision(near_plane=-1.0)

    def test_negative_far_rejected(self) -> None:
        with pytest.raises(ValueError, match="far_plane"):
            _make_decision(far_plane=-1.0)

    def test_far_equal_near_rejected(self) -> None:
        with pytest.raises(ValueError, match="far_plane"):
            _make_decision(near_plane=10.0, far_plane=10.0)

    def test_far_less_than_near_rejected(self) -> None:
        with pytest.raises(ValueError, match="far_plane"):
            _make_decision(near_plane=100.0, far_plane=1.0)

    def test_zero_near_accepted(self) -> None:
        d = _make_decision(near_plane=0.0, far_plane=100.0)
        assert d.camera_state.near_plane == 0.0


# ===================================================================
# 5. Projection mode classification
# ===================================================================


class TestProjectionModeClassification:
    """Projection mode derived from world model."""

    def test_pseudo_3d_gives_pseudo_doom_view(self) -> None:
        world = _make_pseudo_3d_world()
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == PSEUDO_DOOM_VIEW

    def test_true_3d_primitives_gives_perspective_retro(self) -> None:
        world = _make_true_3d_world(primitive_count=3, sector_count=0)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == PERSPECTIVE_RETRO

    def test_true_3d_mixed_gives_hybrid_camera(self) -> None:
        world = _make_true_3d_world(primitive_count=2, sector_count=2)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == HYBRID_CAMERA

    def test_true_3d_empty_gives_orthographic(self) -> None:
        world = _make_empty_world("TRUE_3D")
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == ORTHOGRAPHIC_RETRO

    def test_classify_projection_mode_direct(self) -> None:
        assert classify_projection_mode("PSEUDO_3D", 3, 0) == PSEUDO_DOOM_VIEW
        assert classify_projection_mode("TRUE_3D", 0, 5) == PERSPECTIVE_RETRO
        assert classify_projection_mode("TRUE_3D", 2, 3) == HYBRID_CAMERA
        assert classify_projection_mode("TRUE_3D", 0, 0) == ORTHOGRAPHIC_RETRO


# ===================================================================
# 6. Span generation tests
# ===================================================================


class TestSpanGeneration:
    """Projection spans — Doom-style column slices."""

    def test_spans_count_matches_visible_sectors(self) -> None:
        world = _make_pseudo_3d_world(sector_count=5)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert len(d.projection_spans) == d.visible_sector_count

    def test_span_indices_sequential(self) -> None:
        world = _make_pseudo_3d_world(sector_count=4)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        for i, span in enumerate(d.projection_spans):
            assert span.span_index == i

    def test_span_depth_buckets_sequential(self) -> None:
        world = _make_pseudo_3d_world(sector_count=4)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        for i, span in enumerate(d.projection_spans):
            assert span.depth_bucket == i

    def test_spans_have_positive_dimensions(self) -> None:
        world = _make_pseudo_3d_world(sector_count=3)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        for span in d.projection_spans:
            assert span.projected_width > 0
            assert span.projected_height > 0

    def test_spans_width_decreases_with_depth(self) -> None:
        """Nearer spans should be wider than farther spans."""
        world = _make_pseudo_3d_world(sector_count=4)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        if len(d.projection_spans) >= 2:
            first = d.projection_spans[0]
            last = d.projection_spans[-1]
            assert first.projected_width >= last.projected_width

    def test_empty_world_no_spans(self) -> None:
        world = _make_empty_world("TRUE_3D")
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert len(d.projection_spans) == 0

    def test_build_projection_spans_direct_zero(self) -> None:
        spans = build_projection_spans(0, 90.0, 0.1, 100.0)
        assert spans == ()

    def test_build_projection_spans_direct_single(self) -> None:
        spans = build_projection_spans(1, 90.0, 0.1, 100.0)
        assert len(spans) == 1
        assert spans[0].span_index == 0


# ===================================================================
# 7. Visibility classes
# ===================================================================


class TestVisibilityClasses:
    """Span visibility classification."""

    def test_first_span_visible(self) -> None:
        world = _make_pseudo_3d_world(sector_count=8)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        if len(d.projection_spans) >= 4:
            assert d.projection_spans[0].visibility_class == "VISIBLE"

    def test_last_span_occluded(self) -> None:
        world = _make_pseudo_3d_world(sector_count=8)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        if len(d.projection_spans) >= 4:
            assert d.projection_spans[-1].visibility_class == "OCCLUDED"

    def test_visibility_classes_valid(self) -> None:
        valid = {"VISIBLE", "PARTIAL", "DISTANT", "OCCLUDED"}
        world = _make_pseudo_3d_world(sector_count=8)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        for span in d.projection_spans:
            assert span.visibility_class in valid

    def test_single_span_visible(self) -> None:
        """A single span should be classified as VISIBLE."""
        spans = build_projection_spans(1, 90.0, 0.1, 100.0)
        assert spans[0].visibility_class == "VISIBLE"


# ===================================================================
# 8. Horizon line classification
# ===================================================================


class TestHorizonClassification:
    """Horizon line derived from camera rotation."""

    def test_zero_rotation_center(self) -> None:
        assert classify_horizon_line((0.0, 0.0, 0.0)) == CENTER_HORIZON

    def test_looking_up_low_horizon(self) -> None:
        assert classify_horizon_line((20.0, 0.0, 0.0)) == LOW_HORIZON

    def test_looking_down_high_horizon(self) -> None:
        assert classify_horizon_line((-20.0, 0.0, 0.0)) == HIGH_HORIZON

    def test_roll_tilted_horizon(self) -> None:
        assert classify_horizon_line((0.0, 0.0, 10.0)) == TILTED_HORIZON

    def test_tilted_takes_priority(self) -> None:
        """Roll > 5.0 should produce TILTED even with large pitch."""
        assert classify_horizon_line((30.0, 0.0, 10.0)) == TILTED_HORIZON

    def test_boundary_pitch_15_center(self) -> None:
        assert classify_horizon_line((15.0, 0.0, 0.0)) == CENTER_HORIZON

    def test_boundary_pitch_neg15_center(self) -> None:
        assert classify_horizon_line((-15.0, 0.0, 0.0)) == CENTER_HORIZON

    def test_boundary_roll_5_center(self) -> None:
        assert classify_horizon_line((0.0, 0.0, 5.0)) == CENTER_HORIZON

    def test_decision_horizon_propagated(self) -> None:
        d = _make_decision(rotation=(30.0, 0.0, 0.0))
        assert d.horizon_line_class == LOW_HORIZON


# ===================================================================
# 9. Depth complexity classification
# ===================================================================


class TestDepthComplexity:
    """Depth complexity derived from visible spans and sectors."""

    def test_minimal_depth(self) -> None:
        assert classify_depth_complexity(1, 1) == MINIMAL_DEPTH

    def test_compact_depth(self) -> None:
        assert classify_depth_complexity(3, 3) == COMPACT_DEPTH

    def test_structured_depth(self) -> None:
        assert classify_depth_complexity(5, 5) == STRUCTURED_DEPTH

    def test_dense_depth(self) -> None:
        assert classify_depth_complexity(7, 7) == DENSE_DEPTH

    def test_zero_minimal(self) -> None:
        assert classify_depth_complexity(0, 0) == MINIMAL_DEPTH


# ===================================================================
# 10. Camera class classification
# ===================================================================


class TestCameraClassification:
    """Camera class from position and rotation."""

    def test_fixed_camera(self) -> None:
        assert classify_camera((0, 0, 0), (0.0, 0.0, 0.0)) == FIXED_CAMERA

    def test_tracking_camera(self) -> None:
        assert classify_camera((0, 0, 0), (10.0, 5.0, 0.0)) == TRACKING_CAMERA

    def test_orbital_camera(self) -> None:
        assert classify_camera((0, 0, 0), (0.0, 60.0, 0.0)) == ORBITAL_CAMERA

    def test_orbital_negative_yaw(self) -> None:
        assert classify_camera((0, 0, 0), (0.0, -60.0, 0.0)) == ORBITAL_CAMERA


# ===================================================================
# 11. Stable hashing
# ===================================================================


class TestStableHashing:
    """SHA-256 hashes must be deterministic and stable."""

    def test_camera_state_hash_is_sha256(self) -> None:
        d = _make_decision()
        assert len(d.camera_state.stable_hash) == 64

    def test_span_hash_is_sha256(self) -> None:
        d = _make_decision()
        for span in d.projection_spans:
            assert len(span.stable_hash) == 64

    def test_decision_hash_is_sha256(self) -> None:
        d = _make_decision()
        assert len(d.stable_hash) == 64

    def test_ledger_hash_is_sha256(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger([d])
        assert len(ledger.stable_hash) == 64

    def test_same_input_same_hash(self) -> None:
        d1 = _make_decision()
        d2 = _make_decision()
        assert d1.stable_hash == d2.stable_hash

    def test_different_input_different_hash(self) -> None:
        d1 = _make_decision(fov_degrees=90.0)
        d2 = _make_decision(fov_degrees=60.0)
        assert d1.stable_hash != d2.stable_hash

    def test_camera_hash_differs_with_position(self) -> None:
        d1 = _make_decision(position=(0.0, 0.0, 0.0))
        d2 = _make_decision(position=(1.0, 0.0, 0.0))
        assert d1.camera_state.stable_hash != d2.camera_state.stable_hash

    def test_all_hex_chars(self) -> None:
        d = _make_decision()
        assert all(c in "0123456789abcdef" for c in d.stable_hash)


# ===================================================================
# 12. Canonical export equality
# ===================================================================


class TestCanonicalExport:
    """Export bundles must be deterministic and complete."""

    def test_bundle_keys(self) -> None:
        d = _make_decision()
        bundle = export_retro_camera_projection_bundle(d)
        expected = {
            "camera_state", "depth_complexity_class", "horizon_line_class",
            "projection_mode", "projection_spans", "projection_symbolic_trace",
            "stable_hash", "version", "visible_sector_count",
        }
        assert set(bundle.keys()) == expected

    def test_bundle_camera_state_keys(self) -> None:
        d = _make_decision()
        bundle = export_retro_camera_projection_bundle(d)
        expected = {
            "camera_class", "far_plane", "fov_degrees", "near_plane",
            "position", "rotation", "stable_hash", "version",
        }
        assert set(bundle["camera_state"].keys()) == expected

    def test_bundle_span_keys(self) -> None:
        d = _make_decision()
        bundle = export_retro_camera_projection_bundle(d)
        if bundle["projection_spans"]:
            expected = {
                "depth_bucket", "projected_height", "projected_width",
                "span_index", "stable_hash", "version", "visibility_class",
            }
            assert set(bundle["projection_spans"][0].keys()) == expected

    def test_export_deterministic(self) -> None:
        d1 = _make_decision()
        d2 = _make_decision()
        b1 = export_retro_camera_projection_bundle(d1)
        b2 = export_retro_camera_projection_bundle(d2)
        j1 = json.dumps(b1, sort_keys=True)
        j2 = json.dumps(b2, sort_keys=True)
        assert j1 == j2

    def test_ledger_export_keys(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger([d])
        bundle = export_retro_camera_projection_ledger(ledger)
        assert set(bundle.keys()) == {"decision_count", "decisions", "stable_hash"}

    def test_ledger_export_count(self) -> None:
        d1 = _make_decision()
        d2 = _make_decision(fov_degrees=60.0)
        ledger = build_retro_camera_projection_ledger([d1, d2])
        bundle = export_retro_camera_projection_ledger(ledger)
        assert bundle["decision_count"] == 2
        assert len(bundle["decisions"]) == 2


# ===================================================================
# 13. 100-run replay
# ===================================================================


class TestReplayDeterminism:
    """100-run replay must produce byte-identical results."""

    def test_100_run_decision_replay(self) -> None:
        world = _make_pseudo_3d_world(sector_count=5)
        reference = build_retro_camera_projection(
            world, (1.0, 2.0, 3.0), (10.0, 20.0, 0.0),
            90.0, 0.1, 100.0,
        )
        ref_bundle = export_retro_camera_projection_bundle(reference)
        ref_json = json.dumps(ref_bundle, sort_keys=True, separators=(",", ":"))

        for _ in range(100):
            d = build_retro_camera_projection(
                world, (1.0, 2.0, 3.0), (10.0, 20.0, 0.0),
                90.0, 0.1, 100.0,
            )
            bundle = export_retro_camera_projection_bundle(d)
            j = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
            assert j == ref_json

    def test_100_run_ledger_replay(self) -> None:
        world = _make_pseudo_3d_world(sector_count=3)
        d1 = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        d2 = build_retro_camera_projection(
            world, (5, 5, 5), (10, 0, 0), 60.0, 1.0, 200.0,
        )
        reference = build_retro_camera_projection_ledger([d1, d2])
        ref_bundle = export_retro_camera_projection_ledger(reference)
        ref_json = json.dumps(ref_bundle, sort_keys=True, separators=(",", ":"))

        for _ in range(100):
            ledger = build_retro_camera_projection_ledger([d1, d2])
            bundle = export_retro_camera_projection_ledger(ledger)
            j = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
            assert j == ref_json


# ===================================================================
# 14. Invalid input rejection
# ===================================================================


class TestInvalidInputRejection:
    """Invalid inputs must be rejected with clear errors."""

    def test_world_model_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="RetroWorldModel"):
            build_retro_camera_projection(
                "not_a_model", (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
            )

    def test_world_model_none(self) -> None:
        with pytest.raises(TypeError):
            build_retro_camera_projection(
                None, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
            )

    def test_position_string(self) -> None:
        with pytest.raises(TypeError, match="non-string"):
            _make_decision(position="abc")

    def test_position_none(self) -> None:
        with pytest.raises(TypeError, match="iterable"):
            _make_decision(position=None)

    def test_position_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="3 elements"):
            _make_decision(position=(1.0, 2.0))

    def test_rotation_string(self) -> None:
        with pytest.raises(TypeError, match="non-string"):
            _make_decision(rotation="abc")

    def test_fov_string(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            _make_decision(fov_degrees="wide")

    def test_fov_none(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            _make_decision(fov_degrees=None)

    def test_near_plane_string(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            _make_decision(near_plane="close")

    def test_far_plane_string(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            _make_decision(far_plane="far")

    def test_export_bundle_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="RetroCameraProjectionDecision"):
            export_retro_camera_projection_bundle("not_a_decision")  # type: ignore

    def test_export_ledger_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="RetroCameraProjectionLedger"):
            export_retro_camera_projection_ledger("not_a_ledger")  # type: ignore

    def test_ledger_string_input(self) -> None:
        with pytest.raises(TypeError, match="non-string"):
            build_retro_camera_projection_ledger("abc")

    def test_ledger_none_input(self) -> None:
        with pytest.raises(TypeError, match="iterable"):
            build_retro_camera_projection_ledger(None)

    def test_ledger_wrong_element_type(self) -> None:
        with pytest.raises(TypeError, match="RetroCameraProjectionDecision"):
            build_retro_camera_projection_ledger([42])

    def test_position_string_element(self) -> None:
        with pytest.raises(TypeError, match="int or float"):
            _make_decision(position=(1.0, "two", 3.0))


# ===================================================================
# 15. Iterable acceptance
# ===================================================================


class TestIterableAcceptance:
    """Non-string iterables must be accepted for collections."""

    def test_ledger_accepts_tuple(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger((d,))
        assert ledger.decision_count == 1

    def test_ledger_accepts_generator(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger(x for x in [d])
        assert ledger.decision_count == 1

    def test_ledger_accepts_list(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger([d])
        assert ledger.decision_count == 1

    def test_position_accepts_list(self) -> None:
        d = _make_decision(position=[1.0, 2.0, 3.0])
        assert d.camera_state.position == (1.0, 2.0, 3.0)

    def test_rotation_accepts_list(self) -> None:
        d = _make_decision(rotation=[10.0, 20.0, 0.0])
        assert d.camera_state.rotation == (10.0, 20.0, 0.0)

    def test_position_accepts_generator(self) -> None:
        d = _make_decision(position=(float(x) for x in range(3)))
        assert d.camera_state.position == (0.0, 1.0, 2.0)

    def test_position_accepts_integers(self) -> None:
        d = _make_decision(position=(1, 2, 3))
        assert d.camera_state.position == (1.0, 2.0, 3.0)


# ===================================================================
# 16. No decoder contamination
# ===================================================================


class TestNoDecoderContamination:
    """Camera projection pipeline must not import decoder."""

    def test_no_decoder_import(self) -> None:
        import qec.analysis.retro_camera_projection_pipeline as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_decoder_in_imports(self) -> None:
        """Verify no decoder imports exist in the module source."""
        import qec.analysis.retro_camera_projection_pipeline as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "qec.decoder" not in stripped, (
                f"Found decoder reference in: {stripped}"
            )


# ===================================================================
# 17. TRUE_3D vs PSEUDO_3D behavior
# ===================================================================


class TestTrue3DVsPseudo3D:
    """Different world modes produce different projection modes."""

    def test_pseudo_3d_projection_mode(self) -> None:
        world = _make_pseudo_3d_world()
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == PSEUDO_DOOM_VIEW

    def test_true_3d_primitives_projection_mode(self) -> None:
        world = _make_true_3d_world(primitive_count=3)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == PERSPECTIVE_RETRO

    def test_true_3d_empty_projection_mode(self) -> None:
        world = _make_empty_world("TRUE_3D")
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == ORTHOGRAPHIC_RETRO

    def test_true_3d_hybrid_projection_mode(self) -> None:
        world = _make_true_3d_world(primitive_count=2, sector_count=2)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert d.projection_mode == HYBRID_CAMERA

    def test_pseudo_3d_has_spans(self) -> None:
        world = _make_pseudo_3d_world(sector_count=3)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert len(d.projection_spans) > 0

    def test_true_3d_no_sectors_no_spans(self) -> None:
        world = _make_true_3d_world(primitive_count=3, sector_count=0)
        d = build_retro_camera_projection(
            world, (0, 0, 0), (0, 0, 0), 90.0, 0.1, 100.0,
        )
        assert len(d.projection_spans) == 0


# ===================================================================
# 18. Symbolic trace correctness
# ===================================================================


class TestSymbolicTrace:
    """Symbolic trace must be byte-identical and well-formed."""

    def test_trace_format(self) -> None:
        trace = build_projection_symbolic_trace(
            PSEUDO_DOOM_VIEW, 5, CENTER_HORIZON, STRUCTURED_DEPTH,
        )
        assert trace == "PSEUDO_DOOM_VIEW -> 5 spans -> CENTER_HORIZON -> STRUCTURED_DEPTH"

    def test_trace_in_decision(self) -> None:
        d = _make_decision()
        assert " -> " in d.projection_symbolic_trace
        assert "spans" in d.projection_symbolic_trace

    def test_trace_contains_projection_mode(self) -> None:
        d = _make_decision()
        assert d.projection_mode in d.projection_symbolic_trace

    def test_trace_contains_horizon(self) -> None:
        d = _make_decision()
        assert d.horizon_line_class in d.projection_symbolic_trace

    def test_trace_contains_depth_class(self) -> None:
        d = _make_decision()
        assert d.depth_complexity_class in d.projection_symbolic_trace

    def test_trace_replay_stable(self) -> None:
        d1 = _make_decision()
        d2 = _make_decision()
        assert d1.projection_symbolic_trace == d2.projection_symbolic_trace

    def test_trace_zero_spans(self) -> None:
        trace = build_projection_symbolic_trace(
            ORTHOGRAPHIC_RETRO, 0, CENTER_HORIZON, MINIMAL_DEPTH,
        )
        assert "0 spans" in trace


# ===================================================================
# 19. Version tag
# ===================================================================


class TestVersionTag:
    """All artifacts must carry v137.0.11 version."""

    def test_camera_state_version(self) -> None:
        d = _make_decision()
        assert d.camera_state.version == "v137.0.11"

    def test_span_version(self) -> None:
        d = _make_decision()
        for span in d.projection_spans:
            assert span.version == "v137.0.11"

    def test_decision_version(self) -> None:
        d = _make_decision()
        assert d.version == "v137.0.11"

    def test_module_version_constant(self) -> None:
        assert RETRO_CAMERA_PROJECTION_VERSION == "v137.0.11"


# ===================================================================
# 20. Visible sector count
# ===================================================================


class TestVisibleSectorCount:
    """Visible sector count derived deterministically."""

    def test_zero_sectors_zero_visible(self) -> None:
        assert compute_visible_sector_count(0, 90.0, 0.1, 100.0) == 0

    def test_sectors_bounded_by_total(self) -> None:
        for n in range(1, 10):
            visible = compute_visible_sector_count(n, 90.0, 0.1, 100.0)
            assert 0 < visible <= n

    def test_narrow_fov_reduces_visibility(self) -> None:
        wide = compute_visible_sector_count(10, 90.0, 0.1, 100.0)
        narrow = compute_visible_sector_count(10, 10.0, 0.1, 100.0)
        assert narrow <= wide

    def test_short_depth_reduces_visibility(self) -> None:
        long_range = compute_visible_sector_count(10, 90.0, 0.1, 100.0)
        short_range = compute_visible_sector_count(10, 90.0, 0.1, 5.0)
        assert short_range <= long_range


# ===================================================================
# 21. Ledger tests
# ===================================================================


class TestLedger:
    """Ledger construction and validation."""

    def test_empty_ledger(self) -> None:
        ledger = build_retro_camera_projection_ledger([])
        assert ledger.decision_count == 0
        assert ledger.decisions == ()

    def test_single_decision_ledger(self) -> None:
        d = _make_decision()
        ledger = build_retro_camera_projection_ledger([d])
        assert ledger.decision_count == 1

    def test_multi_decision_ledger(self) -> None:
        d1 = _make_decision(fov_degrees=90.0)
        d2 = _make_decision(fov_degrees=60.0)
        ledger = build_retro_camera_projection_ledger([d1, d2])
        assert ledger.decision_count == 2

    def test_ledger_preserves_order(self) -> None:
        d1 = _make_decision(fov_degrees=90.0)
        d2 = _make_decision(fov_degrees=60.0)
        ledger = build_retro_camera_projection_ledger([d1, d2])
        assert ledger.decisions[0].stable_hash == d1.stable_hash
        assert ledger.decisions[1].stable_hash == d2.stable_hash

    def test_ledger_hash_differs_with_order(self) -> None:
        d1 = _make_decision(fov_degrees=90.0)
        d2 = _make_decision(fov_degrees=60.0)
        l1 = build_retro_camera_projection_ledger([d1, d2])
        l2 = build_retro_camera_projection_ledger([d2, d1])
        assert l1.stable_hash != l2.stable_hash


# ===================================================================
# 22. Finalization hardening — v137.0.11 pre-tag
# ===================================================================


class TestFinalizationHardening:
    """Pre-tag semantic hardening tests."""

    def test_classify_projection_mode_no_projection_class_param(self) -> None:
        """classify_projection_mode takes 3 args: world_mode, sector_count,
        primitive_count. No projection_class parameter."""
        import inspect
        sig = inspect.signature(classify_projection_mode)
        params = list(sig.parameters.keys())
        assert params == ["world_mode", "sector_count", "primitive_count"]

    def test_classify_projection_mode_pseudo_3d_ignores_counts(self) -> None:
        """PSEUDO_3D always yields PSEUDO_DOOM_VIEW regardless of counts."""
        assert classify_projection_mode("PSEUDO_3D", 0, 0) == PSEUDO_DOOM_VIEW
        assert classify_projection_mode("PSEUDO_3D", 5, 5) == PSEUDO_DOOM_VIEW

    def test_visible_sector_fov_factor_uses_90(self) -> None:
        """fov_factor = min(fov / 90, 1.0), so fov=90 gives factor 1.0."""
        # With fov=90, large depth range, all sectors visible
        assert compute_visible_sector_count(5, 90.0, 0.1, 100.0) == 5

    def test_visible_sector_fov_over_90_capped(self) -> None:
        """fov > 90 still caps factor at 1.0."""
        v90 = compute_visible_sector_count(5, 90.0, 0.1, 100.0)
        v180 = compute_visible_sector_count(5, 180.0, 0.1, 100.0)
        assert v90 == v180

    def test_visible_sector_minimum_is_one(self) -> None:
        """Non-zero sector count always yields at least 1 visible."""
        assert compute_visible_sector_count(1, 1.0, 0.1, 100.0) >= 1
        assert compute_visible_sector_count(10, 1.0, 0.1, 5.0) >= 1

    def test_roadmap_version_metadata(self) -> None:
        """ROADMAP.md exposes canonical stable-tip metadata for v137.*."""
        import pathlib
        roadmap = pathlib.Path(__file__).resolve().parent.parent / "ROADMAP.md"
        assert roadmap.exists(), "ROADMAP.md must exist"
        text = roadmap.read_text(encoding="utf-8")
        assert "stable tip" in text.lower()
        assert "v137." in text.lower()
        assert "published tags are authoritative" in text.lower()
