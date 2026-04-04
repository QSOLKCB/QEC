"""v137.0.11 — Retro Camera + Projection Pipeline.

Deterministic camera-space view abstraction:

  retro world model
  -> camera pose
  -> deterministic view transform
  -> projection spans
  -> visibility abstraction
  -> replay-safe camera artifact

Inspired by:
  SGI geometry pipeline
  Doom pseudo-3D camera spans
  Amiga-era constrained scene projection

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from qec.analysis.retro_3d_world_modelling import (
    PSEUDO_3D,
    TRUE_3D,
    RetroWorldModel,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

RETRO_CAMERA_PROJECTION_VERSION: str = "v137.0.11"

# ---------------------------------------------------------------------------
# Constants — projection modes
# ---------------------------------------------------------------------------

PERSPECTIVE_RETRO: str = "PERSPECTIVE_RETRO"
PSEUDO_DOOM_VIEW: str = "PSEUDO_DOOM_VIEW"
ORTHOGRAPHIC_RETRO: str = "ORTHOGRAPHIC_RETRO"
HYBRID_CAMERA: str = "HYBRID_CAMERA"

VALID_PROJECTION_MODES: Tuple[str, ...] = (
    PERSPECTIVE_RETRO,
    PSEUDO_DOOM_VIEW,
    ORTHOGRAPHIC_RETRO,
    HYBRID_CAMERA,
)

# ---------------------------------------------------------------------------
# Constants — visibility classes
# ---------------------------------------------------------------------------

VISIBLE: str = "VISIBLE"
PARTIAL: str = "PARTIAL"
DISTANT: str = "DISTANT"
OCCLUDED: str = "OCCLUDED"

# ---------------------------------------------------------------------------
# Constants — horizon line classes
# ---------------------------------------------------------------------------

LOW_HORIZON: str = "LOW_HORIZON"
CENTER_HORIZON: str = "CENTER_HORIZON"
HIGH_HORIZON: str = "HIGH_HORIZON"
TILTED_HORIZON: str = "TILTED_HORIZON"

# ---------------------------------------------------------------------------
# Constants — depth complexity classes
# ---------------------------------------------------------------------------

MINIMAL_DEPTH: str = "MINIMAL_DEPTH"
COMPACT_DEPTH: str = "COMPACT_DEPTH"
STRUCTURED_DEPTH: str = "STRUCTURED_DEPTH"
DENSE_DEPTH: str = "DENSE_DEPTH"

# ---------------------------------------------------------------------------
# Constants — camera classes
# ---------------------------------------------------------------------------

FIXED_CAMERA: str = "FIXED_CAMERA"
TRACKING_CAMERA: str = "TRACKING_CAMERA"
ORBITAL_CAMERA: str = "ORBITAL_CAMERA"

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroCameraState:
    """Immutable camera state for retro projection pipeline."""

    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    fov_degrees: float
    near_plane: float
    far_plane: float
    camera_class: str
    stable_hash: str
    version: str = RETRO_CAMERA_PROJECTION_VERSION


@dataclass(frozen=True)
class RetroProjectionSpan:
    """Immutable projection span — Doom-style wall column abstraction."""

    span_index: int
    depth_bucket: int
    projected_width: float
    projected_height: float
    visibility_class: str
    stable_hash: str
    version: str = RETRO_CAMERA_PROJECTION_VERSION


@dataclass(frozen=True)
class RetroCameraProjectionDecision:
    """Immutable camera projection decision artifact."""

    camera_state: RetroCameraState
    projection_spans: Tuple[RetroProjectionSpan, ...]
    visible_sector_count: int
    depth_complexity_class: str
    projection_mode: str
    horizon_line_class: str
    projection_symbolic_trace: str
    stable_hash: str
    version: str = RETRO_CAMERA_PROJECTION_VERSION


@dataclass(frozen=True)
class RetroCameraProjectionLedger:
    """Immutable ledger of camera projection decisions."""

    decisions: Tuple[RetroCameraProjectionDecision, ...]
    decision_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _round_triple(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Round a 3-tuple to canonical precision."""
    return (_round(t[0]), _round(t[1]), _round(t[2]))


# ---------------------------------------------------------------------------
# Helpers — input normalization
# ---------------------------------------------------------------------------


def _normalize_triple(raw: Any, field_name: str) -> Tuple[float, float, float]:
    """Normalize an input to a (float, float, float) tuple.

    Accepts any non-string iterable of length 3 with numeric elements.
    """
    if isinstance(raw, str):
        raise TypeError(
            f"{field_name} must be a non-string iterable, got str"
        )
    try:
        seq = tuple(raw)
    except TypeError:
        raise TypeError(
            f"{field_name} must be iterable, got {type(raw).__name__}"
        )
    if len(seq) != 3:
        raise ValueError(
            f"{field_name} must have exactly 3 elements, got {len(seq)}"
        )
    result = []
    for i, elem in enumerate(seq):
        if isinstance(elem, bool):
            raise TypeError(
                f"{field_name}[{i}] must be int or float, got bool"
            )
        if not isinstance(elem, (int, float)):
            raise TypeError(
                f"{field_name}[{i}] must be int or float, "
                f"got {type(elem).__name__}"
            )
        result.append(float(elem))
    return (result[0], result[1], result[2])


def _validate_positive_float(value: Any, field_name: str) -> float:
    """Validate that a value is a positive float (not bool)."""
    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be numeric, got bool"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    return float(value)


# ---------------------------------------------------------------------------
# Helpers — hashing
# ---------------------------------------------------------------------------


def _compute_camera_state_hash(
    position: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    fov_degrees: float,
    near_plane: float,
    far_plane: float,
    camera_class: str,
) -> str:
    """SHA-256 of canonical JSON of a camera state."""
    payload = {
        "camera_class": camera_class,
        "far_plane": _round(far_plane),
        "fov_degrees": _round(fov_degrees),
        "near_plane": _round(near_plane),
        "position": list(_round_triple(position)),
        "rotation": list(_round_triple(rotation)),
        "version": RETRO_CAMERA_PROJECTION_VERSION,
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_span_hash(
    span_index: int,
    depth_bucket: int,
    projected_width: float,
    projected_height: float,
    visibility_class: str,
) -> str:
    """SHA-256 of canonical JSON of a projection span."""
    payload = {
        "depth_bucket": depth_bucket,
        "projected_height": _round(projected_height),
        "projected_width": _round(projected_width),
        "span_index": span_index,
        "version": RETRO_CAMERA_PROJECTION_VERSION,
        "visibility_class": visibility_class,
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_decision_hash(
    camera_hash: str,
    span_hashes: Tuple[str, ...],
    visible_sector_count: int,
    depth_complexity_class: str,
    projection_mode: str,
    horizon_line_class: str,
    projection_symbolic_trace: str,
) -> str:
    """SHA-256 of canonical JSON of a projection decision."""
    payload = {
        "camera_hash": camera_hash,
        "depth_complexity_class": depth_complexity_class,
        "horizon_line_class": horizon_line_class,
        "projection_mode": projection_mode,
        "projection_symbolic_trace": projection_symbolic_trace,
        "span_hashes": list(span_hashes),
        "version": RETRO_CAMERA_PROJECTION_VERSION,
        "visible_sector_count": visible_sector_count,
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[RetroCameraProjectionDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = tuple(d.stable_hash for d in decisions)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Classification — projection mode
# ---------------------------------------------------------------------------


def classify_projection_mode(
    world_mode: str,
    sector_count: int,
    primitive_count: int,
) -> str:
    """Deterministic projection mode classification.

    Rules:
    - PSEUDO_3D world -> PSEUDO_DOOM_VIEW
    - TRUE_3D + primitives (no mixed sectors) -> PERSPECTIVE_RETRO
    - mixed sectors + primitives -> HYBRID_CAMERA
    - otherwise -> ORTHOGRAPHIC_RETRO
    """
    if world_mode == PSEUDO_3D:
        return PSEUDO_DOOM_VIEW
    if world_mode == TRUE_3D:
        if sector_count > 0 and primitive_count > 0:
            return HYBRID_CAMERA
        if primitive_count > 0:
            return PERSPECTIVE_RETRO
        return ORTHOGRAPHIC_RETRO
    return ORTHOGRAPHIC_RETRO


# ---------------------------------------------------------------------------
# Classification — horizon line
# ---------------------------------------------------------------------------


def classify_horizon_line(
    rotation: Tuple[float, float, float],
) -> str:
    """Deterministic horizon line classification from camera rotation.

    Uses pitch (rotation[0]) to classify horizon:
    - abs(roll) > 5.0 -> TILTED_HORIZON
    - pitch > 15.0 -> LOW_HORIZON (looking up shifts horizon down)
    - pitch < -15.0 -> HIGH_HORIZON (looking down shifts horizon up)
    - otherwise -> CENTER_HORIZON
    """
    pitch = rotation[0]
    roll = rotation[2]
    if abs(roll) > 5.0:
        return TILTED_HORIZON
    if pitch > 15.0:
        return LOW_HORIZON
    if pitch < -15.0:
        return HIGH_HORIZON
    return CENTER_HORIZON


# ---------------------------------------------------------------------------
# Classification — depth complexity
# ---------------------------------------------------------------------------


def classify_depth_complexity(
    visible_sector_count: int,
    span_count: int,
) -> str:
    """Deterministic depth complexity classification.

    total = visible_sector_count + span_count

    Rules:
    - total <= 2 -> MINIMAL_DEPTH
    - total <= 6 -> COMPACT_DEPTH
    - total <= 12 -> STRUCTURED_DEPTH
    - total > 12 -> DENSE_DEPTH
    """
    total = visible_sector_count + span_count
    if total <= 2:
        return MINIMAL_DEPTH
    if total <= 6:
        return COMPACT_DEPTH
    if total <= 12:
        return STRUCTURED_DEPTH
    return DENSE_DEPTH


# ---------------------------------------------------------------------------
# Classification — camera class
# ---------------------------------------------------------------------------


def classify_camera(
    position: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
) -> str:
    """Deterministic camera class classification.

    Rules:
    - rotation is all zeros -> FIXED_CAMERA
    - abs(rotation[1]) > 45.0 -> ORBITAL_CAMERA (large yaw)
    - otherwise -> TRACKING_CAMERA
    """
    if rotation[0] == 0.0 and rotation[1] == 0.0 and rotation[2] == 0.0:
        return FIXED_CAMERA
    if abs(rotation[1]) > 45.0:
        return ORBITAL_CAMERA
    return TRACKING_CAMERA


# ---------------------------------------------------------------------------
# Visibility — sector count derivation
# ---------------------------------------------------------------------------


def compute_visible_sector_count(
    sector_count: int,
    fov_degrees: float,
    near_plane: float,
    far_plane: float,
) -> int:
    """Derive visible sector count deterministically.

    Simple bounded rules:
    - base = sector_count
    - fov_factor = min(fov_degrees / 90.0, 1.0)
    - depth_range = far_plane - near_plane
    - if depth_range < 10.0: effective = sector_count * 0.5
    - visible = effective * fov_factor
    - clamp to [1, sector_count] (0 only if sector_count == 0)
    """
    if sector_count == 0:
        return 0
    fov_factor = min(fov_degrees / 90.0, 1.0)
    depth_range = far_plane - near_plane
    if depth_range < 10.0:
        effective = sector_count * 0.5
    else:
        effective = float(sector_count)
    visible = effective * fov_factor
    result = max(1, min(sector_count, int(math.ceil(visible))))
    return result


# ---------------------------------------------------------------------------
# Projection spans — Doom-style column abstraction
# ---------------------------------------------------------------------------


def _classify_span_visibility(
    depth_bucket: int,
    total_buckets: int,
) -> str:
    """Classify span visibility from depth bucket position.

    Rules:
    - first quarter -> VISIBLE
    - second quarter -> PARTIAL
    - third quarter -> DISTANT
    - last quarter -> OCCLUDED
    """
    if total_buckets <= 0:
        return VISIBLE
    fraction = depth_bucket / total_buckets
    if fraction < 0.25:
        return VISIBLE
    if fraction < 0.5:
        return PARTIAL
    if fraction < 0.75:
        return DISTANT
    return OCCLUDED


def build_projection_spans(
    visible_sector_count: int,
    fov_degrees: float,
    near_plane: float,
    far_plane: float,
) -> Tuple[RetroProjectionSpan, ...]:
    """Build deterministic projection spans.

    Creates one span per visible sector, distributing depth across buckets.
    Each span has projected width/height derived from FOV and depth.
    """
    if visible_sector_count == 0:
        return ()

    depth_range = far_plane - near_plane
    spans = []
    for i in range(visible_sector_count):
        depth_bucket = i
        depth_fraction = (i + 1) / visible_sector_count
        depth = near_plane + depth_fraction * depth_range

        # Projected width narrows with depth (perspective)
        projected_width = _round(fov_degrees / max(depth, 0.001))
        # Projected height proportional to width
        projected_height = _round(projected_width * 0.75)

        visibility_class = _classify_span_visibility(
            i, visible_sector_count,
        )

        h = _compute_span_hash(
            i, depth_bucket, projected_width, projected_height,
            visibility_class,
        )

        spans.append(RetroProjectionSpan(
            span_index=i,
            depth_bucket=depth_bucket,
            projected_width=projected_width,
            projected_height=projected_height,
            visibility_class=visibility_class,
            stable_hash=h,
        ))

    return tuple(spans)


# ---------------------------------------------------------------------------
# Symbolic trace
# ---------------------------------------------------------------------------


def build_projection_symbolic_trace(
    projection_mode: str,
    span_count: int,
    horizon_line_class: str,
    depth_complexity_class: str,
) -> str:
    """Build deterministic projection symbolic trace.

    Format:
      PROJECTION_MODE -> N spans -> HORIZON_CLASS -> DEPTH_CLASS
    """
    return (
        f"{projection_mode} -> {span_count} spans -> "
        f"{horizon_line_class} -> {depth_complexity_class}"
    )


# ---------------------------------------------------------------------------
# Core — build_retro_camera_projection
# ---------------------------------------------------------------------------


def build_retro_camera_projection(
    world_model: Any,
    position: Any,
    rotation: Any,
    fov_degrees: Any,
    near_plane: Any,
    far_plane: Any,
) -> RetroCameraProjectionDecision:
    """Build a deterministic retro camera projection decision.

    Parameters
    ----------
    world_model : RetroWorldModel
        The world model to project.
    position : 3-element iterable
        Camera position as (x, y, z).
    rotation : 3-element iterable
        Camera rotation as (pitch, yaw, roll).
    fov_degrees : float
        Field of view in degrees.
    near_plane : float
        Near clipping plane distance.
    far_plane : float
        Far clipping plane distance.
    """
    # Validate world model
    if not isinstance(world_model, RetroWorldModel):
        raise TypeError(
            f"world_model must be RetroWorldModel, "
            f"got {type(world_model).__name__}"
        )

    # Normalize position and rotation
    pos = _normalize_triple(position, "position")
    rot = _normalize_triple(rotation, "rotation")

    # Validate FOV
    fov = _validate_positive_float(fov_degrees, "fov_degrees")
    if fov <= 0.0 or fov > 180.0:
        raise ValueError(
            f"fov_degrees must be in (0, 180], got {fov}"
        )

    # Validate near/far planes
    near = _validate_positive_float(near_plane, "near_plane")
    far = _validate_positive_float(far_plane, "far_plane")
    if near < 0.0:
        raise ValueError(f"near_plane must be >= 0, got {near}")
    if far < 0.0:
        raise ValueError(f"far_plane must be >= 0, got {far}")
    if far <= near:
        raise ValueError(
            f"far_plane ({far}) must be > near_plane ({near})"
        )

    # Classify camera
    camera_class = classify_camera(pos, rot)

    # Build camera state
    camera_hash = _compute_camera_state_hash(
        pos, rot, fov, near, far, camera_class,
    )
    camera_state = RetroCameraState(
        position=pos,
        rotation=rot,
        fov_degrees=fov,
        near_plane=near,
        far_plane=far,
        camera_class=camera_class,
        stable_hash=camera_hash,
    )

    # Classify projection mode from world model
    projection_mode = classify_projection_mode(
        world_model.world_mode,
        world_model.sector_count,
        world_model.primitive_count,
    )

    # Compute visible sectors
    visible_sector_count = compute_visible_sector_count(
        world_model.sector_count, fov, near, far,
    )

    # Build projection spans
    projection_spans = build_projection_spans(
        visible_sector_count, fov, near, far,
    )

    # Classify horizon
    horizon_line_class = classify_horizon_line(rot)

    # Classify depth complexity
    depth_complexity_class = classify_depth_complexity(
        visible_sector_count, len(projection_spans),
    )

    # Build symbolic trace
    projection_symbolic_trace = build_projection_symbolic_trace(
        projection_mode, len(projection_spans),
        horizon_line_class, depth_complexity_class,
    )

    # Compute decision hash
    span_hashes = tuple(s.stable_hash for s in projection_spans)
    decision_hash = _compute_decision_hash(
        camera_hash, span_hashes, visible_sector_count,
        depth_complexity_class, projection_mode,
        horizon_line_class, projection_symbolic_trace,
    )

    return RetroCameraProjectionDecision(
        camera_state=camera_state,
        projection_spans=projection_spans,
        visible_sector_count=visible_sector_count,
        depth_complexity_class=depth_complexity_class,
        projection_mode=projection_mode,
        horizon_line_class=horizon_line_class,
        projection_symbolic_trace=projection_symbolic_trace,
        stable_hash=decision_hash,
    )


# ---------------------------------------------------------------------------
# Core — build_retro_camera_projection_ledger
# ---------------------------------------------------------------------------


def build_retro_camera_projection_ledger(
    decisions: Any,
) -> RetroCameraProjectionLedger:
    """Build an immutable ledger of camera projection decisions.

    Parameters
    ----------
    decisions : iterable of RetroCameraProjectionDecision
        Projection decisions to include in the ledger.
    """
    if isinstance(decisions, str):
        raise TypeError("decisions must be a non-string iterable")
    try:
        decs = tuple(decisions)
    except TypeError:
        raise TypeError(
            f"decisions must be iterable, "
            f"got {type(decisions).__name__}"
        )
    for i, d in enumerate(decs):
        if not isinstance(d, RetroCameraProjectionDecision):
            raise TypeError(
                f"decisions[{i}] must be RetroCameraProjectionDecision, "
                f"got {type(d).__name__}"
            )

    h = _compute_ledger_hash(decs)

    return RetroCameraProjectionLedger(
        decisions=decs,
        decision_count=len(decs),
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core — export_retro_camera_projection_bundle
# ---------------------------------------------------------------------------


def export_retro_camera_projection_bundle(
    decision: RetroCameraProjectionDecision,
) -> Dict[str, Any]:
    """Export a projection decision as a replay-safe canonical bundle."""
    if not isinstance(decision, RetroCameraProjectionDecision):
        raise TypeError(
            f"decision must be RetroCameraProjectionDecision, "
            f"got {type(decision).__name__}"
        )
    return {
        "camera_state": {
            "camera_class": decision.camera_state.camera_class,
            "far_plane": _round(decision.camera_state.far_plane),
            "fov_degrees": _round(decision.camera_state.fov_degrees),
            "near_plane": _round(decision.camera_state.near_plane),
            "position": list(_round_triple(decision.camera_state.position)),
            "rotation": list(_round_triple(decision.camera_state.rotation)),
            "stable_hash": decision.camera_state.stable_hash,
            "version": decision.camera_state.version,
        },
        "depth_complexity_class": decision.depth_complexity_class,
        "horizon_line_class": decision.horizon_line_class,
        "projection_mode": decision.projection_mode,
        "projection_spans": [
            {
                "depth_bucket": s.depth_bucket,
                "projected_height": _round(s.projected_height),
                "projected_width": _round(s.projected_width),
                "span_index": s.span_index,
                "stable_hash": s.stable_hash,
                "version": s.version,
                "visibility_class": s.visibility_class,
            }
            for s in decision.projection_spans
        ],
        "projection_symbolic_trace": decision.projection_symbolic_trace,
        "stable_hash": decision.stable_hash,
        "version": decision.version,
        "visible_sector_count": decision.visible_sector_count,
    }


# ---------------------------------------------------------------------------
# Core — export_retro_camera_projection_ledger
# ---------------------------------------------------------------------------


def export_retro_camera_projection_ledger(
    ledger: RetroCameraProjectionLedger,
) -> Dict[str, Any]:
    """Export a projection ledger as a replay-safe canonical bundle."""
    if not isinstance(ledger, RetroCameraProjectionLedger):
        raise TypeError(
            f"ledger must be RetroCameraProjectionLedger, "
            f"got {type(ledger).__name__}"
        )
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_retro_camera_projection_bundle(d)
            for d in ledger.decisions
        ],
        "stable_hash": ledger.stable_hash,
    }
