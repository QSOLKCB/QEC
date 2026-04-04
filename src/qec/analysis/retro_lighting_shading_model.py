"""v137.0.12 — Retro Lighting + Shading Model.

Deterministic lighting and shading pipeline:

  world
  -> camera
  -> projection
  -> light zones
  -> shading lanes
  -> luminance field
  -> stable ledger

Inspired by:
  3dfx Voodoo antialiasing / FSAA
  Matrox Parhelia triple-head output
  SGI InfiniteReality shading pipeline

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

RETRO_LIGHTING_SHADING_VERSION: str = "v137.0.12"

# ---------------------------------------------------------------------------
# Constants — lighting modes
# ---------------------------------------------------------------------------

AMBIENT_RETRO: str = "AMBIENT_RETRO"
SECTOR_BANDED: str = "SECTOR_BANDED"
DEPTH_SHADED: str = "DEPTH_SHADED"
HYBRID_LUMINANCE: str = "HYBRID_LUMINANCE"

VALID_LIGHTING_MODES: Tuple[str, ...] = (
    AMBIENT_RETRO,
    SECTOR_BANDED,
    DEPTH_SHADED,
    HYBRID_LUMINANCE,
)

# ---------------------------------------------------------------------------
# Constants — light classes
# ---------------------------------------------------------------------------

BRIGHT: str = "BRIGHT"
MID: str = "MID"
DIM: str = "DIM"
DARK: str = "DARK"

# ---------------------------------------------------------------------------
# Constants — shadow classes
# ---------------------------------------------------------------------------

NO_SHADOW: str = "NO_SHADOW"
SOFT_SHADOW: str = "SOFT_SHADOW"
HARD_SHADOW: str = "HARD_SHADOW"
FULL_SHADOW: str = "FULL_SHADOW"

# ---------------------------------------------------------------------------
# Constants — blend modes
# ---------------------------------------------------------------------------

QUAD_BLEND: str = "QUAD_BLEND"
VOODOO_FSAA: str = "VOODOO_FSAA"

VALID_BLEND_MODES: Tuple[str, ...] = (
    QUAD_BLEND,
    VOODOO_FSAA,
)

# ---------------------------------------------------------------------------
# Constants — antialias band classes
# ---------------------------------------------------------------------------

AA_NONE: str = "AA_NONE"
AA_EDGE: str = "AA_EDGE"
AA_FULL: str = "AA_FULL"
AA_SUPER: str = "AA_SUPER"

# ---------------------------------------------------------------------------
# Constants — viewport classes (Parhelia triple-head)
# ---------------------------------------------------------------------------

LEFT_HEAD: str = "LEFT_HEAD"
CENTER_HEAD: str = "CENTER_HEAD"
RIGHT_HEAD: str = "RIGHT_HEAD"

VALID_VIEWPORT_CLASSES: Tuple[str, ...] = (
    LEFT_HEAD,
    CENTER_HEAD,
    RIGHT_HEAD,
)

# ---------------------------------------------------------------------------
# Constants — viewport mode
# ---------------------------------------------------------------------------

TRIPLE_HEAD_RETRO: str = "TRIPLE_HEAD_RETRO"

# ---------------------------------------------------------------------------
# Constants — gigacolor classes
# ---------------------------------------------------------------------------

STANDARD_RETRO: str = "STANDARD_RETRO"
GIGACOLOR_RETRO: str = "GIGACOLOR_RETRO"

# ---------------------------------------------------------------------------
# Constants — defaults
# ---------------------------------------------------------------------------

DEFAULT_LANE_COUNT: int = 4

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroLightZone:
    """Immutable light zone in the retro shading pipeline."""

    zone_index: int
    luminance_level: float
    light_class: str
    shadow_class: str
    stable_hash: str
    version: str = RETRO_LIGHTING_SHADING_VERSION


@dataclass(frozen=True)
class RetroShadingLane:
    """Immutable shading lane — Voodoo-style blend abstraction."""

    lane_index: int
    blend_factor: float
    antialias_band_class: str
    stable_hash: str
    version: str = RETRO_LIGHTING_SHADING_VERSION


@dataclass(frozen=True)
class RetroLuminanceViewport:
    """Immutable luminance viewport — Parhelia triple-head abstraction."""

    viewport_index: int
    viewport_class: str
    luminance_mean: float
    stable_hash: str
    version: str = RETRO_LIGHTING_SHADING_VERSION


@dataclass(frozen=True)
class RetroShadingDecision:
    """Immutable shading decision artifact."""

    light_zones: Tuple[RetroLightZone, ...]
    shading_lanes: Tuple[RetroShadingLane, ...]
    viewports: Tuple[RetroLuminanceViewport, ...]
    contrast_ratio: float
    lighting_mode: str
    blend_mode: str
    gigacolor_class: str
    luminance_symbolic_trace: str
    stable_hash: str
    version: str = RETRO_LIGHTING_SHADING_VERSION


@dataclass(frozen=True)
class RetroLightingLedger:
    """Immutable ledger of lighting/shading decisions."""

    decisions: Tuple[RetroShadingDecision, ...]
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


# ---------------------------------------------------------------------------
# Helpers — hashing
# ---------------------------------------------------------------------------


def _compute_zone_hash(
    zone_index: int,
    luminance_level: float,
    light_class: str,
    shadow_class: str,
) -> str:
    """SHA-256 of canonical JSON of a light zone."""
    payload = {
        "light_class": light_class,
        "luminance_level": _round(luminance_level),
        "shadow_class": shadow_class,
        "version": RETRO_LIGHTING_SHADING_VERSION,
        "zone_index": zone_index,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_lane_hash(
    lane_index: int,
    blend_factor: float,
    antialias_band_class: str,
) -> str:
    """SHA-256 of canonical JSON of a shading lane."""
    payload = {
        "antialias_band_class": antialias_band_class,
        "blend_factor": _round(blend_factor),
        "lane_index": lane_index,
        "version": RETRO_LIGHTING_SHADING_VERSION,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_viewport_hash(
    viewport_index: int,
    viewport_class: str,
    luminance_mean: float,
) -> str:
    """SHA-256 of canonical JSON of a luminance viewport."""
    payload = {
        "luminance_mean": _round(luminance_mean),
        "version": RETRO_LIGHTING_SHADING_VERSION,
        "viewport_class": viewport_class,
        "viewport_index": viewport_index,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_decision_hash(
    zone_hashes: Tuple[str, ...],
    lane_hashes: Tuple[str, ...],
    viewport_hashes: Tuple[str, ...],
    contrast_ratio: float,
    lighting_mode: str,
    blend_mode: str,
    gigacolor_class: str,
    luminance_symbolic_trace: str,
) -> str:
    """SHA-256 of canonical JSON of a shading decision."""
    payload = {
        "blend_mode": blend_mode,
        "contrast_ratio": _round(contrast_ratio),
        "gigacolor_class": gigacolor_class,
        "lane_hashes": list(lane_hashes),
        "lighting_mode": lighting_mode,
        "luminance_symbolic_trace": luminance_symbolic_trace,
        "version": RETRO_LIGHTING_SHADING_VERSION,
        "viewport_hashes": list(viewport_hashes),
        "zone_hashes": list(zone_hashes),
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[RetroShadingDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = [d.stable_hash for d in decisions]
    return hashlib.sha256(
        _canonical_json(hashes).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Classification — light class
# ---------------------------------------------------------------------------


def classify_light(luminance: float) -> str:
    """Deterministic light class from luminance level [0, 1].

    Rules:
    - luminance >= 0.75 -> BRIGHT
    - luminance >= 0.5  -> MID
    - luminance >= 0.25 -> DIM
    - otherwise         -> DARK
    """
    if luminance >= 0.75:
        return BRIGHT
    if luminance >= 0.5:
        return MID
    if luminance >= 0.25:
        return DIM
    return DARK


# ---------------------------------------------------------------------------
# Classification — shadow class
# ---------------------------------------------------------------------------


def classify_shadow(luminance: float) -> str:
    """Deterministic shadow class from luminance level [0, 1].

    Rules:
    - luminance >= 0.75 -> NO_SHADOW
    - luminance >= 0.5  -> SOFT_SHADOW
    - luminance >= 0.25 -> HARD_SHADOW
    - otherwise         -> FULL_SHADOW
    """
    if luminance >= 0.75:
        return NO_SHADOW
    if luminance >= 0.5:
        return SOFT_SHADOW
    if luminance >= 0.25:
        return HARD_SHADOW
    return FULL_SHADOW


# ---------------------------------------------------------------------------
# Classification — lighting mode
# ---------------------------------------------------------------------------


def classify_lighting_mode(
    zone_count: int,
    has_depth: bool,
) -> str:
    """Deterministic lighting mode classification.

    Rules:
    - zone_count == 1 and no depth -> AMBIENT_RETRO
    - zone_count > 1 and no depth  -> SECTOR_BANDED
    - zone_count == 1 and depth    -> DEPTH_SHADED
    - zone_count > 1 and depth     -> HYBRID_LUMINANCE
    """
    if zone_count <= 1 and not has_depth:
        return AMBIENT_RETRO
    if zone_count > 1 and not has_depth:
        return SECTOR_BANDED
    if zone_count <= 1 and has_depth:
        return DEPTH_SHADED
    return HYBRID_LUMINANCE


# ---------------------------------------------------------------------------
# Classification — antialias band
# ---------------------------------------------------------------------------


def classify_antialias_band(lane_index: int, lane_count: int) -> str:
    """Deterministic antialias band class for a lane.

    Rules:
    - first lane  -> AA_NONE
    - last lane   -> AA_SUPER
    - second lane -> AA_EDGE
    - otherwise   -> AA_FULL
    """
    if lane_count <= 1:
        return AA_NONE
    if lane_index == 0:
        return AA_NONE
    if lane_index == lane_count - 1:
        return AA_SUPER
    if lane_index == 1:
        return AA_EDGE
    return AA_FULL


# ---------------------------------------------------------------------------
# Classification — gigacolor
# ---------------------------------------------------------------------------


def classify_gigacolor(viewport_count: int) -> str:
    """Deterministic gigacolor class.

    Rules:
    - 3 or more viewports -> GIGACOLOR_RETRO
    - otherwise           -> STANDARD_RETRO
    """
    if viewport_count >= 3:
        return GIGACOLOR_RETRO
    return STANDARD_RETRO


# ---------------------------------------------------------------------------
# Helpers — input validation
# ---------------------------------------------------------------------------


def _validate_luminance_levels(levels: Any) -> Tuple[float, ...]:
    """Validate and normalize luminance levels.

    Accepts a non-string iterable of numeric values in [0, 1].
    """
    if isinstance(levels, str):
        raise TypeError("luminance_levels must be a non-string iterable")
    try:
        seq = tuple(levels)
    except TypeError:
        raise TypeError(
            f"luminance_levels must be iterable, "
            f"got {type(levels).__name__}"
        )
    if len(seq) == 0:
        raise ValueError("luminance_levels must not be empty")
    result = []
    for i, v in enumerate(seq):
        if isinstance(v, bool):
            raise TypeError(
                f"luminance_levels[{i}] must be numeric, got bool"
            )
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"luminance_levels[{i}] must be numeric, "
                f"got {type(v).__name__}"
            )
        fv = float(v)
        if fv < 0.0 or fv > 1.0:
            raise ValueError(
                f"luminance_levels[{i}] must be in [0, 1], got {fv}"
            )
        result.append(fv)
    return tuple(result)


def _validate_bool(value: Any, field_name: str) -> bool:
    """Validate a strict boolean."""
    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be bool, got {type(value).__name__}"
        )
    return value


def _validate_positive_int(value: Any, field_name: str) -> int:
    """Validate a positive integer (not bool)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


# ---------------------------------------------------------------------------
# Builders — light zones
# ---------------------------------------------------------------------------


def build_light_zones(
    luminance_levels: Tuple[float, ...],
) -> Tuple[RetroLightZone, ...]:
    """Build deterministic light zones from luminance levels."""
    zones = []
    for i, lum in enumerate(luminance_levels):
        lc = classify_light(lum)
        sc = classify_shadow(lum)
        h = _compute_zone_hash(i, lum, lc, sc)
        zones.append(RetroLightZone(
            zone_index=i,
            luminance_level=_round(lum),
            light_class=lc,
            shadow_class=sc,
            stable_hash=h,
        ))
    return tuple(zones)


# ---------------------------------------------------------------------------
# Builders — shading lanes
# ---------------------------------------------------------------------------


def build_shading_lanes(
    lane_count: int,
    smoothing: bool,
) -> Tuple[RetroShadingLane, ...]:
    """Build deterministic shading lanes.

    Default 4 lanes. Blend factor distributed evenly across lanes.
    """
    lanes = []
    for i in range(lane_count):
        blend_factor = _round((i + 1) / lane_count)
        aa_class = classify_antialias_band(i, lane_count)
        h = _compute_lane_hash(i, blend_factor, aa_class)
        lanes.append(RetroShadingLane(
            lane_index=i,
            blend_factor=blend_factor,
            antialias_band_class=aa_class,
            stable_hash=h,
        ))
    return tuple(lanes)


# ---------------------------------------------------------------------------
# Builders — luminance viewports
# ---------------------------------------------------------------------------


def build_luminance_viewports(
    luminance_levels: Tuple[float, ...],
) -> Tuple[RetroLuminanceViewport, ...]:
    """Build Parhelia triple-head luminance viewports.

    Always produces 3 viewports (LEFT_HEAD, CENTER_HEAD, RIGHT_HEAD).
    Luminance is split across viewports deterministically.
    """
    n = len(luminance_levels)
    third = max(1, n // 3)

    left_levels = luminance_levels[:third]
    center_levels = luminance_levels[third:2 * third]
    right_levels = luminance_levels[2 * third:]

    def _mean(vals: Tuple[float, ...]) -> float:
        if len(vals) == 0:
            return 0.0
        return _round(sum(vals) / len(vals))

    viewport_specs = (
        (0, LEFT_HEAD, _mean(left_levels)),
        (1, CENTER_HEAD, _mean(center_levels)),
        (2, RIGHT_HEAD, _mean(right_levels)),
    )

    viewports = []
    for idx, vc, lm in viewport_specs:
        h = _compute_viewport_hash(idx, vc, lm)
        viewports.append(RetroLuminanceViewport(
            viewport_index=idx,
            viewport_class=vc,
            luminance_mean=lm,
            stable_hash=h,
        ))
    return tuple(viewports)


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------


def compute_contrast_ratio(
    luminance_levels: Tuple[float, ...],
) -> float:
    """Compute contrast ratio: max_luma - min_luma, clamped [0, 1]."""
    if len(luminance_levels) == 0:
        return 0.0
    max_l = max(luminance_levels)
    min_l = min(luminance_levels)
    contrast = max_l - min_l
    return _round(max(0.0, min(1.0, contrast)))


# ---------------------------------------------------------------------------
# Symbolic trace
# ---------------------------------------------------------------------------


def build_luminance_symbolic_trace(
    blend_mode: str,
    gigacolor_class: str,
) -> str:
    """Build deterministic luminance symbolic trace.

    Format:
      BLEND_MODE -> TRIPLE_HEAD_RETRO -> GIGACOLOR_CLASS
    """
    return f"{blend_mode} -> {TRIPLE_HEAD_RETRO} -> {gigacolor_class}"


# ---------------------------------------------------------------------------
# Core — build_retro_shading_decision
# ---------------------------------------------------------------------------


def build_retro_shading_decision(
    luminance_levels: Any,
    smoothing: Any,
    has_depth: Any,
    lane_count: Any = DEFAULT_LANE_COUNT,
) -> RetroShadingDecision:
    """Build a deterministic retro shading decision.

    Parameters
    ----------
    luminance_levels : iterable of float
        Luminance values in [0, 1] for each light zone.
    smoothing : bool
        Whether antialiasing smoothing is active.
    has_depth : bool
        Whether depth shading is present.
    lane_count : int, optional
        Number of shading lanes (default 4).
    """
    # Validate inputs
    levels = _validate_luminance_levels(luminance_levels)
    smooth = _validate_bool(smoothing, "smoothing")
    depth = _validate_bool(has_depth, "has_depth")
    lc = _validate_positive_int(lane_count, "lane_count")

    # Build components
    light_zones = build_light_zones(levels)
    shading_lanes = build_shading_lanes(lc, smooth)
    viewports = build_luminance_viewports(levels)

    # Classifications
    contrast_ratio = compute_contrast_ratio(levels)
    lighting_mode = classify_lighting_mode(len(levels), depth)
    blend_mode = VOODOO_FSAA if smooth else QUAD_BLEND
    gigacolor_class = classify_gigacolor(len(viewports))

    # Symbolic trace
    luminance_symbolic_trace = build_luminance_symbolic_trace(
        blend_mode, gigacolor_class,
    )

    # Hashing
    zone_hashes = tuple(z.stable_hash for z in light_zones)
    lane_hashes = tuple(l.stable_hash for l in shading_lanes)
    viewport_hashes = tuple(v.stable_hash for v in viewports)

    decision_hash = _compute_decision_hash(
        zone_hashes, lane_hashes, viewport_hashes,
        contrast_ratio, lighting_mode, blend_mode,
        gigacolor_class, luminance_symbolic_trace,
    )

    return RetroShadingDecision(
        light_zones=light_zones,
        shading_lanes=shading_lanes,
        viewports=viewports,
        contrast_ratio=contrast_ratio,
        lighting_mode=lighting_mode,
        blend_mode=blend_mode,
        gigacolor_class=gigacolor_class,
        luminance_symbolic_trace=luminance_symbolic_trace,
        stable_hash=decision_hash,
    )


# ---------------------------------------------------------------------------
# Core — build_retro_lighting_ledger
# ---------------------------------------------------------------------------


def build_retro_lighting_ledger(
    decisions: Any,
) -> RetroLightingLedger:
    """Build an immutable ledger of shading decisions.

    Parameters
    ----------
    decisions : iterable of RetroShadingDecision
        Shading decisions to include in the ledger.
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
        if not isinstance(d, RetroShadingDecision):
            raise TypeError(
                f"decisions[{i}] must be RetroShadingDecision, "
                f"got {type(d).__name__}"
            )

    h = _compute_ledger_hash(decs)

    return RetroLightingLedger(
        decisions=decs,
        decision_count=len(decs),
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core — export_retro_shading_bundle
# ---------------------------------------------------------------------------


def export_retro_shading_bundle(
    decision: RetroShadingDecision,
) -> Dict[str, Any]:
    """Export a shading decision as a replay-safe canonical bundle."""
    if not isinstance(decision, RetroShadingDecision):
        raise TypeError(
            f"decision must be RetroShadingDecision, "
            f"got {type(decision).__name__}"
        )
    return {
        "blend_mode": decision.blend_mode,
        "contrast_ratio": _round(decision.contrast_ratio),
        "gigacolor_class": decision.gigacolor_class,
        "light_zones": [
            {
                "light_class": z.light_class,
                "luminance_level": _round(z.luminance_level),
                "shadow_class": z.shadow_class,
                "stable_hash": z.stable_hash,
                "version": z.version,
                "zone_index": z.zone_index,
            }
            for z in decision.light_zones
        ],
        "lighting_mode": decision.lighting_mode,
        "luminance_symbolic_trace": decision.luminance_symbolic_trace,
        "shading_lanes": [
            {
                "antialias_band_class": l.antialias_band_class,
                "blend_factor": _round(l.blend_factor),
                "lane_index": l.lane_index,
                "stable_hash": l.stable_hash,
                "version": l.version,
            }
            for l in decision.shading_lanes
        ],
        "stable_hash": decision.stable_hash,
        "version": decision.version,
        "viewports": [
            {
                "luminance_mean": _round(v.luminance_mean),
                "stable_hash": v.stable_hash,
                "version": v.version,
                "viewport_class": v.viewport_class,
                "viewport_index": v.viewport_index,
            }
            for v in decision.viewports
        ],
    }


# ---------------------------------------------------------------------------
# Core — export_retro_lighting_ledger
# ---------------------------------------------------------------------------


def export_retro_lighting_ledger(
    ledger: RetroLightingLedger,
) -> Dict[str, Any]:
    """Export a lighting ledger as a replay-safe canonical bundle."""
    if not isinstance(ledger, RetroLightingLedger):
        raise TypeError(
            f"ledger must be RetroLightingLedger, "
            f"got {type(ledger).__name__}"
        )
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_retro_shading_bundle(d)
            for d in ledger.decisions
        ],
        "stable_hash": ledger.stable_hash,
    }
