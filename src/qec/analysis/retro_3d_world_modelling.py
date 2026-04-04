"""v137.0.10 — Retro 3D World Modelling.

Deterministic retro-style world representation substrate:

  symbolic supervisory state
  -> retro world primitives
  -> deterministic scene composition
  -> bounded projection model
  -> replay-safe world artifact

Supports two world modes:

  TRUE_3D   — geometry-first scene representation (SGI IRIS style)
  PSEUDO_3D — Doom-like constrained projection model

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

RETRO_3D_WORLD_MODELLING_VERSION: str = "v137.0.10"

# ---------------------------------------------------------------------------
# Constants — world modes
# ---------------------------------------------------------------------------

TRUE_3D: str = "TRUE_3D"
PSEUDO_3D: str = "PSEUDO_3D"

VALID_WORLD_MODES: Tuple[str, ...] = (TRUE_3D, PSEUDO_3D)

# ---------------------------------------------------------------------------
# Constants — sector classes
# ---------------------------------------------------------------------------

OPEN_SECTOR: str = "OPEN_SECTOR"
CORRIDOR_SECTOR: str = "CORRIDOR_SECTOR"
CHAMBER_SECTOR: str = "CHAMBER_SECTOR"
VERTICAL_SECTOR: str = "VERTICAL_SECTOR"
DENSE_SECTOR: str = "DENSE_SECTOR"

# ---------------------------------------------------------------------------
# Constants — projection classes
# ---------------------------------------------------------------------------

ORTHO_LIKE: str = "ORTHO_LIKE"
PERSPECTIVE_3D: str = "PERSPECTIVE_3D"
PSEUDO_DOOM: str = "PSEUDO_DOOM"
HYBRID_RETRO: str = "HYBRID_RETRO"

# ---------------------------------------------------------------------------
# Constants — complexity classes
# ---------------------------------------------------------------------------

MINIMAL: str = "MINIMAL"
COMPACT: str = "COMPACT"
STRUCTURED: str = "STRUCTURED"
DENSE: str = "DENSE"

# ---------------------------------------------------------------------------
# Constants — primitive types
# ---------------------------------------------------------------------------

PRIMITIVE_WALL: str = "WALL"
PRIMITIVE_ENTITY: str = "ENTITY"
PRIMITIVE_VERTEX: str = "VERTEX"
PRIMITIVE_EDGE: str = "EDGE"
PRIMITIVE_VOLUME: str = "VOLUME"

VALID_PRIMITIVE_TYPES: Tuple[str, ...] = (
    PRIMITIVE_WALL,
    PRIMITIVE_ENTITY,
    PRIMITIVE_VERTEX,
    PRIMITIVE_EDGE,
    PRIMITIVE_VOLUME,
)

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroWorldPrimitive:
    """Immutable retro world geometry primitive."""

    primitive_id: str
    primitive_type: str
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    symbolic_class: str
    stable_hash: str
    version: str = RETRO_3D_WORLD_MODELLING_VERSION


@dataclass(frozen=True)
class RetroWorldSector:
    """Immutable retro world sector."""

    sector_id: str
    floor_height: float
    ceiling_height: float
    wall_count: int
    entity_count: int
    sector_class: str
    stable_hash: str
    version: str = RETRO_3D_WORLD_MODELLING_VERSION


@dataclass(frozen=True)
class RetroWorldModel:
    """Immutable retro world model composition."""

    world_mode: str
    primitive_count: int
    sector_count: int
    entity_count: int
    camera_pose: Tuple[float, float, float]
    projection_class: str
    complexity_class: str
    world_symbolic_trace: str
    stable_hash: str
    version: str = RETRO_3D_WORLD_MODELLING_VERSION


@dataclass(frozen=True)
class RetroWorldLedger:
    """Immutable ledger of retro world models."""

    models: Tuple[RetroWorldModel, ...]
    model_count: int
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


def _validate_non_negative_int(value: Any, field_name: str) -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def _validate_string(value: Any, field_name: str) -> str:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


# ---------------------------------------------------------------------------
# Helpers — primitive hashing
# ---------------------------------------------------------------------------


def _primitive_to_canonical_dict(
    primitive_id: str,
    primitive_type: str,
    position: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    symbolic_class: str,
) -> Dict[str, Any]:
    """Convert primitive fields to a canonical dict for hashing."""
    return {
        "position": list(_round_triple(position)),
        "primitive_id": primitive_id,
        "primitive_type": primitive_type,
        "rotation": list(_round_triple(rotation)),
        "scale": list(_round_triple(scale)),
        "symbolic_class": symbolic_class,
        "version": RETRO_3D_WORLD_MODELLING_VERSION,
    }


def _compute_primitive_hash(
    primitive_id: str,
    primitive_type: str,
    position: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    symbolic_class: str,
) -> str:
    """SHA-256 of canonical JSON of a primitive."""
    payload = _primitive_to_canonical_dict(
        primitive_id, primitive_type, position, scale, rotation,
        symbolic_class,
    )
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers — sector hashing
# ---------------------------------------------------------------------------


def _sector_to_canonical_dict(
    sector_id: str,
    floor_height: float,
    ceiling_height: float,
    wall_count: int,
    entity_count: int,
    sector_class: str,
) -> Dict[str, Any]:
    """Convert sector fields to a canonical dict for hashing."""
    return {
        "ceiling_height": _round(ceiling_height),
        "entity_count": entity_count,
        "floor_height": _round(floor_height),
        "sector_class": sector_class,
        "sector_id": sector_id,
        "version": RETRO_3D_WORLD_MODELLING_VERSION,
        "wall_count": wall_count,
    }


def _compute_sector_hash(
    sector_id: str,
    floor_height: float,
    ceiling_height: float,
    wall_count: int,
    entity_count: int,
    sector_class: str,
) -> str:
    """SHA-256 of canonical JSON of a sector."""
    payload = _sector_to_canonical_dict(
        sector_id, floor_height, ceiling_height, wall_count, entity_count,
        sector_class,
    )
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers — world model hashing
# ---------------------------------------------------------------------------


def _model_to_canonical_dict(
    world_mode: str,
    primitive_count: int,
    sector_count: int,
    entity_count: int,
    camera_pose: Tuple[float, float, float],
    projection_class: str,
    complexity_class: str,
    world_symbolic_trace: str,
) -> Dict[str, Any]:
    """Convert world model fields to a canonical dict for hashing."""
    return {
        "camera_pose": list(_round_triple(camera_pose)),
        "complexity_class": complexity_class,
        "entity_count": entity_count,
        "primitive_count": primitive_count,
        "projection_class": projection_class,
        "sector_count": sector_count,
        "version": RETRO_3D_WORLD_MODELLING_VERSION,
        "world_mode": world_mode,
        "world_symbolic_trace": world_symbolic_trace,
    }


def _compute_model_hash(
    world_mode: str,
    primitive_count: int,
    sector_count: int,
    entity_count: int,
    camera_pose: Tuple[float, float, float],
    projection_class: str,
    complexity_class: str,
    world_symbolic_trace: str,
) -> str:
    """SHA-256 of canonical JSON of a world model."""
    payload = _model_to_canonical_dict(
        world_mode, primitive_count, sector_count, entity_count,
        camera_pose, projection_class, complexity_class,
        world_symbolic_trace,
    )
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    models: Tuple[RetroWorldModel, ...],
) -> str:
    """SHA-256 of ordered model hashes."""
    hashes = tuple(m.stable_hash for m in models)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------


def classify_sector(
    floor_height: float,
    ceiling_height: float,
    wall_count: int,
    entity_count: int,
) -> str:
    """Deterministic sector classification.

    Rules applied in priority order:
    1. Vertical gap > 10.0 -> VERTICAL_SECTOR
    2. wall_count >= 8 and entity_count >= 4 -> DENSE_SECTOR
    3. wall_count >= 6 -> CHAMBER_SECTOR
    4. wall_count <= 2 -> CORRIDOR_SECTOR
    5. Otherwise -> OPEN_SECTOR
    """
    gap = ceiling_height - floor_height
    if gap > 10.0:
        return VERTICAL_SECTOR
    if wall_count >= 8 and entity_count >= 4:
        return DENSE_SECTOR
    if wall_count >= 6:
        return CHAMBER_SECTOR
    if wall_count <= 2:
        return CORRIDOR_SECTOR
    return OPEN_SECTOR


# ---------------------------------------------------------------------------
# Projection classification
# ---------------------------------------------------------------------------


def classify_projection(
    world_mode: str,
    sector_count: int,
    primitive_count: int,
) -> str:
    """Deterministic projection classification.

    Rules:
    - PSEUDO_3D mode with sectors -> PSEUDO_DOOM
    - TRUE_3D with both sectors and primitives -> HYBRID_RETRO
    - TRUE_3D with primitives only -> PERSPECTIVE_3D
    - Otherwise -> ORTHO_LIKE
    """
    if world_mode == PSEUDO_3D:
        if sector_count > 0:
            return PSEUDO_DOOM
        return ORTHO_LIKE
    if world_mode == TRUE_3D:
        if sector_count > 0 and primitive_count > 0:
            return HYBRID_RETRO
        if primitive_count > 0:
            return PERSPECTIVE_3D
        return ORTHO_LIKE
    return ORTHO_LIKE


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------


def classify_complexity(
    primitive_count: int,
    sector_count: int,
    entity_count: int,
) -> str:
    """Deterministic scene complexity classification.

    Total = primitive_count + sector_count + entity_count.

    Rules:
    - total <= 3 -> MINIMAL
    - total <= 10 -> COMPACT
    - total <= 30 -> STRUCTURED
    - total > 30 -> DENSE
    """
    total = primitive_count + sector_count + entity_count
    if total <= 3:
        return MINIMAL
    if total <= 10:
        return COMPACT
    if total <= 30:
        return STRUCTURED
    return DENSE


# ---------------------------------------------------------------------------
# Symbolic trace
# ---------------------------------------------------------------------------


def build_symbolic_trace(
    world_mode: str,
    sector_classes: Tuple[str, ...],
    projection_class: str,
    complexity_class: str,
) -> str:
    """Build deterministic symbolic scene trace.

    Format:
      WORLD_MODE -> SECTOR_1 -> SECTOR_2 -> ... -> PROJECTION -> COMPLEXITY

    If no sectors, omits sector segment.
    """
    parts = [world_mode]
    for sc in sector_classes:
        parts.append(sc)
    parts.append(projection_class)
    parts.append(complexity_class)
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Primitive builder
# ---------------------------------------------------------------------------


def build_primitive(
    primitive_id: str,
    primitive_type: str,
    position: Any,
    scale: Any,
    rotation: Any,
    symbolic_class: str,
) -> RetroWorldPrimitive:
    """Build a validated, hash-stable RetroWorldPrimitive."""
    _validate_string(primitive_id, "primitive_id")
    _validate_string(primitive_type, "primitive_type")
    _validate_string(symbolic_class, "symbolic_class")

    if primitive_type not in VALID_PRIMITIVE_TYPES:
        raise ValueError(
            f"primitive_type must be one of {VALID_PRIMITIVE_TYPES}, "
            f"got {primitive_type!r}"
        )

    pos = _normalize_triple(position, "position")
    scl = _normalize_triple(scale, "scale")
    rot = _normalize_triple(rotation, "rotation")

    h = _compute_primitive_hash(
        primitive_id, primitive_type, pos, scl, rot, symbolic_class,
    )

    return RetroWorldPrimitive(
        primitive_id=primitive_id,
        primitive_type=primitive_type,
        position=pos,
        scale=scl,
        rotation=rot,
        symbolic_class=symbolic_class,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Sector builder
# ---------------------------------------------------------------------------


def build_sector(
    sector_id: str,
    floor_height: float,
    ceiling_height: float,
    wall_count: int,
    entity_count: int,
) -> RetroWorldSector:
    """Build a validated, hash-stable RetroWorldSector."""
    _validate_string(sector_id, "sector_id")

    if not isinstance(floor_height, (int, float)) or isinstance(floor_height, bool):
        raise TypeError(
            f"floor_height must be numeric, got {type(floor_height).__name__}"
        )
    if not isinstance(ceiling_height, (int, float)) or isinstance(ceiling_height, bool):
        raise TypeError(
            f"ceiling_height must be numeric, got {type(ceiling_height).__name__}"
        )

    floor_height = float(floor_height)
    ceiling_height = float(ceiling_height)

    if ceiling_height < floor_height:
        raise ValueError(
            f"ceiling_height ({ceiling_height}) must be >= "
            f"floor_height ({floor_height})"
        )

    _validate_non_negative_int(wall_count, "wall_count")
    _validate_non_negative_int(entity_count, "entity_count")

    sector_class = classify_sector(
        floor_height, ceiling_height, wall_count, entity_count,
    )

    h = _compute_sector_hash(
        sector_id, floor_height, ceiling_height, wall_count,
        entity_count, sector_class,
    )

    return RetroWorldSector(
        sector_id=sector_id,
        floor_height=floor_height,
        ceiling_height=ceiling_height,
        wall_count=wall_count,
        entity_count=entity_count,
        sector_class=sector_class,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core — build_retro_world_model
# ---------------------------------------------------------------------------


def build_retro_world_model(
    world_mode: str,
    primitives: Any,
    sectors: Any,
    camera_pose: Any,
    *,
    policy_hint: Optional[str] = None,
) -> RetroWorldModel:
    """Build a deterministic retro 3D world model.

    Parameters
    ----------
    world_mode : str
        One of TRUE_3D or PSEUDO_3D.
    primitives : iterable of RetroWorldPrimitive
        Scene primitives. May be empty for sector-only scenes.
    sectors : iterable of RetroWorldSector
        Scene sectors. May be empty for primitive-only scenes.
    camera_pose : 3-element iterable
        Camera position as (x, y, z).
    policy_hint : str, optional
        Optional supervisory policy hint for future coupling.
        Not used in v137.0.10 derivation logic.
    """
    _validate_string(world_mode, "world_mode")
    if world_mode not in VALID_WORLD_MODES:
        raise ValueError(
            f"world_mode must be one of {VALID_WORLD_MODES}, "
            f"got {world_mode!r}"
        )

    if policy_hint is not None and not isinstance(policy_hint, str):
        raise TypeError(
            f"policy_hint must be None or str, "
            f"got {type(policy_hint).__name__}"
        )

    cam = _normalize_triple(camera_pose, "camera_pose")

    # Normalize primitives
    if isinstance(primitives, str):
        raise TypeError("primitives must be a non-string iterable")
    try:
        prims = tuple(primitives)
    except TypeError:
        raise TypeError(
            f"primitives must be iterable, got {type(primitives).__name__}"
        )
    for i, p in enumerate(prims):
        if not isinstance(p, RetroWorldPrimitive):
            raise TypeError(
                f"primitives[{i}] must be RetroWorldPrimitive, "
                f"got {type(p).__name__}"
            )

    # Normalize sectors
    if isinstance(sectors, str):
        raise TypeError("sectors must be a non-string iterable")
    try:
        sects = tuple(sectors)
    except TypeError:
        raise TypeError(
            f"sectors must be iterable, got {type(sectors).__name__}"
        )
    for i, s in enumerate(sects):
        if not isinstance(s, RetroWorldSector):
            raise TypeError(
                f"sectors[{i}] must be RetroWorldSector, "
                f"got {type(s).__name__}"
            )

    primitive_count = len(prims)
    sector_count = len(sects)

    # Count entities from primitives
    entity_count = sum(
        1 for p in prims if p.primitive_type == PRIMITIVE_ENTITY
    )
    # Also count entities from sectors
    entity_count += sum(s.entity_count for s in sects)

    projection_class = classify_projection(
        world_mode, sector_count, primitive_count,
    )
    complexity_class = classify_complexity(
        primitive_count, sector_count, entity_count,
    )

    sector_classes = tuple(s.sector_class for s in sects)
    world_symbolic_trace = build_symbolic_trace(
        world_mode, sector_classes, projection_class, complexity_class,
    )

    h = _compute_model_hash(
        world_mode, primitive_count, sector_count, entity_count,
        cam, projection_class, complexity_class, world_symbolic_trace,
    )

    return RetroWorldModel(
        world_mode=world_mode,
        primitive_count=primitive_count,
        sector_count=sector_count,
        entity_count=entity_count,
        camera_pose=cam,
        projection_class=projection_class,
        complexity_class=complexity_class,
        world_symbolic_trace=world_symbolic_trace,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core — build_retro_world_ledger
# ---------------------------------------------------------------------------


def build_retro_world_ledger(
    models: Any,
) -> RetroWorldLedger:
    """Build an immutable ledger of retro world models.

    Parameters
    ----------
    models : iterable of RetroWorldModel
        World models to include in the ledger.
    """
    if isinstance(models, str):
        raise TypeError("models must be a non-string iterable")
    try:
        mdls = tuple(models)
    except TypeError:
        raise TypeError(
            f"models must be iterable, got {type(models).__name__}"
        )
    for i, m in enumerate(mdls):
        if not isinstance(m, RetroWorldModel):
            raise TypeError(
                f"models[{i}] must be RetroWorldModel, "
                f"got {type(m).__name__}"
            )

    h = _compute_ledger_hash(mdls)

    return RetroWorldLedger(
        models=mdls,
        model_count=len(mdls),
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core — export_retro_world_bundle
# ---------------------------------------------------------------------------


def export_retro_world_bundle(
    model: RetroWorldModel,
) -> Dict[str, Any]:
    """Export a world model as a replay-safe canonical bundle.

    Returns a deterministic dict suitable for JSON serialization.
    """
    if not isinstance(model, RetroWorldModel):
        raise TypeError(
            f"model must be RetroWorldModel, got {type(model).__name__}"
        )
    return {
        "camera_pose": list(_round_triple(model.camera_pose)),
        "complexity_class": model.complexity_class,
        "entity_count": model.entity_count,
        "primitive_count": model.primitive_count,
        "projection_class": model.projection_class,
        "sector_count": model.sector_count,
        "stable_hash": model.stable_hash,
        "version": model.version,
        "world_mode": model.world_mode,
        "world_symbolic_trace": model.world_symbolic_trace,
    }


# ---------------------------------------------------------------------------
# Core — export_retro_world_ledger
# ---------------------------------------------------------------------------


def export_retro_world_ledger(
    ledger: RetroWorldLedger,
) -> Dict[str, Any]:
    """Export a world ledger as a replay-safe canonical bundle.

    Returns a deterministic dict suitable for JSON serialization.
    """
    if not isinstance(ledger, RetroWorldLedger):
        raise TypeError(
            f"ledger must be RetroWorldLedger, got {type(ledger).__name__}"
        )
    return {
        "model_count": ledger.model_count,
        "models": [export_retro_world_bundle(m) for m in ledger.models],
        "stable_hash": ledger.stable_hash,
    }
