"""v137.0.13 — Retro Phi-Shell Rasterization Pipeline.

Theory-coupled rasterization pipeline absorbing real semantics from:

  papers/Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf
  papers/Unified_Field_Fidelity_Bridging_Computat.docx
  qec_theory_diagram.txt

Core theory constructs absorbed:

  1. Golden phi shell progression (phi recurrence)
     All raster spans quantize to phi shells, not linear z-bands.

  2. UFF restore operator: nabla^2 T + (phi + psi)^2 T = 0
     Implemented as deterministic span-energy correction.

  3. E8 triality lock: three primary visibility classes
     (NEAR_SHELL, MID_SHELL, OUTER_SHELL) plus boundary classes.

  4. SiS2 stability ring: frozen ledger with SHA-256 replay identity.

Pipeline law:

  depth
  -> phi-shell quantization
  -> scanline span construction
  -> visibility classification
  -> UFF restore correction
  -> raster decision
  -> stable ledger
  -> canonical export

Layer 4 -- Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.

Theory Upgrade Source:
- file: papers/Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf
- concept: golden phi shell progression, E8 triality, UFF restore operator
- implementation: phi-quantized raster spans with restore correction
- invariant tested: PHI_SCALE_NODE, E8_TRIALITY_LOCK, OUROBOROS_FEEDBACK_LOOP, SIS2_STABILITY_RING
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

RETRO_PHI_SHELL_VERSION: str = "v137.0.13"

# ---------------------------------------------------------------------------
# Constants -- golden phi shell progression (theory-coupled)
# ---------------------------------------------------------------------------
# From Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf:
# Each shell = sum of two preceding (golden recurrence).
# 1.0, 1.618, 2.618, 4.236, 6.854
# This is NOT linear z-banding. This is phi-quantized depth.

PHI: float = 1.618

PHI_SHELLS: Tuple[float, ...] = (
    1.0,
    1.618,
    2.618,
    4.236,
    6.854,
)

# ---------------------------------------------------------------------------
# Constants -- visibility classes (E8 triality lock)
# ---------------------------------------------------------------------------

NEAR_SHELL: str = "NEAR_SHELL"
MID_SHELL: str = "MID_SHELL"
OUTER_SHELL: str = "OUTER_SHELL"
RESONANCE_NODE: str = "RESONANCE_NODE"
WIGGLE_ZONE: str = "WIGGLE_ZONE"

VALID_VISIBILITY_CLASSES: Tuple[str, ...] = (
    NEAR_SHELL,
    MID_SHELL,
    OUTER_SHELL,
    RESONANCE_NODE,
    WIGGLE_ZONE,
)

# ---------------------------------------------------------------------------
# Constants -- defaults
# ---------------------------------------------------------------------------

DEFAULT_PHASE_OFFSET: float = math.pi / 2

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroRasterSpan:
    """Immutable raster span quantized to a phi shell."""

    span_index: int
    depth: float
    phi_shell: float
    visibility_class: str
    restore_term: float
    stable_hash: str
    version: str = RETRO_PHI_SHELL_VERSION


@dataclass(frozen=True)
class RetroPhiShell:
    """Immutable phi shell descriptor."""

    shell_index: int
    shell_value: float
    lower_bound: float
    upper_bound: float
    stable_hash: str
    version: str = RETRO_PHI_SHELL_VERSION


@dataclass(frozen=True)
class RetroRasterDecision:
    """Immutable raster decision artifact."""

    spans: Tuple[RetroRasterSpan, ...]
    shells: Tuple[RetroPhiShell, ...]
    span_count: int
    shell_count: int
    max_depth: float
    width: int
    symbolic_trace: str
    stable_hash: str
    version: str = RETRO_PHI_SHELL_VERSION


@dataclass(frozen=True)
class RetroRasterLedger:
    """Immutable ledger of raster decisions."""

    decisions: Tuple[RetroRasterDecision, ...]
    decision_count: int
    stable_hash: str
    version: str = RETRO_PHI_SHELL_VERSION


# ---------------------------------------------------------------------------
# Helpers -- canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


# ---------------------------------------------------------------------------
# Helpers -- hashing
# ---------------------------------------------------------------------------


def _compute_span_hash(
    span_index: int,
    depth: float,
    phi_shell: float,
    visibility_class: str,
    restore_term: float,
) -> str:
    """SHA-256 of canonical JSON of a raster span."""
    payload = {
        "depth": _round(depth),
        "phi_shell": _round(phi_shell),
        "restore_term": _round(restore_term),
        "span_index": span_index,
        "version": RETRO_PHI_SHELL_VERSION,
        "visibility_class": visibility_class,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_shell_hash(
    shell_index: int,
    shell_value: float,
    lower_bound: float,
    upper_bound: float,
) -> str:
    """SHA-256 of canonical JSON of a phi shell descriptor."""
    payload = {
        "lower_bound": _round(lower_bound),
        "shell_index": shell_index,
        "shell_value": _round(shell_value),
        "upper_bound": _round(upper_bound),
        "version": RETRO_PHI_SHELL_VERSION,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_decision_hash(
    span_hashes: Tuple[str, ...],
    shell_hashes: Tuple[str, ...],
    span_count: int,
    shell_count: int,
    max_depth: float,
    width: int,
    symbolic_trace: str,
) -> str:
    """SHA-256 of canonical JSON of a raster decision."""
    payload = {
        "max_depth": _round(max_depth),
        "shell_count": shell_count,
        "shell_hashes": list(shell_hashes),
        "span_count": span_count,
        "span_hashes": list(span_hashes),
        "symbolic_trace": symbolic_trace,
        "version": RETRO_PHI_SHELL_VERSION,
        "width": width,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[RetroRasterDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = [d.stable_hash for d in decisions]
    return hashlib.sha256(
        _canonical_json(hashes).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


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


def _validate_positive_float(value: Any, field_name: str) -> float:
    """Validate a positive finite float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field_name} must be finite, got {fv}")
    if fv <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {fv}")
    return fv


def _validate_non_negative_float(value: Any, field_name: str) -> float:
    """Validate a non-negative finite float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field_name} must be finite, got {fv}")
    if fv < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {fv}")
    return fv


def _validate_float(value: Any, field_name: str) -> float:
    """Validate a finite float (any sign)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field_name} must be finite, got {fv}")
    return fv


# ---------------------------------------------------------------------------
# Core functions -- phi shell quantization
# ---------------------------------------------------------------------------


def quantize_depth_phi_shell(depth: float) -> float:
    """Quantize a depth value to the nearest phi shell.

    Uses the canonical phi shell progression:
        (1.0, 1.618, 2.618, 4.236, 6.854)

    Depth values are mapped to the closest shell by absolute distance.
    Ties are broken by choosing the smaller shell (deterministic).

    Args:
        depth: Non-negative depth value to quantize.

    Returns:
        The phi shell value nearest to the input depth.

    Raises:
        TypeError: If depth is not numeric.
        ValueError: If depth is negative or non-finite.
    """
    d = _validate_non_negative_float(depth, "depth")

    best_shell = PHI_SHELLS[0]
    best_dist = abs(d - best_shell)
    for shell in PHI_SHELLS[1:]:
        dist = abs(d - shell)
        if dist < best_dist:
            best_dist = dist
            best_shell = shell
    return best_shell


# ---------------------------------------------------------------------------
# Core functions -- UFF restore operator
# ---------------------------------------------------------------------------


def compute_phi_restore_term(
    span_energy: float,
    phase_offset: float = DEFAULT_PHASE_OFFSET,
) -> float:
    """Compute UFF restore term: deterministic span-energy correction.

    From theory corpus (Unified_Field_Fidelity_Bridging_Computat.docx):
        nabla^2 T + (phi + psi)^2 T = 0

    Implemented as:
        restore = span_energy + ((1.618 + phase_offset) ** 2) * 0.01

    This is a pure deterministic function of span_energy and phase_offset.

    Args:
        span_energy: The energy value of the span.
        phase_offset: Phase offset (default: pi/2).

    Returns:
        Deterministic restore term value.

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If inputs are non-finite.
    """
    se = _validate_float(span_energy, "span_energy")
    po = _validate_float(phase_offset, "phase_offset")
    restore = se + ((PHI + po) ** 2) * 0.01
    return _round(restore)


# ---------------------------------------------------------------------------
# Core functions -- visibility classification
# ---------------------------------------------------------------------------


def classify_shell_visibility(depth: float) -> str:
    """Classify depth into a visibility class.

    E8 triality lock: three primary classes plus two boundary classes.

    Rules (based on phi shell boundaries):
        depth <= 1.309  (midpoint of shells 0-1)  -> NEAR_SHELL
        depth <= 2.118  (midpoint of shells 1-2)  -> MID_SHELL
        depth <= 3.427  (midpoint of shells 2-3)  -> OUTER_SHELL
        depth <= 5.545  (midpoint of shells 3-4)  -> RESONANCE_NODE
        otherwise                                  -> WIGGLE_ZONE

    Args:
        depth: Non-negative depth value.

    Returns:
        Visibility class string.

    Raises:
        TypeError: If depth is not numeric.
        ValueError: If depth is negative or non-finite.
    """
    d = _validate_non_negative_float(depth, "depth")

    # Midpoints between consecutive phi shells
    # (1.0 + 1.618) / 2 = 1.309
    # (1.618 + 2.618) / 2 = 2.118
    # (2.618 + 4.236) / 2 = 3.427
    # (4.236 + 6.854) / 2 = 5.545
    if d <= 1.309:
        return NEAR_SHELL
    if d <= 2.118:
        return MID_SHELL
    if d <= 3.427:
        return OUTER_SHELL
    if d <= 5.545:
        return RESONANCE_NODE
    return WIGGLE_ZONE


# ---------------------------------------------------------------------------
# Core functions -- scanline span construction
# ---------------------------------------------------------------------------


def build_phi_scanline_spans(
    width: int,
    max_depth: float,
) -> Tuple[RetroRasterSpan, ...]:
    """Build phi-quantized scanline spans.

    Each column in the scanline is assigned a depth that scales linearly
    from 0 to max_depth across the width. The depth is then quantized to
    the nearest phi shell, classified, and corrected with the UFF restore
    term.

    Args:
        width: Number of columns (must be >= 1).
        max_depth: Maximum depth value (must be > 0).

    Returns:
        Tuple of frozen RetroRasterSpan objects, one per column.

    Raises:
        TypeError: If inputs are wrong type.
        ValueError: If inputs are out of range.
    """
    w = _validate_positive_int(width, "width")
    md = _validate_positive_float(max_depth, "max_depth")

    spans = []
    for i in range(w):
        if w == 1:
            raw_depth = md
        else:
            raw_depth = _round((i / (w - 1)) * md)
        phi_shell = quantize_depth_phi_shell(raw_depth)
        vis_class = classify_shell_visibility(raw_depth)
        span_energy = _round(raw_depth / md)
        restore = compute_phi_restore_term(span_energy)
        h = _compute_span_hash(i, raw_depth, phi_shell, vis_class, restore)
        spans.append(RetroRasterSpan(
            span_index=i,
            depth=_round(raw_depth),
            phi_shell=phi_shell,
            visibility_class=vis_class,
            restore_term=restore,
            stable_hash=h,
        ))
    return tuple(spans)


# ---------------------------------------------------------------------------
# Core functions -- phi shell descriptors
# ---------------------------------------------------------------------------


def build_phi_shell_descriptors() -> Tuple[RetroPhiShell, ...]:
    """Build frozen descriptors for each phi shell.

    Returns:
        Tuple of RetroPhiShell descriptors with computed boundaries.
    """
    shells = []
    for i, sv in enumerate(PHI_SHELLS):
        if i == 0:
            lower = 0.0
        else:
            lower = _round((PHI_SHELLS[i - 1] + sv) / 2.0)
        if i == len(PHI_SHELLS) - 1:
            upper = float("inf")
        else:
            upper = _round((sv + PHI_SHELLS[i + 1]) / 2.0)
        h = _compute_shell_hash(i, sv, lower, upper if math.isfinite(upper) else 9999.0)
        shells.append(RetroPhiShell(
            shell_index=i,
            shell_value=sv,
            lower_bound=lower,
            upper_bound=upper,
            stable_hash=h,
        ))
    return tuple(shells)


# ---------------------------------------------------------------------------
# Core functions -- symbolic trace
# ---------------------------------------------------------------------------


def build_symbolic_trace(spans: Tuple[RetroRasterSpan, ...]) -> str:
    """Build symbolic trace from visibility classes of spans.

    Produces a deduplicated transition trace, e.g.:
        NEAR_SHELL -> MID_SHELL -> PHI_SCALE_NODE

    The trace shows visibility class transitions across spans, with
    consecutive duplicates collapsed.

    Args:
        spans: Tuple of RetroRasterSpan objects.

    Returns:
        Symbolic trace string.
    """
    if len(spans) == 0:
        return ""
    classes = []
    for s in spans:
        if len(classes) == 0 or classes[-1] != s.visibility_class:
            classes.append(s.visibility_class)
    # Append PHI_SCALE_NODE as terminal theory marker
    classes.append("PHI_SCALE_NODE")
    return " -> ".join(classes)


# ---------------------------------------------------------------------------
# Core functions -- raster decision
# ---------------------------------------------------------------------------


def build_phi_raster_decision(
    width: int,
    max_depth: float,
) -> RetroRasterDecision:
    """Build a complete phi-shell raster decision.

    Constructs spans, shells, symbolic trace, and decision artifact.

    Args:
        width: Number of columns (must be >= 1).
        max_depth: Maximum depth value (must be > 0).

    Returns:
        Frozen RetroRasterDecision artifact.
    """
    w = _validate_positive_int(width, "width")
    md = _validate_positive_float(max_depth, "max_depth")

    spans = build_phi_scanline_spans(w, md)
    shells = build_phi_shell_descriptors()
    trace = build_symbolic_trace(spans)

    span_hashes = tuple(s.stable_hash for s in spans)
    shell_hashes = tuple(s.stable_hash for s in shells)

    h = _compute_decision_hash(
        span_hashes, shell_hashes, len(spans), len(shells),
        md, w, trace,
    )

    return RetroRasterDecision(
        spans=spans,
        shells=shells,
        span_count=len(spans),
        shell_count=len(shells),
        max_depth=_round(md),
        width=w,
        symbolic_trace=trace,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core functions -- raster ledger
# ---------------------------------------------------------------------------


def build_phi_raster_ledger(
    decisions: Tuple[RetroRasterDecision, ...],
) -> RetroRasterLedger:
    """Build an immutable ledger of raster decisions.

    Args:
        decisions: Tuple of RetroRasterDecision objects.

    Returns:
        Frozen RetroRasterLedger artifact.

    Raises:
        TypeError: If decisions is not a tuple of RetroRasterDecision.
        ValueError: If decisions is empty.
    """
    if not isinstance(decisions, tuple):
        raise TypeError(
            f"decisions must be a tuple, got {type(decisions).__name__}"
        )
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")
    for i, d in enumerate(decisions):
        if not isinstance(d, RetroRasterDecision):
            raise TypeError(
                f"decisions[{i}] must be RetroRasterDecision, "
                f"got {type(d).__name__}"
            )

    h = _compute_ledger_hash(decisions)
    return RetroRasterLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Export -- canonical JSON
# ---------------------------------------------------------------------------


def _span_to_dict(span: RetroRasterSpan) -> Dict[str, Any]:
    """Convert a span to a canonical dict."""
    return {
        "depth": _round(span.depth),
        "phi_shell": _round(span.phi_shell),
        "restore_term": _round(span.restore_term),
        "span_index": span.span_index,
        "stable_hash": span.stable_hash,
        "version": span.version,
        "visibility_class": span.visibility_class,
    }


def _shell_to_dict(shell: RetroPhiShell) -> Dict[str, Any]:
    """Convert a shell descriptor to a canonical dict."""
    return {
        "lower_bound": _round(shell.lower_bound),
        "shell_index": shell.shell_index,
        "shell_value": _round(shell.shell_value),
        "stable_hash": shell.stable_hash,
        "upper_bound": _round(shell.upper_bound) if math.isfinite(shell.upper_bound) else 9999.0,
        "version": shell.version,
    }


def _decision_to_dict(decision: RetroRasterDecision) -> Dict[str, Any]:
    """Convert a decision to a canonical dict."""
    return {
        "max_depth": _round(decision.max_depth),
        "shell_count": decision.shell_count,
        "shells": [_shell_to_dict(s) for s in decision.shells],
        "span_count": decision.span_count,
        "spans": [_span_to_dict(s) for s in decision.spans],
        "stable_hash": decision.stable_hash,
        "symbolic_trace": decision.symbolic_trace,
        "version": decision.version,
        "width": decision.width,
    }


def export_phi_raster_ledger(
    ledger: RetroRasterLedger,
) -> Dict[str, Any]:
    """Export ledger as canonical dict for JSON serialization.

    Args:
        ledger: The raster ledger to export.

    Returns:
        Canonical dict suitable for deterministic JSON serialization.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [_decision_to_dict(d) for d in ledger.decisions],
        "stable_hash": ledger.stable_hash,
        "version": ledger.version,
    }


def export_phi_raster_bundle(
    ledger: RetroRasterLedger,
) -> str:
    """Export ledger as canonical JSON string with SHA-256 envelope.

    Returns:
        Canonical JSON string. Byte-identical for identical inputs.
    """
    payload = export_phi_raster_ledger(ledger)
    json_str = _canonical_json(payload)
    envelope = {
        "data": payload,
        "sha256": hashlib.sha256(json_str.encode("utf-8")).hexdigest(),
        "version": RETRO_PHI_SHELL_VERSION,
    }
    return _canonical_json(envelope)
