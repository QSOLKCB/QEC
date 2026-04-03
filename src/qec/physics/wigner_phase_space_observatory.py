"""
Wigner Phase-Space Observatory — v136.8.9

Canonical phase-space representation layer for QEC state.
Inspired by the Wigner quasi-probability distribution W(q, p).

q-axis: confidence / governance coordinate
p-axis: drift / rollback momentum coordinate

Negative quasi-probability regions are preserved (never clamped).
All outputs are deterministic and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: int = 11
Q_MIN: float = -1.0
Q_MAX: float = 1.0
P_MIN: float = -1.0
P_MAX: float = 1.0
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhasePoint:
    """Single point in Wigner phase space."""
    q: float
    p: float
    probability: float


@dataclass(frozen=True)
class WignerGrid:
    """Deterministic grid of phase-space points."""
    points: Tuple[PhasePoint, ...]
    grid_size: int
    negative_mass: float
    stable_hash: str


@dataclass(frozen=True)
class PhaseSpaceResult:
    """Complete phase-space observation result."""
    wigner_grid: WignerGrid
    centroid_q: float
    centroid_p: float
    drift_momentum: float
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — float normalisation
# ---------------------------------------------------------------------------

def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


# ---------------------------------------------------------------------------
# Feature extraction from integration targets
# ---------------------------------------------------------------------------

def _extract_confidence(cognition_result: Any) -> float:
    """Extract confidence scalar from cognition result."""
    if cognition_result is None:
        return 0.0
    if isinstance(cognition_result, dict):
        # Priority 1: nested match.confidence (exported cognition bundle)
        match_block = cognition_result.get("match")
        if isinstance(match_block, dict) and "confidence" in match_block:
            return float(match_block["confidence"])
        # Priority 2: top-level confidence
        return float(cognition_result.get("confidence", 0.0))
    # Priority 3: dataclass-like match.confidence
    if hasattr(cognition_result, "match") and hasattr(cognition_result.match, "confidence"):
        return float(cognition_result.match.confidence)
    # Priority 4: dataclass-like confidence
    if hasattr(cognition_result, "confidence"):
        return float(cognition_result.confidence)
    return 0.0


def _extract_gate_verdict_confidence(gate_result: Any) -> Tuple[str, float]:
    """Extract verdict and confidence from gate result."""
    if gate_result is None:
        return ("hold", 0.0)
    if isinstance(gate_result, dict):
        # Priority 1: nested decision block (exported gate bundle)
        decision_block = gate_result.get("decision")
        if isinstance(decision_block, dict):
            verdict = str(decision_block.get("verdict", "hold")).lower()
            confidence = float(decision_block.get("confidence", 0.0))
            return (verdict, confidence)
        # Priority 2: top-level verdict
        verdict = str(gate_result.get("verdict", "hold")).lower()
        confidence = float(gate_result.get("confidence", 0.0))
        return (verdict, confidence)
    # Priority 3: dataclass-like verdict
    if hasattr(gate_result, "verdict"):
        verdict = str(gate_result.verdict).lower()
        confidence = float(getattr(gate_result, "confidence", 0.0))
        return (verdict, confidence)
    return ("hold", 0.0)


def _extract_drift_score(history_ledger: Any) -> float:
    """Extract drift score from promotion history ledger."""
    if history_ledger is None:
        return 0.0
    if isinstance(history_ledger, dict):
        return float(history_ledger.get("drift_score", 0.0))
    if hasattr(history_ledger, "drift_score"):
        return float(history_ledger.drift_score)
    return 0.0


def _extract_rollback_rate(history_ledger: Any) -> float:
    """Extract rollback rate from promotion history ledger."""
    if history_ledger is None:
        return 0.0
    if isinstance(history_ledger, dict):
        return float(history_ledger.get("rollback_rate", 0.0))
    if hasattr(history_ledger, "rollback_rate"):
        return float(history_ledger.rollback_rate)
    return 0.0


def _extract_promotion_rate(history_ledger: Any) -> float:
    """Extract promotion rate from promotion history ledger."""
    if history_ledger is None:
        return 0.0
    if isinstance(history_ledger, dict):
        return float(history_ledger.get("promotion_rate", 0.0))
    if hasattr(history_ledger, "promotion_rate"):
        return float(history_ledger.promotion_rate)
    return 0.0


# ---------------------------------------------------------------------------
# Wigner kernel — deterministic quasi-probability computation
# ---------------------------------------------------------------------------

def _wigner_kernel(
    q: float,
    p: float,
    confidence: float,
    gate_confidence: float,
    verdict: str,
    drift_score: float,
    rollback_rate: float,
    promotion_rate: float,
) -> float:
    """
    Compute quasi-probability at a single phase-space point.

    The kernel is a deterministic function that produces negative
    regions — these MUST be preserved and never clamped.

    The design uses a modulated Gaussian-like kernel with interference
    terms that produce negative quasi-probability in regions of
    high drift or rollback pressure.
    """
    # Centre of the distribution based on governance state
    q0 = _round(confidence * 0.5 + gate_confidence * 0.5)
    p0 = _round(drift_score * 0.5 - rollback_rate * 0.5)

    # Widths
    sigma_q = max(0.2, 1.0 - confidence)
    sigma_p = max(0.2, 0.5 + drift_score)

    dq = q - q0
    dp = p - p0

    # Primary Gaussian envelope
    exponent = -0.5 * ((dq / sigma_q) ** 2 + (dp / sigma_p) ** 2)
    gauss = math.exp(exponent) / (2.0 * math.pi * sigma_q * sigma_p)

    # Interference term — produces negative regions
    # Rollback pressure and drift create oscillatory fringes
    rollback_factor = rollback_rate + drift_score
    interference = rollback_factor * math.cos(
        math.pi * (2.0 * dq + 3.0 * dp)
    ) * math.exp(-0.25 * (dq ** 2 + dp ** 2))

    # Verdict modulation
    verdict_sign = 1.0
    if verdict == "rollback":
        verdict_sign = -0.5
    elif verdict == "hold":
        verdict_sign = 0.0

    promotion_bias = promotion_rate * 0.1 * verdict_sign

    probability = _round(gauss + interference + promotion_bias)
    return probability


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _build_grid_axes(grid_size: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Build deterministic q and p axis values."""
    if grid_size <= 1:
        q_axis = (_round((Q_MIN + Q_MAX) / 2.0),)
        p_axis = (_round((P_MIN + P_MAX) / 2.0),)
        return (q_axis, p_axis)
    q_step = (Q_MAX - Q_MIN) / (grid_size - 1)
    p_step = (P_MAX - P_MIN) / (grid_size - 1)
    q_axis = tuple(_round(Q_MIN + i * q_step) for i in range(grid_size))
    p_axis = tuple(_round(P_MIN + i * p_step) for i in range(grid_size))
    return (q_axis, p_axis)


def build_wigner_grid_from_qec_state(
    cognition_result: Any,
    gate_result: Any,
    history_ledger: Any,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> WignerGrid:
    """
    Build a deterministic Wigner phase-space grid from QEC state.

    Points are ordered: primary sort by q, secondary sort by p.
    """
    confidence = _extract_confidence(cognition_result)
    verdict, gate_confidence = _extract_gate_verdict_confidence(gate_result)
    drift_score = _extract_drift_score(history_ledger)
    rollback_rate = _extract_rollback_rate(history_ledger)
    promotion_rate = _extract_promotion_rate(history_ledger)

    q_axis, p_axis = _build_grid_axes(grid_size)

    points = []
    for q_val in q_axis:
        for p_val in p_axis:
            prob = _wigner_kernel(
                q=q_val,
                p=p_val,
                confidence=confidence,
                gate_confidence=gate_confidence,
                verdict=verdict,
                drift_score=drift_score,
                rollback_rate=rollback_rate,
                promotion_rate=promotion_rate,
            )
            points.append(PhasePoint(q=q_val, p=p_val, probability=prob))

    points_tuple: Tuple[PhasePoint, ...] = tuple(points)
    neg_mass = compute_negative_mass_from_points(points_tuple)
    grid_hash = _compute_grid_hash(points_tuple, grid_size, neg_mass)

    return WignerGrid(
        points=points_tuple,
        grid_size=grid_size,
        negative_mass=neg_mass,
        stable_hash=grid_hash,
    )


# ---------------------------------------------------------------------------
# Phase-space observables
# ---------------------------------------------------------------------------

def compute_negative_mass_from_points(points: Tuple[PhasePoint, ...]) -> float:
    """Sum of |probability| for all points where probability < 0."""
    total = 0.0
    for pt in points:
        if pt.probability < 0.0:
            total += abs(pt.probability)
    return _round(total)


def compute_negative_mass(grid: WignerGrid) -> float:
    """Compute negative mass from a WignerGrid."""
    return compute_negative_mass_from_points(grid.points)


def compute_phase_centroid(grid: WignerGrid) -> Tuple[float, float]:
    """
    Compute the centroid (mean q, mean p) weighted by |probability|.

    Uses absolute probability to ensure centroid is well-defined
    even with negative quasi-probability regions.
    """
    total_weight = 0.0
    sum_q = 0.0
    sum_p = 0.0
    for pt in grid.points:
        w = abs(pt.probability)
        sum_q += pt.q * w
        sum_p += pt.p * w
        total_weight += w
    if total_weight == 0.0:
        return (0.0, 0.0)
    return (_round(sum_q / total_weight), _round(sum_p / total_weight))


def compute_phase_drift_momentum(grid: WignerGrid) -> float:
    """
    Compute drift momentum scalar from the p-axis structure.

    This is the |probability|-weighted mean of p values,
    representing the net momentum in drift/rollback space.
    """
    total_weight = 0.0
    sum_p = 0.0
    for pt in grid.points:
        w = abs(pt.probability)
        sum_p += pt.p * w
        total_weight += w
    if total_weight == 0.0:
        return 0.0
    return _round(sum_p / total_weight)


# ---------------------------------------------------------------------------
# Hashing — canonical JSON + SHA-256
# ---------------------------------------------------------------------------

def _point_to_dict(pt: PhasePoint) -> Dict[str, float]:
    """Convert PhasePoint to dict with normalised floats."""
    return {
        "q": _round(pt.q),
        "p": _round(pt.p),
        "probability": _round(pt.probability),
    }


def _compute_grid_hash(
    points: Tuple[PhasePoint, ...],
    grid_size: int,
    negative_mass: float,
) -> str:
    """SHA-256 of canonical JSON representation of grid data."""
    payload = {
        "grid_size": grid_size,
        "negative_mass": _round(negative_mass),
        "points": [_point_to_dict(pt) for pt in points],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_phase_space_hash(result: PhaseSpaceResult) -> str:
    """SHA-256 of the full PhaseSpaceResult."""
    payload = {
        "centroid_q": _round(result.centroid_q),
        "centroid_p": _round(result.centroid_p),
        "drift_momentum": _round(result.drift_momentum),
        "grid_hash": result.wigner_grid.stable_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_phase_space_cycle(
    cognition_result: Any,
    gate_result: Any,
    history_ledger: Any,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> PhaseSpaceResult:
    """
    Execute a full Wigner phase-space observation cycle.

    Deterministic: same inputs → identical PhaseSpaceResult.
    """
    grid = build_wigner_grid_from_qec_state(
        cognition_result=cognition_result,
        gate_result=gate_result,
        history_ledger=history_ledger,
        grid_size=grid_size,
    )

    centroid_q, centroid_p = compute_phase_centroid(grid)
    drift_momentum = compute_phase_drift_momentum(grid)

    # Build result without hash first, then compute hash
    preliminary = PhaseSpaceResult(
        wigner_grid=grid,
        centroid_q=centroid_q,
        centroid_p=centroid_p,
        drift_momentum=drift_momentum,
        stable_hash="",
    )
    result_hash = compute_phase_space_hash(preliminary)

    return PhaseSpaceResult(
        wigner_grid=grid,
        centroid_q=centroid_q,
        centroid_p=centroid_p,
        drift_momentum=drift_momentum,
        stable_hash=result_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_phase_space_bundle(result: PhaseSpaceResult) -> Dict[str, Any]:
    """Export phase-space result as a stable dictionary bundle."""
    return {
        "version": "v136.8.9",
        "layer": "wigner_phase_space_observatory",
        "centroid_q": _round(result.centroid_q),
        "centroid_p": _round(result.centroid_p),
        "drift_momentum": _round(result.drift_momentum),
        "grid_size": result.wigner_grid.grid_size,
        "negative_mass": _round(result.wigner_grid.negative_mass),
        "grid_hash": result.wigner_grid.stable_hash,
        "stable_hash": result.stable_hash,
        "point_count": len(result.wigner_grid.points),
        "has_negative_regions": any(
            pt.probability < 0.0 for pt in result.wigner_grid.points
        ),
    }
