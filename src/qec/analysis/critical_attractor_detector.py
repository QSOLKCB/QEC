"""v118.0.0 — Deterministic critical attractor detector layer."""

from __future__ import annotations

import math
from typing import Iterable

MIN_CYCLE_LENGTH = 2
MAX_CYCLE_SCAN = 32
CRITICAL_RISK_THRESHOLD = 0.75
SIMILARITY_EPSILON = 1e-9


def _clamp_unit(value: float) -> float:
    """Clamp a numeric value to [0, 1] with non-finite fallback to 0.0."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return 0.0
    if numeric_value <= 0.0:
        return 0.0
    if numeric_value >= 1.0:
        return 1.0
    return numeric_value


def detect_cycles(
    state_sequence: Iterable[object],
    max_scan: int = MAX_CYCLE_SCAN,
) -> tuple[bool, list[object]]:
    """Deterministically detect repeated cycles in a bounded suffix scan."""
    sequence = list(state_sequence)
    if len(sequence) < MIN_CYCLE_LENGTH * 2:
        return False, []

    scan = max(1, int(max_scan))
    tail = sequence[-scan:]
    tail_length = len(tail)

    max_cycle_length = min(tail_length // 2, scan, MAX_CYCLE_SCAN)
    for cycle_length in range(MIN_CYCLE_LENGTH, max_cycle_length + 1):
        suffix = tail[-cycle_length:]
        previous = tail[-(2 * cycle_length) : -cycle_length]
        if suffix == previous:
            return True, list(suffix)

    return False, []


def compute_cycle_signature(cycle: Iterable[object]) -> tuple[object, ...]:
    """Return a deterministic hashable cycle signature."""
    return tuple(cycle)


def compare_attractors(
    cycle_a: Iterable[object],
    cycle_b: Iterable[object],
) -> float:
    """Compute deterministic normalized overlap similarity in [0, 1]."""
    sequence_a = list(cycle_a)
    sequence_b = list(cycle_b)

    if not sequence_a and not sequence_b:
        return 0.0
    if not sequence_a or not sequence_b:
        return 0.0

    max_len = max(len(sequence_a), len(sequence_b))
    overlap = min(len(sequence_a), len(sequence_b))
    exact_matches = 0

    for index in range(overlap):
        if sequence_a[index] == sequence_b[index]:
            exact_matches += 1

    return _clamp_unit(exact_matches / max_len)


def compute_critical_risk_score(
    warning_score: float,
    cycle_length: int,
    similarity_score: float,
) -> float:
    """Compute deterministic bounded critical risk score."""
    warning = _clamp_unit(warning_score)
    similarity = _clamp_unit(similarity_score)
    cycle_length_normalized = _clamp_unit(float(cycle_length) / float(MAX_CYCLE_SCAN))

    score = (0.5 * warning) + (0.3 * similarity) + (0.2 * cycle_length_normalized)
    return _clamp_unit(score)


def detect_basin_lock(
    cycle_length: int,
    similarity_score: float,
) -> bool:
    """Detect stable critical basin lock signature."""
    return int(cycle_length) >= MIN_CYCLE_LENGTH and _clamp_unit(similarity_score) >= 0.8


def classify_attractor_state(risk_score: float) -> str:
    """Classify attractor state as nominal, elevated, or critical."""
    score = _clamp_unit(risk_score)
    if score < 0.3:
        return "nominal"
    if score < CRITICAL_RISK_THRESHOLD:
        return "elevated"
    return "critical"


def run_critical_attractor_detector(
    state_sequence: Iterable[object],
    warning_score: float,
    baseline_cycle: Iterable[object] | None = None,
) -> dict[str, bool | float | int | tuple[object, ...] | str]:
    """Run deterministic critical attractor detector and return bounded summary."""
    cycle_detected, cycle = detect_cycles(state_sequence)
    cycle_length = len(cycle)
    cycle_signature = compute_cycle_signature(cycle)

    if not cycle_detected:
        similarity_score = 0.0
        risk_score = 0.0
        basin_lock_detected = False
    else:
        baseline = list(baseline_cycle) if baseline_cycle is not None else cycle
        similarity_score = compare_attractors(cycle, baseline)
        risk_score = compute_critical_risk_score(
            warning_score=warning_score,
            cycle_length=cycle_length,
            similarity_score=similarity_score,
        )
        basin_lock_detected = detect_basin_lock(cycle_length, similarity_score)
    attractor_state = classify_attractor_state(risk_score)

    result: dict[str, bool | float | int | tuple[object, ...] | str] = {
        "cycle_detected": cycle_detected,
        "cycle_length": cycle_length,
        "cycle_signature": cycle_signature,
        "attractor_similarity_score": _clamp_unit(similarity_score),
        "critical_risk_score": _clamp_unit(risk_score),
        "attractor_state": attractor_state,
        "basin_lock_detected": basin_lock_detected,
        "risk_triggered": _clamp_unit(risk_score) >= CRITICAL_RISK_THRESHOLD,
    }
    return result


__all__ = [
    "MIN_CYCLE_LENGTH",
    "MAX_CYCLE_SCAN",
    "CRITICAL_RISK_THRESHOLD",
    "SIMILARITY_EPSILON",
    "detect_cycles",
    "compute_cycle_signature",
    "compare_attractors",
    "compute_critical_risk_score",
    "detect_basin_lock",
    "classify_attractor_state",
    "run_critical_attractor_detector",
]
