"""v142.1 — Generalized Invariant Detector.

Deterministic Layer-4 analysis-only module for simple structural invariants in
iterative execution traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from qec.analysis.iterative_system_abstraction_layer import (
    IterativeExecutionReceipt,
    IterativeExecutionTrace,
    IterativeStateSnapshot,
    IterativeTransition,
)

GENERALIZED_INVARIANT_DETECTOR_VERSION = "v142.1"
_CONTROL_MODE = "generalized_invariant_advisory"
_LABEL_TO_RANK: dict[str, int] = {
    "none": 0,
    "fixed_point": 1,
    "repeated_state": 2,
    "plateau": 3,
    "oscillation": 4,
}
_PATTERN_TYPES: tuple[str, ...] = ("fixed_point", "repeated_state", "plateau", "oscillation")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return out


@dataclass(frozen=True)
class InvariantSignal:
    repeated_state_score: float
    fixed_point_score: float
    plateau_score: float
    oscillation_score: float
    invariant_pressure: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "repeated_state_score", _bounded(self.repeated_state_score, "repeated_state_score"))
        object.__setattr__(self, "fixed_point_score", _bounded(self.fixed_point_score, "fixed_point_score"))
        object.__setattr__(self, "plateau_score", _bounded(self.plateau_score, "plateau_score"))
        object.__setattr__(self, "oscillation_score", _bounded(self.oscillation_score, "oscillation_score"))
        object.__setattr__(self, "invariant_pressure", _bounded(self.invariant_pressure, "invariant_pressure"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeated_state_score": self.repeated_state_score,
            "fixed_point_score": self.fixed_point_score,
            "plateau_score": self.plateau_score,
            "oscillation_score": self.oscillation_score,
            "invariant_pressure": self.invariant_pressure,
        }


@dataclass(frozen=True)
class InvariantDecision:
    dominant_invariant: str
    invariant_rank: int
    invariant_detected: bool
    invariant_confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if self.dominant_invariant not in _LABEL_TO_RANK:
            raise ValueError("invalid invariant label")
        if not isinstance(self.invariant_rank, int) or self.invariant_rank != _LABEL_TO_RANK[self.dominant_invariant]:
            raise ValueError("invalid rank mapping")
        if not isinstance(self.invariant_detected, bool):
            raise ValueError("invariant_detected must be bool")
        object.__setattr__(self, "invariant_confidence", _bounded(self.invariant_confidence, "invariant_confidence"))

        expected_rationale = {
            "none": "no_invariant",
            "fixed_point": "fixed_point_detected",
            "repeated_state": "repeated_state_detected",
            "plateau": "plateau_detected",
            "oscillation": "oscillation_detected",
        }[self.dominant_invariant]
        if self.rationale != expected_rationale:
            raise ValueError("invalid rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_invariant": self.dominant_invariant,
            "invariant_rank": self.invariant_rank,
            "invariant_detected": self.invariant_detected,
            "invariant_confidence": self.invariant_confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class InvariantPattern:
    pattern_type: str
    key: str
    support: int
    confidence: float

    def __post_init__(self) -> None:
        if self.pattern_type not in _PATTERN_TYPES:
            raise ValueError("invalid invariant label")
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("key must be non-empty str")
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 1:
            raise ValueError("support must be int >= 1")
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "key": self.key,
            "support": self.support,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class InvariantDetectionReceipt:
    version: str
    signal: InvariantSignal
    decision: InvariantDecision
    patterns: tuple[InvariantPattern, ...]
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if not isinstance(self.signal, InvariantSignal):
            raise ValueError("signal must be InvariantSignal")
        if not isinstance(self.decision, InvariantDecision):
            raise ValueError("decision must be InvariantDecision")
        if not isinstance(self.patterns, tuple):
            raise ValueError("patterns must be tuple[InvariantPattern, ...]")
        for pattern in self.patterns:
            if not isinstance(pattern, InvariantPattern):
                raise ValueError("patterns must be tuple[InvariantPattern, ...]")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", _sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signal": self.signal.to_dict(),
            "decision": self.decision.to_dict(),
            "patterns": tuple(pattern.to_dict() for pattern in self.patterns),
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _detect_fixed_point(snapshots: tuple[IterativeStateSnapshot, ...], transitions: tuple[IterativeTransition, ...]) -> tuple[float, InvariantPattern | None]:
    if len(snapshots) < 2 or not transitions:
        return 0.0, None
    if snapshots[-1].state_id == snapshots[-2].state_id and transitions[-1].transition_label in {"stabilize", "converge"}:
        return 1.0, InvariantPattern(pattern_type="fixed_point", key=snapshots[-1].state_id, support=2, confidence=1.0)
    return 0.0, None


def _detect_plateau(snapshots: tuple[IterativeStateSnapshot, ...]) -> tuple[float, InvariantPattern | None]:
    n = len(snapshots)
    if n == 0:
        return 0.0, None
    longest_run = 1
    current_run = 1
    for idx in range(1, n):
        delta = abs(snapshots[idx].convergence_metric - snapshots[idx - 1].convergence_metric)
        if delta < 0.01:
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 1

    if longest_run < 3:
        return 0.0, None
    score = _clamp01((longest_run - 2) / max(n - 2, 1))
    return score, InvariantPattern(pattern_type="plateau", key="plateau", support=longest_run, confidence=score)


def _detect_oscillation(snapshots: tuple[IterativeStateSnapshot, ...]) -> tuple[float, InvariantPattern | None]:
    n = len(snapshots)
    longest = 0
    for start in range(0, max(n - 1, 0)):
        a = snapshots[start].state_id
        b = snapshots[start + 1].state_id
        if a == b:
            continue
        run = 2
        idx = start + 2
        while idx < n:
            expected = a if ((idx - start) % 2 == 0) else b
            if snapshots[idx].state_id != expected:
                break
            run += 1
            idx += 1
        if run > longest:
            longest = run

    if longest < 4:
        return 0.0, None
    score = _clamp01((longest - 3) / max(n - 3, 1))
    return score, InvariantPattern(pattern_type="oscillation", key="A<->B", support=longest, confidence=score)


def _dominant_decision(signal: InvariantSignal) -> InvariantDecision:
    candidates: tuple[tuple[str, float], ...] = (
        ("fixed_point", signal.fixed_point_score),
        ("repeated_state", signal.repeated_state_score),
        ("plateau", signal.plateau_score),
        ("oscillation", signal.oscillation_score),
    )
    max_score = max(score for _, score in candidates)
    if max_score <= 0.0:
        return InvariantDecision(
            dominant_invariant="none",
            invariant_rank=_LABEL_TO_RANK["none"],
            invariant_detected=False,
            invariant_confidence=0.0,
            rationale="no_invariant",
        )

    for label, score in candidates:
        if score == max_score:
            return InvariantDecision(
                dominant_invariant=label,
                invariant_rank=_LABEL_TO_RANK[label],
                invariant_detected=True,
                invariant_confidence=score,
                rationale={
                    "fixed_point": "fixed_point_detected",
                    "repeated_state": "repeated_state_detected",
                    "plateau": "plateau_detected",
                    "oscillation": "oscillation_detected",
                }[label],
            )
    raise ValueError("invalid invariant label")


def evaluate_generalized_invariant_detector(
    execution_receipt: IterativeExecutionReceipt,
    *,
    version: str,
) -> InvariantDetectionReceipt:
    if not isinstance(execution_receipt, IterativeExecutionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    trace: IterativeExecutionTrace = execution_receipt.trace
    snapshots = trace.snapshots
    transitions = trace.transitions
    n = len(snapshots)

    unique_ids = {snapshot.state_id for snapshot in snapshots}
    repeated_state_score = 0.0 if n == 0 else _clamp01(1.0 - (len(unique_ids) / float(n)))

    repeated_patterns: list[InvariantPattern] = []
    if n > 0:
        counts: dict[str, int] = {}
        for snapshot in snapshots:
            counts[snapshot.state_id] = counts.get(snapshot.state_id, 0) + 1
        for state_id in sorted(k for k, count in counts.items() if count > 1):
            count = counts[state_id]
            repeated_patterns.append(
                InvariantPattern(
                    pattern_type="repeated_state",
                    key=state_id,
                    support=count,
                    confidence=(count - 1) / float(max(n - 1, 1)),
                )
            )

    fixed_point_score, fixed_point_pattern = _detect_fixed_point(snapshots, transitions)
    plateau_score, plateau_pattern = _detect_plateau(snapshots)
    oscillation_score, oscillation_pattern = _detect_oscillation(snapshots)

    invariant_pressure = _clamp01(
        0.3 * repeated_state_score + 0.3 * fixed_point_score + 0.2 * plateau_score + 0.2 * oscillation_score
    )

    signal = InvariantSignal(
        repeated_state_score=repeated_state_score,
        fixed_point_score=fixed_point_score,
        plateau_score=plateau_score,
        oscillation_score=oscillation_score,
        invariant_pressure=invariant_pressure,
    )
    decision = _dominant_decision(signal)

    ordered_patterns: list[InvariantPattern] = []
    if fixed_point_pattern is not None:
        ordered_patterns.append(fixed_point_pattern)
    ordered_patterns.extend(repeated_patterns)
    if plateau_pattern is not None:
        ordered_patterns.append(plateau_pattern)
    if oscillation_pattern is not None:
        ordered_patterns.append(oscillation_pattern)

    return InvariantDetectionReceipt(
        version=version,
        signal=signal,
        decision=decision,
        patterns=tuple(ordered_patterns),
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "GENERALIZED_INVARIANT_DETECTOR_VERSION",
    "InvariantSignal",
    "InvariantDecision",
    "InvariantPattern",
    "InvariantDetectionReceipt",
    "evaluate_generalized_invariant_detector",
]
