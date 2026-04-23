"""v144.0 — Periodicity Structure Kernel (RDPK).

Deterministic Kasiski-style spacing analysis for periodic motif detection over
bounded QEC state traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from functools import reduce
import math

from qec.analysis.canonical_hashing import (
    canonical_bytes as to_canonical_bytes,
    canonical_json as to_canonical_json,
    sha256_hex,
)

MAX_TRACE_LENGTH = 4096
MIN_MOTIF_SIZE = 2
MAX_MOTIF_SIZE = 5
STRONG_PERIODIC_THRESHOLD = 0.3
_DECIMAL_12 = Decimal("0.000000000001")


def _round12(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("non-finite floats are not allowed")
    return float(Decimal(str(value)).quantize(_DECIMAL_12, rounding=ROUND_HALF_EVEN))


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _is_canonical_trace_token(token: str) -> bool:
    return token == token.strip() and token != ""


def _is_valid_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


@dataclass(frozen=True)
class PeriodicityCandidate:
    motif_signature: str
    gcd_period: int
    repetition_count: int
    spacing_variance: float
    confidence: float

    def __post_init__(self) -> None:
        if self.gcd_period < 2:
            raise ValueError("gcd_period must be >= 2")
        if self.repetition_count < 2:
            raise ValueError("repetition_count must be >= 2")
        if self.spacing_variance < 0.0:
            raise ValueError("spacing_variance must be >= 0")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be in [0,1]")

    def to_dict(self) -> dict[str, str | int | float | bool | None | tuple[object, ...] | dict[str, object]]:
        return {
            "motif_signature": self.motif_signature,
            "gcd_period": self.gcd_period,
            "repetition_count": self.repetition_count,
            "spacing_variance": _round12(self.spacing_variance),
            "confidence": _round12(self.confidence),
        }

    def to_canonical_json(self) -> str:
        return to_canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return to_canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PeriodicityReceipt:
    trace_length: int
    candidates: tuple[PeriodicityCandidate, ...]
    dominant_period: int | None
    dominant_confidence: float
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.trace_length < 1 or self.trace_length > MAX_TRACE_LENGTH:
            raise ValueError("trace_length out of bounds")
        if self.dominant_confidence < 0.0 or self.dominant_confidence > 1.0:
            raise ValueError("dominant_confidence must be in [0,1]")
        if self.classification not in {"aperiodic", "weak_periodic", "strong_periodic"}:
            raise ValueError("invalid classification")
        if not isinstance(self.stable_hash, str) or not _is_valid_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be 64-char hex string")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch with canonical payload")

    def to_dict(self) -> dict[str, str | int | float | bool | None | tuple[object, ...] | dict[str, object]]:
        return {
            "trace_length": self.trace_length,
            "candidates": tuple(candidate.to_dict() for candidate in self.candidates),
            "dominant_period": self.dominant_period,
            "dominant_confidence": _round12(self.dominant_confidence),
            "classification": self.classification,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return to_canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return to_canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        payload = {
            "trace_length": self.trace_length,
            "candidates": tuple(candidate.to_dict() for candidate in self.candidates),
            "dominant_period": self.dominant_period,
            "dominant_confidence": _round12(self.dominant_confidence),
            "classification": self.classification,
        }
        return sha256_hex(payload)


def _distance_variance(distances: tuple[int, ...]) -> float:
    mean_distance = sum(distances) / float(len(distances))
    return sum((distance - mean_distance) ** 2 for distance in distances) / float(len(distances))


def _normalized_variance(distances: tuple[int, ...], variance: float) -> float:
    mean_distance = sum(distances) / float(len(distances))
    denom = (mean_distance * mean_distance) if mean_distance > 0.0 else 1.0
    return _clamp01(variance / denom)


def _motif_signature(motif: tuple[str, ...]) -> str:
    return to_canonical_json(motif)


def detect_periodicity(trace: list[str]) -> PeriodicityReceipt:
    if not isinstance(trace, list):
        raise ValueError("trace must be a list of canonical strings")
    if not trace:
        raise ValueError("trace must be non-empty")
    if len(trace) > MAX_TRACE_LENGTH:
        raise ValueError(f"trace too large: {len(trace)} > {MAX_TRACE_LENGTH}")

    for item in trace:
        if not isinstance(item, str):
            raise ValueError("trace elements must be strings")
        if not _is_canonical_trace_token(item):
            raise ValueError("trace elements must be canonical non-empty strings")

    motif_occurrences: dict[tuple[str, ...], list[int]] = {}
    trace_len = len(trace)

    max_k = min(MAX_MOTIF_SIZE, trace_len)
    for k in range(MIN_MOTIF_SIZE, max_k + 1):
        window_count = trace_len - k + 1
        for idx in range(window_count):
            motif = tuple(trace[idx : idx + k])
            motif_occurrences.setdefault(motif, []).append(idx)

    candidates: list[PeriodicityCandidate] = []
    for motif in sorted(motif_occurrences.keys()):
        indices = motif_occurrences[motif]
        if len(indices) < 2:
            continue
        distances = tuple(indices[i + 1] - indices[i] for i in range(len(indices) - 1))
        repetition_count = len(indices)
        if repetition_count < 2:
            continue
        gcd_period = reduce(math.gcd, distances)
        if gcd_period < 2:
            continue

        variance = _distance_variance(distances)
        normalized_variance = _normalized_variance(distances, variance)
        confidence = _clamp01((repetition_count / float(trace_len)) * (1.0 - normalized_variance))

        candidates.append(
            PeriodicityCandidate(
                motif_signature=_motif_signature(motif),
                gcd_period=gcd_period,
                repetition_count=repetition_count,
                spacing_variance=_round12(variance),
                confidence=_round12(confidence),
            )
        )

    ordered_candidates = tuple(
        sorted(candidates, key=lambda candidate: (-candidate.confidence, candidate.motif_signature))
    )

    if not ordered_candidates:
        dominant_period: int | None = None
        dominant_confidence = 0.0
        classification = "aperiodic"
    else:
        dominant = ordered_candidates[0]
        dominant_period = dominant.gcd_period
        dominant_confidence = dominant.confidence
        classification = (
            "strong_periodic"
            if dominant_confidence >= STRONG_PERIODIC_THRESHOLD
            else "weak_periodic"
        )

    payload_wo_hash = {
        "trace_length": trace_len,
        "candidates": tuple(candidate.to_dict() for candidate in ordered_candidates),
        "dominant_period": dominant_period,
        "dominant_confidence": _round12(dominant_confidence),
        "classification": classification,
    }
    stable_hash = sha256_hex(payload_wo_hash)

    return PeriodicityReceipt(
        trace_length=trace_len,
        candidates=ordered_candidates,
        dominant_period=dominant_period,
        dominant_confidence=_round12(dominant_confidence),
        classification=classification,
        stable_hash=stable_hash,
    )


__all__ = [
    "MAX_TRACE_LENGTH",
    "STRONG_PERIODIC_THRESHOLD",
    "PeriodicityCandidate",
    "PeriodicityReceipt",
    "detect_periodicity",
]
