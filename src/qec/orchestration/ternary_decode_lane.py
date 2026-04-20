"""v138.4.0 — deterministic additive ternary decode lane.

This module provides a replay-safe, advisory-only ternary decode lane artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

_RELEASE_VERSION = "v138.4.0"
_LANE_KIND = "ternary_decode_lane"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_int(symbol: Any, *, field: str) -> int:
    if isinstance(symbol, bool) or not isinstance(symbol, int):
        raise ValueError(f"{field} must contain only integers")
    return int(symbol)


def _bounded(value: float, *, field: str) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    return float(value)


def _normalize_ternary_symbols(
    symbols: Sequence[int] | Iterable[int], *, allow_balanced_ternary: bool
) -> tuple[int, ...]:
    if not isinstance(symbols, Sequence):
        symbols = tuple(symbols)
    normalized: list[int] = []
    if len(symbols) == 0:
        raise ValueError("ternary_symbols must be non-empty")

    for index, raw_symbol in enumerate(symbols):
        symbol = _require_int(raw_symbol, field=f"ternary_symbols[{index}]")
        if symbol in (0, 1, 2):
            normalized.append(symbol)
            continue
        if allow_balanced_ternary and symbol in (-1,):
            normalized.append(2)
            continue
        if allow_balanced_ternary:
            raise ValueError("ternary_symbols must use canonical {0,1,2} or balanced {-1,0,1}")
        raise ValueError("ternary_symbols must use canonical basis {0,1,2}")
    return tuple(normalized)


def _validate_correction_shape(correction: tuple[int, ...], expected_len: int, *, field: str) -> tuple[int, ...]:
    if len(correction) != expected_len:
        raise ValueError(f"{field} must match canonical syndrome length")
    out: list[int] = []
    for index, value in enumerate(correction):
        symbol = _require_int(value, field=f"{field}[{index}]")
        if symbol not in (0, 1, 2):
            raise ValueError(f"{field}[{index}] must be one of 0, 1, 2")
        out.append(symbol)
    return tuple(out)


def _build_metrics(
    *,
    syndrome_len: int,
    correction_weight: int,
    mismatch_count: int,
    gf3_residual_sum: int,
) -> dict[str, float]:
    syndrome_match_score = _bounded(1.0 - (mismatch_count / syndrome_len), field="syndrome_match_score")
    correction_sparsity_score = _bounded(1.0 - (correction_weight / syndrome_len), field="correction_sparsity_score")
    gf3_consistency_score = _bounded(1.0 - (gf3_residual_sum / (2.0 * syndrome_len)), field="gf3_consistency_score")
    readiness_gap = abs(correction_weight - mismatch_count) / syndrome_len
    hardware_lane_readiness = _bounded(1.0 - readiness_gap, field="hardware_lane_readiness")
    bounded_confidence = _bounded(
        (syndrome_match_score + correction_sparsity_score + gf3_consistency_score + hardware_lane_readiness) / 4.0,
        field="bounded_confidence",
    )
    return {
        "syndrome_match_score": syndrome_match_score,
        "correction_sparsity_score": correction_sparsity_score,
        "gf3_consistency_score": gf3_consistency_score,
        "hardware_lane_readiness": hardware_lane_readiness,
        "bounded_confidence": bounded_confidence,
    }


def _compute_candidate(candidate_id: str, syndrome: tuple[int, ...], correction: tuple[int, ...]) -> "TernaryDecodeCandidate":
    normalized_correction = _validate_correction_shape(
        correction,
        len(syndrome),
        field=f"proposed_correction for candidate_id={candidate_id}",
    )
    residual = tuple((syndrome[i] + normalized_correction[i]) % 3 for i in range(len(syndrome)))
    mismatch_count = sum(1 for value in residual if value != 0)
    correction_weight = sum(1 for value in normalized_correction if value != 0)
    gf3_residual_sum = sum(residual)
    metrics = _build_metrics(
        syndrome_len=len(syndrome),
        correction_weight=correction_weight,
        mismatch_count=mismatch_count,
        gf3_residual_sum=gf3_residual_sum,
    )
    composite_score = _bounded(
        (0.35 * metrics["syndrome_match_score"])
        + (0.25 * metrics["gf3_consistency_score"])
        + (0.20 * metrics["correction_sparsity_score"])
        + (0.10 * metrics["hardware_lane_readiness"])
        + (0.10 * metrics["bounded_confidence"]),
        field="composite_score",
    )
    return TernaryDecodeCandidate(
        candidate_id=candidate_id,
        proposed_correction=normalized_correction,
        correction_weight=correction_weight,
        syndrome_mismatch_count=mismatch_count,
        metric_bundle=metrics,
        composite_score=composite_score,
    )


def _build_candidates(syndrome: tuple[int, ...]) -> tuple["TernaryDecodeCandidate", ...]:
    return (
        _compute_candidate("cand_cancel", syndrome, tuple((-value) % 3 for value in syndrome)),
        _compute_candidate("cand_identity", syndrome, tuple(0 for _ in syndrome)),
        _compute_candidate("cand_copy", syndrome, tuple(value % 3 for value in syndrome)),
        _compute_candidate("cand_shift_plus_one", syndrome, tuple((value + 1) % 3 for value in syndrome)),
    )


def _candidate_rank_key(candidate: "TernaryDecodeCandidate") -> tuple[float, int, int, str, tuple[int, ...]]:
    return (
        -candidate.composite_score,
        candidate.correction_weight,
        candidate.syndrome_mismatch_count,
        candidate.candidate_id,
        candidate.proposed_correction,
    )


def _rank_candidates(candidates: Sequence["TernaryDecodeCandidate"]) -> tuple["TernaryDecodeCandidate", ...]:
    return tuple(sorted(candidates, key=_candidate_rank_key))


@dataclass(frozen=True)
class TernaryDecodeInput:
    ternary_symbols: tuple[int, ...]
    allow_balanced_ternary: bool = False

    def __post_init__(self) -> None:
        canonical = _normalize_ternary_symbols(self.ternary_symbols, allow_balanced_ternary=self.allow_balanced_ternary)
        object.__setattr__(self, "ternary_symbols", canonical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ternary_symbols": self.ternary_symbols,
            "allow_balanced_ternary": bool(self.allow_balanced_ternary),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryDecodeCandidate:
    candidate_id: str
    proposed_correction: tuple[int, ...]
    correction_weight: int
    syndrome_mismatch_count: int
    metric_bundle: Mapping[str, float]
    composite_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        correction_raw = tuple(self.proposed_correction)
        correction = _validate_correction_shape(
            correction_raw,
            len(correction_raw),
            field="proposed_correction",
        )
        object.__setattr__(self, "proposed_correction", correction)

        if isinstance(self.correction_weight, bool) or not isinstance(self.correction_weight, int):
            raise ValueError("correction_weight must be an integer")
        if self.correction_weight < 0 or self.correction_weight > len(correction):
            raise ValueError("correction_weight out of valid range")
        if isinstance(self.syndrome_mismatch_count, bool) or not isinstance(self.syndrome_mismatch_count, int):
            raise ValueError("syndrome_mismatch_count must be an integer")
        if self.syndrome_mismatch_count < 0 or self.syndrome_mismatch_count > len(correction):
            raise ValueError("syndrome_mismatch_count out of valid range")

        metric_keys = (
            "syndrome_match_score",
            "correction_sparsity_score",
            "gf3_consistency_score",
            "hardware_lane_readiness",
            "bounded_confidence",
        )
        if not isinstance(self.metric_bundle, Mapping):
            raise ValueError("metric_bundle must be a mapping")
        normalized_metrics = {key: _bounded(float(self.metric_bundle[key]), field=key) for key in metric_keys}
        object.__setattr__(self, "metric_bundle", MappingProxyType(dict(normalized_metrics)))
        object.__setattr__(self, "composite_score", _bounded(float(self.composite_score), field="composite_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "proposed_correction": self.proposed_correction,
            "correction_weight": self.correction_weight,
            "syndrome_mismatch_count": self.syndrome_mismatch_count,
            "metric_bundle": dict(self.metric_bundle),
            "composite_score": self.composite_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryDecodeDecision:
    ranked_candidates: tuple[TernaryDecodeCandidate, ...]
    selected_candidate: TernaryDecodeCandidate

    def __post_init__(self) -> None:
        if len(self.ranked_candidates) == 0:
            raise ValueError("ranked_candidates must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_candidates": tuple(candidate.to_dict() for candidate in self.ranked_candidates),
            "selected_candidate": self.selected_candidate.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryDecodeLaneReceipt:
    release_version: str
    lane_kind: str
    input_symbol_count: int
    canonical_ternary_syndrome: tuple[int, ...]
    candidate_count: int
    selected_candidate_id: str
    selected_correction: tuple[int, ...]
    selected_metric_bundle: Mapping[str, float]
    selected_composite_score: float
    advisory_only: bool
    decoder_core_modified: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_version": self.release_version,
            "lane_kind": self.lane_kind,
            "input_symbol_count": self.input_symbol_count,
            "canonical_ternary_syndrome": self.canonical_ternary_syndrome,
            "candidate_count": self.candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_correction": self.selected_correction,
            "selected_metric_bundle": dict(self.selected_metric_bundle),
            "selected_composite_score": self.selected_composite_score,
            "advisory_only": bool(self.advisory_only),
            "decoder_core_modified": bool(self.decoder_core_modified),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


def run_ternary_decode_lane(
    ternary_symbols: Sequence[int] | Iterable[int],
    *,
    allow_balanced_ternary: bool = False,
) -> TernaryDecodeLaneReceipt:
    decode_input = TernaryDecodeInput(
        ternary_symbols=tuple(ternary_symbols),
        allow_balanced_ternary=allow_balanced_ternary,
    )
    candidates = _build_candidates(decode_input.ternary_symbols)
    for candidate in candidates:
        if len(candidate.proposed_correction) != len(decode_input.ternary_symbols):
            raise ValueError("candidate proposed_correction must match canonical syndrome length")

    ranked_candidates = _rank_candidates(candidates)
    selected = ranked_candidates[0]
    decision = TernaryDecodeDecision(ranked_candidates=ranked_candidates, selected_candidate=selected)
    _ = decision  # explicit artifact construction for deterministic lane phase.

    provisional = TernaryDecodeLaneReceipt(
        release_version=_RELEASE_VERSION,
        lane_kind=_LANE_KIND,
        input_symbol_count=len(decode_input.ternary_symbols),
        canonical_ternary_syndrome=decode_input.ternary_symbols,
        candidate_count=len(ranked_candidates),
        selected_candidate_id=selected.candidate_id,
        selected_correction=selected.proposed_correction,
        selected_metric_bundle=selected.metric_bundle,
        selected_composite_score=selected.composite_score,
        advisory_only=True,
        decoder_core_modified=False,
        receipt_hash="",
    )
    return TernaryDecodeLaneReceipt(
        release_version=provisional.release_version,
        lane_kind=provisional.lane_kind,
        input_symbol_count=provisional.input_symbol_count,
        canonical_ternary_syndrome=provisional.canonical_ternary_syndrome,
        candidate_count=provisional.candidate_count,
        selected_candidate_id=provisional.selected_candidate_id,
        selected_correction=provisional.selected_correction,
        selected_metric_bundle=provisional.selected_metric_bundle,
        selected_composite_score=provisional.selected_composite_score,
        advisory_only=provisional.advisory_only,
        decoder_core_modified=provisional.decoder_core_modified,
        receipt_hash=provisional.stable_hash(),
    )


__all__ = [
    "TernaryDecodeInput",
    "TernaryDecodeCandidate",
    "TernaryDecodeDecision",
    "TernaryDecodeLaneReceipt",
    "run_ternary_decode_lane",
]
