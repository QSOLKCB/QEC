"""v138.5.0 — Resonance Lock Diagnostic Kernel.

Deterministic, advisory-only resonance lock diagnostics for symbolic trajectory
analysis. This module is additive and does not couple into decoder behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Mapping

RELEASE_VERSION = "v138.5.0"
DIAGNOSTIC_KIND = "resonance_lock_diagnostic_kernel"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]
StateIdentifier = int | str



def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")



def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )



def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")



def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()



def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)



def _state_sort_key(state: StateIdentifier) -> tuple[int, str]:
    if isinstance(state, bool):
        raise ValueError("state_sequence entries must be int or str (bool is not allowed)")
    if isinstance(state, int):
        return (0, f"{state:+020d}")
    if isinstance(state, str):
        if state == "":
            raise ValueError("state_sequence entries must be non-empty when string")
        return (1, state)
    raise ValueError("state_sequence entries must be int or str")



def _state_token(state: StateIdentifier) -> str:
    if isinstance(state, int):
        return f"i:{state}"
    return f"s:{state}"



def _validate_numeric_sequence(values: tuple[float, ...], *, field_name: str) -> tuple[float, ...]:
    normalized: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[{idx}] must be a finite number")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"{field_name}[{idx}] must be finite")
        normalized.append(as_float)
    return tuple(normalized)



def _immutable_mapping(mapping: Mapping[str, _JSONValue]) -> Mapping[str, _JSONValue]:
    canonical = {key: _canonicalize_json(mapping[key]) for key in sorted(mapping.keys())}
    return types.MappingProxyType(canonical)


@dataclass(frozen=True)
class ResonanceDiagnosticPolicy:
    min_lock_span_length: int = 2
    drift_lock_threshold: float = 0.25
    min_span_lock_strength: float = 0.40
    meaningful_lock_threshold: float = 0.55
    strong_lock_threshold: float = 0.80
    single_attractor_threshold: float = 0.65
    multi_attractor_threshold: float = 0.75
    transient_recurrence_threshold: float = 0.45
    weak_confidence_threshold: float = 0.35

    def __post_init__(self) -> None:
        if self.min_lock_span_length < 2:
            raise ValueError("policy min_lock_span_length must be >= 2")
        for field_name in (
            "drift_lock_threshold",
            "min_span_lock_strength",
            "meaningful_lock_threshold",
            "strong_lock_threshold",
            "single_attractor_threshold",
            "multi_attractor_threshold",
            "transient_recurrence_threshold",
            "weak_confidence_threshold",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"policy {field_name} must be a number in [0, 1]")
            float_value = float(value)
            if not math.isfinite(float_value) or float_value < 0.0 or float_value > 1.0:
                raise ValueError(f"policy {field_name} must be in [0, 1]")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "min_lock_span_length": self.min_lock_span_length,
            "drift_lock_threshold": self.drift_lock_threshold,
            "min_span_lock_strength": self.min_span_lock_strength,
            "meaningful_lock_threshold": self.meaningful_lock_threshold,
            "strong_lock_threshold": self.strong_lock_threshold,
            "single_attractor_threshold": self.single_attractor_threshold,
            "multi_attractor_threshold": self.multi_attractor_threshold,
            "transient_recurrence_threshold": self.transient_recurrence_threshold,
            "weak_confidence_threshold": self.weak_confidence_threshold,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ResonanceLockInput:
    state_sequence: tuple[StateIdentifier, ...]
    drift_sequence: tuple[float, ...] | None = None
    phase_sequence: tuple[StateIdentifier | float, ...] | None = None
    policy: ResonanceDiagnosticPolicy = ResonanceDiagnosticPolicy()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "state_sequence": tuple(_state_token(s) for s in self.state_sequence),
            "drift_sequence": self.drift_sequence,
            "phase_sequence": None if self.phase_sequence is None else tuple(str(v) for v in self.phase_sequence),
            "policy": self.policy.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ResonanceLockSpan:
    start_index: int
    end_index: int
    span_length: int
    dominant_state: StateIdentifier
    state_repeat_ratio: float
    mean_drift: float
    lock_strength: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "span_length": self.span_length,
            "dominant_state": _state_token(self.dominant_state),
            "state_repeat_ratio": self.state_repeat_ratio,
            "mean_drift": self.mean_drift,
            "lock_strength": self.lock_strength,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ResonanceAttractorProfile:
    dominant_states: tuple[StateIdentifier, ...]
    occupancy_counts: Mapping[str, int]
    occupancy_weights: Mapping[str, float]
    lock_span_count: int
    attractor_mode: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "dominant_states": tuple(_state_token(s) for s in self.dominant_states),
            "occupancy_counts": dict(self.occupancy_counts),
            "occupancy_weights": dict(self.occupancy_weights),
            "lock_span_count": self.lock_span_count,
            "attractor_mode": self.attractor_mode,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ResonanceDiagnosticDecision:
    resonance_classification: str
    meaningful_lock_present: bool
    strongest_lock_span: ResonanceLockSpan | None
    recommendation_label: str
    caution_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "resonance_classification": self.resonance_classification,
            "meaningful_lock_present": self.meaningful_lock_present,
            "strongest_lock_span": None if self.strongest_lock_span is None else self.strongest_lock_span.to_dict(),
            "recommendation_label": self.recommendation_label,
            "caution_notes": self.caution_notes,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ResonanceLockDiagnosticReceipt:
    release_version: str
    diagnostic_kind: str
    input_summary: Mapping[str, _JSONValue]
    trajectory_length: int
    has_drift_sequence: bool
    has_phase_sequence: bool
    lock_spans: tuple[ResonanceLockSpan, ...]
    attractor_profile: ResonanceAttractorProfile
    resonance_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    advisory_only: bool
    decoder_core_modified: bool
    replay_identity: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "diagnostic_kind": self.diagnostic_kind,
            "input_summary": dict(self.input_summary),
            "trajectory_length": self.trajectory_length,
            "has_drift_sequence": self.has_drift_sequence,
            "has_phase_sequence": self.has_phase_sequence,
            "lock_spans": tuple(span.to_dict() for span in self.lock_spans),
            "attractor_profile": self.attractor_profile.to_dict(),
            "resonance_classification": self.resonance_classification,
            "recommendation": self.recommendation,
            "bounded_metrics": dict(self.bounded_metrics),
            "advisory_only": self.advisory_only,
            "decoder_core_modified": self.decoder_core_modified,
            "replay_identity": self.replay_identity,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("replay_identity")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self.replay_identity


def _normalize_input(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    drift_sequence: tuple[float, ...] | list[float] | None,
    phase_sequence: tuple[StateIdentifier | float, ...] | list[StateIdentifier | float] | None,
    policy: ResonanceDiagnosticPolicy,
) -> ResonanceLockInput:
    if isinstance(state_sequence, (str, bytes, bytearray)):
        raise ValueError("state_sequence must be an ordered sequence of state identifiers")
    normalized_states = tuple(state_sequence)
    if len(normalized_states) == 0:
        raise ValueError("state_sequence must be non-empty")
    normalized_states = tuple(((_state_sort_key(state), state)[1]) for state in normalized_states)

    normalized_drift: tuple[float, ...] | None = None
    if drift_sequence is not None:
        normalized_drift = _validate_numeric_sequence(tuple(drift_sequence), field_name="drift_sequence")
        if len(normalized_drift) not in (len(normalized_states), len(normalized_states) - 1):
            raise ValueError(
                "drift_sequence length must match state_sequence length or state_sequence length - 1"
            )

    normalized_phase: tuple[StateIdentifier | float, ...] | None = None
    if phase_sequence is not None:
        if isinstance(phase_sequence, (str, bytes, bytearray)):
            raise ValueError("phase_sequence must be a sequence aligned to state_sequence")
        normalized_phase = tuple(phase_sequence)
        if len(normalized_phase) != len(normalized_states):
            raise ValueError("phase_sequence length must match state_sequence length")
        validated: list[StateIdentifier | float] = []
        for idx, value in enumerate(normalized_phase):
            if isinstance(value, bool):
                raise ValueError(f"phase_sequence[{idx}] must be int, str, or finite float")
            if isinstance(value, (int, str)):
                if isinstance(value, str) and value == "":
                    raise ValueError(f"phase_sequence[{idx}] must be non-empty when string")
                validated.append(value)
                continue
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"phase_sequence[{idx}] must be finite")
                validated.append(float(value))
                continue
            raise ValueError(f"phase_sequence[{idx}] must be int, str, or finite float")
        normalized_phase = tuple(validated)

    return ResonanceLockInput(
        state_sequence=normalized_states,
        drift_sequence=normalized_drift,
        phase_sequence=normalized_phase,
        policy=policy,
    )


def _state_aligned_drift(states_len: int, drift_sequence: tuple[float, ...] | None) -> tuple[float, ...]:
    if drift_sequence is None:
        return tuple(0.0 for _ in range(states_len))
    if len(drift_sequence) == states_len:
        return drift_sequence
    if len(drift_sequence) == states_len - 1:
        return drift_sequence + (drift_sequence[-1],)
    raise ValueError("drift_sequence is not aligned")


def _dominant_state(states: tuple[StateIdentifier, ...]) -> tuple[StateIdentifier, int]:
    counts: dict[StateIdentifier, int] = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], _state_sort_key(item[0])))
    return ordered[0]


def _detect_lock_spans(inp: ResonanceLockInput) -> tuple[ResonanceLockSpan, ...]:
    states = inp.state_sequence
    drifts = _state_aligned_drift(len(states), inp.drift_sequence)

    candidate: list[bool] = []
    for idx, state in enumerate(states):
        persistence = idx > 0 and state == states[idx - 1]
        short_cycle = idx > 1 and state == states[idx - 2]
        low_drift = abs(drifts[idx]) <= inp.policy.drift_lock_threshold
        candidate.append(bool(persistence or short_cycle or low_drift))

    spans: list[ResonanceLockSpan] = []
    idx = 0
    while idx < len(states):
        if not candidate[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < len(states) and candidate[idx]:
            idx += 1
        run_end = idx - 1
        start = run_start - 1 if run_start > 0 else 0
        end = run_end
        if end - start + 1 < inp.policy.min_lock_span_length:
            continue
        segment_states = states[start : end + 1]
        segment_drifts = tuple(abs(v) for v in drifts[start : end + 1])
        dominant_state, dominant_count = _dominant_state(segment_states)
        repeat_ratio = dominant_count / float(len(segment_states))
        mean_drift = sum(segment_drifts) / float(len(segment_drifts))
        drift_suppression = 1.0 - _clamp01(mean_drift / max(inp.policy.drift_lock_threshold, 1e-12))
        lock_strength = _clamp01(0.65 * repeat_ratio + 0.35 * drift_suppression)
        if lock_strength < inp.policy.min_span_lock_strength:
            continue
        spans.append(
            ResonanceLockSpan(
                start_index=start,
                end_index=end,
                span_length=end - start + 1,
                dominant_state=dominant_state,
                state_repeat_ratio=_clamp01(repeat_ratio),
                mean_drift=mean_drift,
                lock_strength=lock_strength,
            )
        )

    return tuple(sorted(spans, key=lambda span: (span.start_index, span.end_index)))


def _derive_attractor_profile(
    inp: ResonanceLockInput,
    lock_spans: tuple[ResonanceLockSpan, ...],
    recurrence_score: float,
) -> ResonanceAttractorProfile:
    counts: dict[StateIdentifier, int] = {}
    for state in inp.state_sequence:
        counts[state] = counts.get(state, 0) + 1

    total = len(inp.state_sequence)
    ordered_states = tuple(sorted(counts.keys(), key=_state_sort_key))
    occupancy_counts = { _state_token(state): counts[state] for state in ordered_states }
    occupancy_weights = { _state_token(state): counts[state] / float(total) for state in ordered_states }

    max_count = max(counts.values())
    dominant_states = tuple(
        state for state in ordered_states if counts[state] == max_count
    )

    sorted_weights = sorted(occupancy_weights.values(), reverse=True)
    top1 = sorted_weights[0]
    top2 = sorted_weights[1] if len(sorted_weights) > 1 else 0.0

    if top1 >= inp.policy.single_attractor_threshold and len(lock_spans) > 0:
        attractor_mode = "single_attractor"
    elif top1 + top2 >= inp.policy.multi_attractor_threshold and len(lock_spans) > 0:
        attractor_mode = "multi_attractor"
    elif recurrence_score >= inp.policy.transient_recurrence_threshold:
        attractor_mode = "transient"
    else:
        attractor_mode = "dispersed"

    return ResonanceAttractorProfile(
        dominant_states=dominant_states,
        occupancy_counts=_immutable_mapping(occupancy_counts),
        occupancy_weights=_immutable_mapping(occupancy_weights),
        lock_span_count=len(lock_spans),
        attractor_mode=attractor_mode,
    )


def _cross_span_stability(lock_spans: tuple[ResonanceLockSpan, ...]) -> float:
    if len(lock_spans) == 0:
        return 0.0
    if len(lock_spans) == 1:
        return 1.0
    deltas = [
        abs(lock_spans[i].lock_strength - lock_spans[i - 1].lock_strength)
        for i in range(1, len(lock_spans))
    ]
    mean_delta = sum(deltas) / float(len(deltas))
    return _clamp01(1.0 - mean_delta)


def _metrics(
    inp: ResonanceLockInput,
    lock_spans: tuple[ResonanceLockSpan, ...],
) -> Mapping[str, float]:
    unique_states = len(set(inp.state_sequence))
    trajectory_length = len(inp.state_sequence)

    lock_strength_score = max((span.lock_strength for span in lock_spans), default=0.0)
    counts: dict[StateIdentifier, int] = {}
    for state in inp.state_sequence:
        counts[state] = counts.get(state, 0) + 1
    attractor_concentration_score = max(counts.values()) / float(trajectory_length)
    trajectory_recurrence_score = _clamp01(
        (trajectory_length - unique_states) / float(max(1, trajectory_length - 1))
    )

    if inp.drift_sequence is None:
        drift_suppression_score = 0.0
    else:
        aligned = _state_aligned_drift(trajectory_length, inp.drift_sequence)
        mean_abs_drift = sum(abs(v) for v in aligned) / float(trajectory_length)
        drift_suppression_score = _clamp01(
            1.0 - mean_abs_drift / max(inp.policy.drift_lock_threshold, 1e-12)
        )

    cross_span_stability_score = _cross_span_stability(lock_spans)

    bounded_resonance_confidence = _clamp01(
        (
            lock_strength_score
            + attractor_concentration_score
            + drift_suppression_score
            + trajectory_recurrence_score
            + cross_span_stability_score
        )
        / 5.0
    )

    return _immutable_mapping(
        {
            "lock_strength_score": _clamp01(lock_strength_score),
            "attractor_concentration_score": _clamp01(attractor_concentration_score),
            "drift_suppression_score": _clamp01(drift_suppression_score),
            "trajectory_recurrence_score": _clamp01(trajectory_recurrence_score),
            "cross_span_stability_score": _clamp01(cross_span_stability_score),
            "bounded_resonance_confidence": _clamp01(bounded_resonance_confidence),
        }
    )


def _classification(
    attractor_profile: ResonanceAttractorProfile,
    metrics: Mapping[str, float],
) -> str:
    lock_strength = float(metrics["lock_strength_score"])
    recurrence = float(metrics["trajectory_recurrence_score"])
    confidence = float(metrics["bounded_resonance_confidence"])

    if attractor_profile.attractor_mode == "single_attractor" and lock_strength >= 0.6:
        return "single_attractor_lock"
    if attractor_profile.attractor_mode == "multi_attractor" and lock_strength >= 0.55:
        return "multi_attractor_lock"
    if attractor_profile.attractor_mode == "transient" and recurrence >= 0.45:
        return "resonant_transient"
    if confidence >= 0.35:
        return "weak_lock_field"
    return "dispersed_field"


def _strongest_span(lock_spans: tuple[ResonanceLockSpan, ...]) -> ResonanceLockSpan | None:
    if len(lock_spans) == 0:
        return None
    ordered = sorted(
        lock_spans,
        key=lambda span: (
            -span.lock_strength,
            -span.span_length,
            span.start_index,
            _state_sort_key(span.dominant_state),
        ),
    )
    return ordered[0]


def _decision(
    inp: ResonanceLockInput,
    lock_spans: tuple[ResonanceLockSpan, ...],
    classification: str,
    metrics: Mapping[str, float],
) -> ResonanceDiagnosticDecision:
    strongest = _strongest_span(lock_spans)
    lock_strength = float(metrics["lock_strength_score"])
    confidence = float(metrics["bounded_resonance_confidence"])

    meaningful = strongest is not None and lock_strength >= inp.policy.meaningful_lock_threshold

    if meaningful and lock_strength >= inp.policy.strong_lock_threshold:
        recommendation = "strong_lock_detected"
    elif meaningful:
        recommendation = "localized_lock_detected"
    elif confidence >= inp.policy.weak_confidence_threshold:
        recommendation = "weak_resonance_detected"
    else:
        recommendation = "no_meaningful_lock"

    cautions: list[str] = []
    if strongest is None:
        cautions.append("no_contiguous_lock_span_detected")
    if lock_strength < inp.policy.meaningful_lock_threshold:
        cautions.append("lock_strength_below_meaningful_threshold")
    if classification in ("weak_lock_field", "dispersed_field"):
        cautions.append("classification_indicates_weak_or_dispersed_field")

    return ResonanceDiagnosticDecision(
        resonance_classification=classification,
        meaningful_lock_present=meaningful,
        strongest_lock_span=strongest,
        recommendation_label=recommendation,
        caution_notes=tuple(cautions),
    )


def run_resonance_lock_diagnostic(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    drift_sequence: tuple[float, ...] | list[float] | None = None,
    phase_sequence: tuple[StateIdentifier | float, ...] | list[StateIdentifier | float] | None = None,
    policy: ResonanceDiagnosticPolicy | None = None,
) -> ResonanceLockDiagnosticReceipt:
    """Run deterministic resonance lock diagnostics and emit a replay-safe receipt."""
    effective_policy = policy if policy is not None else ResonanceDiagnosticPolicy()
    normalized = _normalize_input(
        state_sequence=state_sequence,
        drift_sequence=drift_sequence,
        phase_sequence=phase_sequence,
        policy=effective_policy,
    )
    lock_spans = _detect_lock_spans(normalized)
    recurrence = _clamp01((len(normalized.state_sequence) - len(set(normalized.state_sequence))) / float(max(1, len(normalized.state_sequence) - 1)))
    attractor = _derive_attractor_profile(normalized, lock_spans, recurrence)
    metrics = _metrics(normalized, lock_spans)
    classification = _classification(attractor, metrics)
    decision = _decision(normalized, lock_spans, classification, metrics)

    input_summary = _immutable_mapping(
        {
            "state_sequence_hash": normalized.stable_hash(),
            "state_cardinality": len(set(normalized.state_sequence)),
            "drift_alignment": "none"
            if normalized.drift_sequence is None
            else ("state_aligned" if len(normalized.drift_sequence) == len(normalized.state_sequence) else "transition_aligned"),
            "phase_present": normalized.phase_sequence is not None,
            "policy_hash": normalized.policy.stable_hash(),
        }
    )

    proto = ResonanceLockDiagnosticReceipt(
        release_version=RELEASE_VERSION,
        diagnostic_kind=DIAGNOSTIC_KIND,
        input_summary=input_summary,
        trajectory_length=len(normalized.state_sequence),
        has_drift_sequence=normalized.drift_sequence is not None,
        has_phase_sequence=normalized.phase_sequence is not None,
        lock_spans=lock_spans,
        attractor_profile=attractor,
        resonance_classification=decision.resonance_classification,
        recommendation=decision.recommendation_label,
        bounded_metrics=metrics,
        advisory_only=True,
        decoder_core_modified=False,
        replay_identity="",
    )

    replay_identity = _sha256_hex(proto.to_hash_payload_dict())
    return ResonanceLockDiagnosticReceipt(
        release_version=proto.release_version,
        diagnostic_kind=proto.diagnostic_kind,
        input_summary=proto.input_summary,
        trajectory_length=proto.trajectory_length,
        has_drift_sequence=proto.has_drift_sequence,
        has_phase_sequence=proto.has_phase_sequence,
        lock_spans=proto.lock_spans,
        attractor_profile=proto.attractor_profile,
        resonance_classification=proto.resonance_classification,
        recommendation=proto.recommendation,
        bounded_metrics=proto.bounded_metrics,
        advisory_only=proto.advisory_only,
        decoder_core_modified=proto.decoder_core_modified,
        replay_identity=replay_identity,
    )
