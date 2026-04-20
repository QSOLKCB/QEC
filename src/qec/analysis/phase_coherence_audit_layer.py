"""v138.5.1 — Phase-Coherence Audit Layer.

Deterministic, advisory-only phase coherence auditing for symbolic/numeric phase
observability. This module is additive and does not couple into decoder behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Mapping

RELEASE_VERSION = "v138.5.1"
AUDIT_KIND = "phase_coherence_audit_layer"

RESONANCE_RELEASE_VERSION = "v138.5.0"
RESONANCE_DIAGNOSTIC_KIND = "resonance_lock_diagnostic_kernel"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]
StateIdentifier = int | str
PhaseValue = int | str | float


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


def _immutable_mapping(mapping: Mapping[str, _JSONValue]) -> Mapping[str, _JSONValue]:
    canonical = {key: _canonicalize_json(mapping[key]) for key in sorted(mapping.keys())}
    return types.MappingProxyType(canonical)


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


def _phase_token(value: PhaseValue) -> str | float:
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, int):
        return f"i:{value}"
    return float(value)


def _phase_sort_key(phase: PhaseValue) -> tuple[int, str]:
    if isinstance(phase, bool):
        raise ValueError("phase values must not be bool")
    if isinstance(phase, int):
        return (0, f"{phase:+020d}")
    if isinstance(phase, float):
        return (1, format(phase, ".17g"))
    if isinstance(phase, str):
        if phase == "":
            raise ValueError("symbolic phase values must be non-empty")
        return (2, phase)
    raise ValueError("phase values must be int, str, or float")


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


def _state_aligned_drift(length: int, drift_sequence: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if drift_sequence is None:
        return None
    if len(drift_sequence) == length:
        return drift_sequence
    if len(drift_sequence) == length - 1:
        if len(drift_sequence) == 0:
            return (0.0,)
        return drift_sequence + (drift_sequence[-1],)
    raise ValueError("drift_sequence length must match state_sequence length or state_sequence length - 1")


@dataclass(frozen=True)
class PhaseCoherenceAuditPolicy:
    numeric_delta_threshold: float = 0.15
    min_window_length: int = 2
    coherence_presence_threshold: float = 0.50
    strong_coherence_threshold: float = 0.78
    localized_coherence_threshold: float = 0.55
    fragmented_break_threshold: float = 0.45

    def __post_init__(self) -> None:
        if self.min_window_length < 2:
            raise ValueError("policy min_window_length must be >= 2")
        for field_name in (
            "numeric_delta_threshold",
            "coherence_presence_threshold",
            "strong_coherence_threshold",
            "localized_coherence_threshold",
            "fragmented_break_threshold",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"policy {field_name} must be a finite number")
            float_value = float(value)
            if not math.isfinite(float_value):
                raise ValueError(f"policy {field_name} must be finite")
            if field_name != "numeric_delta_threshold" and (float_value < 0.0 or float_value > 1.0):
                raise ValueError(f"policy {field_name} must be in [0, 1]")
            if field_name == "numeric_delta_threshold" and float_value < 0.0:
                raise ValueError("policy numeric_delta_threshold must be >= 0")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "numeric_delta_threshold": self.numeric_delta_threshold,
            "min_window_length": self.min_window_length,
            "coherence_presence_threshold": self.coherence_presence_threshold,
            "strong_coherence_threshold": self.strong_coherence_threshold,
            "localized_coherence_threshold": self.localized_coherence_threshold,
            "fragmented_break_threshold": self.fragmented_break_threshold,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseCoherenceAuditInput:
    state_sequence: tuple[StateIdentifier, ...]
    phase_sequence: tuple[PhaseValue, ...]
    phase_mode: str
    drift_sequence: tuple[float, ...] | None = None
    resonance_source: Mapping[str, _JSONValue] | None = None
    policy: PhaseCoherenceAuditPolicy = PhaseCoherenceAuditPolicy()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "state_sequence": tuple(_state_token(value) for value in self.state_sequence),
            "phase_sequence": tuple(_phase_token(value) for value in self.phase_sequence),
            "phase_mode": self.phase_mode,
            "drift_sequence": self.drift_sequence,
            "resonance_source": None if self.resonance_source is None else dict(self.resonance_source),
            "policy": self.policy.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseCoherenceWindow:
    start_index: int
    end_index: int
    window_length: int
    dominant_phase: PhaseValue
    phase_stability_ratio: float
    mean_phase_delta: float
    coherence_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "window_length": self.window_length,
            "dominant_phase": _phase_token(self.dominant_phase),
            "phase_stability_ratio": self.phase_stability_ratio,
            "mean_phase_delta": self.mean_phase_delta,
            "coherence_score": self.coherence_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseBreakSpan:
    start_index: int
    end_index: int
    span_length: int
    break_severity: float
    break_reason: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "span_length": self.span_length,
            "break_severity": self.break_severity,
            "break_reason": self.break_reason,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseCoherenceAuditDecision:
    coherence_classification: str
    meaningful_coherence_present: bool
    strongest_coherence_window: PhaseCoherenceWindow | None
    lock_phase_alignment_interpretation: str
    recommendation_label: str
    caution_reason: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "coherence_classification": self.coherence_classification,
            "meaningful_coherence_present": self.meaningful_coherence_present,
            "strongest_coherence_window": None
            if self.strongest_coherence_window is None
            else self.strongest_coherence_window.to_dict(),
            "lock_phase_alignment_interpretation": self.lock_phase_alignment_interpretation,
            "recommendation_label": self.recommendation_label,
            "caution_reason": self.caution_reason,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseCoherenceAuditReceipt:
    release_version: str
    audit_kind: str
    input_summary: Mapping[str, _JSONValue]
    trajectory_length: int
    has_phase_sequence: bool
    has_drift_sequence: bool
    resonance_source_identity: str | None
    coherence_windows: tuple[PhaseCoherenceWindow, ...]
    phase_break_spans: tuple[PhaseBreakSpan, ...]
    coherence_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    advisory_only: bool
    decoder_core_modified: bool
    replay_identity: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "audit_kind": self.audit_kind,
            "input_summary": dict(self.input_summary),
            "trajectory_length": self.trajectory_length,
            "has_phase_sequence": self.has_phase_sequence,
            "has_drift_sequence": self.has_drift_sequence,
            "resonance_source_identity": self.resonance_source_identity,
            "coherence_windows": tuple(window.to_dict() for window in self.coherence_windows),
            "phase_break_spans": tuple(span.to_dict() for span in self.phase_break_spans),
            "coherence_classification": self.coherence_classification,
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
        return hashlib.sha256(_canonical_bytes(self.to_hash_payload_dict())).hexdigest()


def _normalize_resonance_receipt(resonance_receipt: Any, trajectory_length: int) -> Mapping[str, _JSONValue]:
    if resonance_receipt is None:
        raise ValueError("resonance_receipt must not be None")
    if hasattr(resonance_receipt, "to_dict") and callable(resonance_receipt.to_dict):
        payload_raw = resonance_receipt.to_dict()
    elif isinstance(resonance_receipt, Mapping):
        payload_raw = dict(resonance_receipt)
    else:
        raise ValueError("resonance_receipt must be a mapping or a receipt-like object")

    try:
        payload = _canonicalize_json(payload_raw)
    except ValueError as exc:
        raise ValueError(f"malformed resonance_receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("malformed resonance_receipt: payload must be a mapping")

    if payload.get("release_version") != RESONANCE_RELEASE_VERSION:
        raise ValueError("resonance_receipt release_version must be 'v138.5.0'")
    if payload.get("diagnostic_kind") != RESONANCE_DIAGNOSTIC_KIND:
        raise ValueError("resonance_receipt diagnostic_kind must be 'resonance_lock_diagnostic_kernel'")

    required_fields = (
        "trajectory_length",
        "resonance_classification",
        "attractor_profile",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
        "replay_identity",
    )
    for field_name in required_fields:
        if field_name not in payload:
            raise ValueError(f"malformed resonance_receipt: missing field '{field_name}'")
    if "ordered_lock_spans" not in payload and "lock_spans" not in payload:
        raise ValueError(
            "malformed resonance_receipt: missing lock spans field "
            "('ordered_lock_spans' or legacy 'lock_spans')"
        )

    if payload["advisory_only"] is not True:
        raise ValueError("resonance_receipt advisory_only must be True")
    if payload["decoder_core_modified"] is not False:
        raise ValueError("resonance_receipt decoder_core_modified must be False")

    if int(payload["trajectory_length"]) != trajectory_length:
        raise ValueError("resonance_receipt trajectory_length must match input trajectory length")

    spans_field = "ordered_lock_spans" if "ordered_lock_spans" in payload else "lock_spans"
    if not isinstance(payload[spans_field], tuple):
        raise ValueError("malformed resonance_receipt: ordered_lock_spans must be a sequence")
    for idx, span in enumerate(payload[spans_field]):
        if not isinstance(span, Mapping):
            raise ValueError(f"malformed resonance_receipt: ordered_lock_spans[{idx}] must be a mapping")
        if "start_index" not in span or "end_index" not in span:
            raise ValueError(f"malformed resonance_receipt: ordered_lock_spans[{idx}] missing indices")

    replay_identity = payload["replay_identity"]
    if not isinstance(replay_identity, str) or replay_identity == "":
        raise ValueError("malformed resonance_receipt: replay_identity must be a non-empty string")

    hash_payload = dict(payload)
    hash_payload.pop("replay_identity", None)
    expected_hash = hashlib.sha256(_canonical_bytes(hash_payload)).hexdigest()
    if expected_hash != replay_identity:
        raise ValueError("malformed resonance_receipt: replay_identity hash mismatch")

    if "ordered_lock_spans" not in payload and "lock_spans" in payload:
        payload["ordered_lock_spans"] = payload["lock_spans"]

    return types.MappingProxyType({key: payload[key] for key in sorted(payload.keys())})


def _normalize_input(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    phase_sequence: tuple[PhaseValue, ...] | list[PhaseValue],
    drift_sequence: tuple[float, ...] | list[float] | None,
    resonance_receipt: Any,
    policy: PhaseCoherenceAuditPolicy,
) -> PhaseCoherenceAuditInput:
    if isinstance(state_sequence, (str, bytes, bytearray)):
        raise ValueError("state_sequence must be an ordered sequence of state identifiers")
    normalized_states = tuple(state_sequence)
    if len(normalized_states) == 0:
        raise ValueError("state_sequence must be non-empty")
    for state in normalized_states:
        _state_sort_key(state)

    if isinstance(phase_sequence, (str, bytes, bytearray)):
        raise ValueError("phase_sequence must be an ordered sequence of phase values")
    normalized_phase = tuple(phase_sequence)
    if len(normalized_phase) == 0:
        raise ValueError("phase_sequence must be non-empty")
    if len(normalized_phase) != len(normalized_states):
        raise ValueError("phase_sequence length must match state_sequence length")

    numeric_candidate = True
    symbolic_candidate = True
    for idx, value in enumerate(normalized_phase):
        if isinstance(value, bool):
            raise ValueError(f"phase_sequence[{idx}] must be int, str, or finite float")
        if isinstance(value, (int, float)):
            as_float = float(value)
            if not math.isfinite(as_float):
                raise ValueError(f"phase_sequence[{idx}] must be finite")
        else:
            numeric_candidate = False

        if isinstance(value, str):
            if value == "":
                raise ValueError(f"phase_sequence[{idx}] must be non-empty when string")
        elif not isinstance(value, int):
            symbolic_candidate = False

    if numeric_candidate:
        phase_mode = "numeric"
        phase_values: tuple[PhaseValue, ...] = tuple(float(value) for value in normalized_phase)
    elif symbolic_candidate:
        phase_mode = "symbolic"
        phase_values = normalized_phase
    else:
        raise ValueError("phase_sequence must be consistently numeric or symbolic")

    normalized_drift: tuple[float, ...] | None = None
    if drift_sequence is not None:
        normalized_drift = _validate_numeric_sequence(tuple(drift_sequence), field_name="drift_sequence")
        normalized_drift = _state_aligned_drift(len(normalized_states), normalized_drift)

    normalized_resonance: Mapping[str, _JSONValue] | None = None
    if resonance_receipt is not None:
        normalized_resonance = _normalize_resonance_receipt(resonance_receipt, len(normalized_states))

    return PhaseCoherenceAuditInput(
        state_sequence=normalized_states,
        phase_sequence=phase_values,
        phase_mode=phase_mode,
        drift_sequence=normalized_drift,
        resonance_source=normalized_resonance,
        policy=policy,
    )


def _dominant_phase(phases: tuple[PhaseValue, ...]) -> tuple[PhaseValue, int]:
    counts: dict[PhaseValue, int] = {}
    for phase in phases:
        counts[phase] = counts.get(phase, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], _phase_sort_key(item[0])))
    return ordered[0]


def _coherent_transition(inp: PhaseCoherenceAuditInput, idx: int) -> tuple[bool, float, str | None]:
    prev_value = inp.phase_sequence[idx - 1]
    curr_value = inp.phase_sequence[idx]
    if inp.phase_mode == "numeric":
        delta = abs(float(curr_value) - float(prev_value))
        if delta <= inp.policy.numeric_delta_threshold:
            return (True, delta, None)
        return (False, delta, "numeric_phase_jump")

    stable = curr_value == prev_value
    recurrent = idx > 1 and curr_value == inp.phase_sequence[idx - 2]
    if stable or recurrent:
        return (True, 0.0, None)
    return (False, 1.0, "symbolic_phase_break")


def _detect_windows_and_breaks(
    inp: PhaseCoherenceAuditInput,
) -> tuple[tuple[PhaseCoherenceWindow, ...], tuple[PhaseBreakSpan, ...], tuple[float, ...], tuple[bool, ...]]:
    if len(inp.phase_sequence) == 1:
        return ((), (), (), ())

    deltas: list[float] = []
    coherent_flags: list[bool] = []
    break_reasons: list[str | None] = []
    for idx in range(1, len(inp.phase_sequence)):
        coherent, delta, reason = _coherent_transition(inp, idx)
        coherent_flags.append(coherent)
        deltas.append(delta)
        break_reasons.append(reason)

    windows: list[PhaseCoherenceWindow] = []
    transition_idx = 0
    while transition_idx < len(coherent_flags):
        if not coherent_flags[transition_idx]:
            transition_idx += 1
            continue
        run_start_transition = transition_idx
        while transition_idx < len(coherent_flags) and coherent_flags[transition_idx]:
            transition_idx += 1
        run_end_transition = transition_idx - 1
        start = run_start_transition
        end = run_end_transition + 1
        length = end - start + 1
        if length < inp.policy.min_window_length:
            continue

        phases = inp.phase_sequence[start : end + 1]
        dominant, dominant_count = _dominant_phase(phases)
        stability_ratio = dominant_count / float(length)
        if inp.phase_mode == "numeric":
            segment_deltas = tuple(deltas[run_start_transition : run_end_transition + 1])
            mean_delta = sum(segment_deltas) / float(len(segment_deltas))
            coherence_score = _clamp01(1.0 - mean_delta / max(inp.policy.numeric_delta_threshold, 1e-12))
        else:
            mean_delta = 0.0
            coherence_score = _clamp01(stability_ratio)

        windows.append(
            PhaseCoherenceWindow(
                start_index=start,
                end_index=end,
                window_length=length,
                dominant_phase=dominant,
                phase_stability_ratio=_clamp01(stability_ratio),
                mean_phase_delta=mean_delta,
                coherence_score=coherence_score,
            )
        )

    breaks: list[PhaseBreakSpan] = []
    transition_idx = 0
    while transition_idx < len(coherent_flags):
        if coherent_flags[transition_idx]:
            transition_idx += 1
            continue
        run_start_transition = transition_idx
        while transition_idx < len(coherent_flags) and not coherent_flags[transition_idx]:
            transition_idx += 1
        run_end_transition = transition_idx - 1
        start = run_start_transition
        end = run_end_transition + 1
        severities = deltas[run_start_transition : run_end_transition + 1]
        if inp.phase_mode == "numeric":
            norm = max(inp.policy.numeric_delta_threshold, 1e-12)
            severity = sum(_clamp01(value / norm) for value in severities) / float(len(severities))
            reason = "numeric_phase_jump"
        else:
            severity = 1.0
            reason = "symbolic_phase_break"
        if break_reasons[run_start_transition] is not None:
            reason = str(break_reasons[run_start_transition])

        breaks.append(
            PhaseBreakSpan(
                start_index=start,
                end_index=end,
                span_length=end - start + 1,
                break_severity=_clamp01(severity),
                break_reason=reason,
            )
        )

    return (
        tuple(sorted(windows, key=lambda window: (window.start_index, window.end_index))),
        tuple(sorted(breaks, key=lambda span: (span.start_index, span.end_index))),
        tuple(deltas),
        tuple(coherent_flags),
    )


def _extract_resonance_lock_spans(resonance_source: Mapping[str, _JSONValue] | None) -> tuple[tuple[int, int], ...]:
    if resonance_source is None:
        return ()
    spans_raw = resonance_source["ordered_lock_spans"]
    if not isinstance(spans_raw, tuple):
        return ()
    normalized: list[tuple[int, int]] = []
    for span in spans_raw:
        if not isinstance(span, Mapping):
            continue
        start = int(span["start_index"])
        end = int(span["end_index"])
        normalized.append((start, end))
    return tuple(sorted(normalized, key=lambda value: (value[0], value[1])))


def _interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if right < left:
        return 0
    return right - left + 1


def _metrics(
    inp: PhaseCoherenceAuditInput,
    windows: tuple[PhaseCoherenceWindow, ...],
    breaks: tuple[PhaseBreakSpan, ...],
) -> Mapping[str, float]:
    n = len(inp.phase_sequence)
    coherent_occupancy = sum(window.window_length for window in windows)
    phase_coherence_score = _clamp01(coherent_occupancy / float(n))

    if len(windows) == 0:
        concentration = 0.0
    else:
        concentration = max(window.window_length for window in windows) / float(n)

    if len(breaks) == 0:
        break_penalty = 0.0
    else:
        severity_mass = sum(span.break_severity * span.span_length for span in breaks)
        break_penalty = _clamp01(severity_mass / float(max(1, n)))

    if len(windows) <= 1:
        cross_stability = 1.0 if len(windows) == 1 else 0.0
    else:
        deltas = [
            abs(windows[idx].coherence_score - windows[idx - 1].coherence_score)
            for idx in range(1, len(windows))
        ]
        cross_stability = _clamp01(1.0 - (sum(deltas) / float(len(deltas))))

    lock_spans = _extract_resonance_lock_spans(inp.resonance_source)
    if len(lock_spans) == 0 or len(windows) == 0:
        lock_alignment = 0.0
    else:
        overlap_total = 0
        for window in windows:
            interval = (window.start_index, window.end_index)
            overlap_total += sum(_interval_overlap(interval, lock) for lock in lock_spans)
        lock_alignment = _clamp01(overlap_total / float(max(1, coherent_occupancy)))

    confidence = _clamp01(
        (phase_coherence_score + lock_alignment + concentration + (1.0 - break_penalty) + cross_stability) / 5.0
    )

    return _immutable_mapping(
        {
            "phase_coherence_score": _clamp01(phase_coherence_score),
            "phase_lock_alignment_score": _clamp01(lock_alignment),
            "coherence_window_concentration_score": _clamp01(concentration),
            "phase_break_penalty_score": _clamp01(break_penalty),
            "cross_audit_stability_score": _clamp01(cross_stability),
            "bounded_audit_confidence": _clamp01(confidence),
        }
    )


def _coherence_classification(inp: PhaseCoherenceAuditInput, metrics: Mapping[str, float]) -> str:
    coherence = float(metrics["phase_coherence_score"])
    penalty = float(metrics["phase_break_penalty_score"])
    confidence = float(metrics["bounded_audit_confidence"])

    if coherence >= inp.policy.strong_coherence_threshold and penalty <= 0.20:
        return "strong_phase_coherence"
    if coherence >= inp.policy.localized_coherence_threshold and confidence >= 0.50:
        return "localized_phase_coherence"
    if penalty >= inp.policy.fragmented_break_threshold:
        return "fragmented_phase_field"
    if coherence >= 0.30:
        return "weak_phase_structure"
    return "phase_incoherent"


def _strongest_window(windows: tuple[PhaseCoherenceWindow, ...]) -> PhaseCoherenceWindow | None:
    if len(windows) == 0:
        return None
    ordered = sorted(
        windows,
        key=lambda window: (
            -window.coherence_score,
            -window.window_length,
            window.start_index,
            _phase_sort_key(window.dominant_phase),
        ),
    )
    return ordered[0]


def _decision(
    inp: PhaseCoherenceAuditInput,
    classification: str,
    windows: tuple[PhaseCoherenceWindow, ...],
    metrics: Mapping[str, float],
) -> PhaseCoherenceAuditDecision:
    strongest = _strongest_window(windows)
    coherence = float(metrics["phase_coherence_score"])
    alignment = float(metrics["phase_lock_alignment_score"])
    penalty = float(metrics["phase_break_penalty_score"])

    meaningful = strongest is not None and coherence >= inp.policy.coherence_presence_threshold

    if inp.resonance_source is None:
        alignment_interpretation = "no_resonance_source"
    elif alignment >= 0.65:
        alignment_interpretation = "coherence_aligned_with_lock_spans"
    elif alignment >= 0.30:
        alignment_interpretation = "coherence_partially_aligned_with_lock_spans"
    else:
        alignment_interpretation = "coherence_not_aligned_with_lock_spans"

    if penalty >= 0.6:
        recommendation = "coherence_breaks_dominate"
    elif inp.resonance_source is not None and alignment >= 0.65:
        recommendation = "coherence_confirms_lock_structure"
    elif inp.resonance_source is not None and alignment >= 0.30:
        recommendation = "coherence_partially_supports_lock_structure"
    else:
        recommendation = "coherence_independent_of_lock_structure"

    caution = ""
    if strongest is None:
        caution = "no_coherence_window_detected"
    elif not meaningful:
        caution = "coherence_below_meaningful_threshold"
    elif classification in ("fragmented_phase_field", "phase_incoherent"):
        caution = "classification_indicates_fragmented_or_incoherent_phase_behavior"

    return PhaseCoherenceAuditDecision(
        coherence_classification=classification,
        meaningful_coherence_present=meaningful,
        strongest_coherence_window=strongest,
        lock_phase_alignment_interpretation=alignment_interpretation,
        recommendation_label=recommendation,
        caution_reason=caution,
    )


def run_phase_coherence_audit(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    phase_sequence: tuple[PhaseValue, ...] | list[PhaseValue],
    drift_sequence: tuple[float, ...] | list[float] | None = None,
    resonance_receipt: Any = None,
    policy: PhaseCoherenceAuditPolicy | None = None,
) -> PhaseCoherenceAuditReceipt:
    """Run deterministic phase coherence auditing and emit a replay-safe receipt."""
    effective_policy = policy if policy is not None else PhaseCoherenceAuditPolicy()
    normalized = _normalize_input(
        state_sequence=state_sequence,
        phase_sequence=phase_sequence,
        drift_sequence=drift_sequence,
        resonance_receipt=resonance_receipt,
        policy=effective_policy,
    )

    windows, breaks, _, _ = _detect_windows_and_breaks(normalized)
    metrics = _metrics(normalized, windows, breaks)
    classification = _coherence_classification(normalized, metrics)
    decision = _decision(normalized, classification, windows, metrics)

    resonance_identity = None
    if normalized.resonance_source is not None:
        resonance_identity = str(normalized.resonance_source["replay_identity"])

    input_summary = _immutable_mapping(
        {
            "input_hash": normalized.stable_hash(),
            "state_cardinality": len(set(normalized.state_sequence)),
            "phase_mode": normalized.phase_mode,
            "drift_alignment": "none"
            if normalized.drift_sequence is None
            else (
                "state_aligned"
                if len(normalized.drift_sequence) == len(normalized.state_sequence)
                else "transition_aligned"
            ),
            "policy_hash": normalized.policy.stable_hash(),
            "resonance_source_identity": resonance_identity,
            "resonance_binding_hash": None
            if normalized.resonance_source is None
            else _sha256_hex(
                {
                    "source_identity": resonance_identity,
                    "source_classification": normalized.resonance_source["resonance_classification"],
                    "source_lock_spans": normalized.resonance_source["ordered_lock_spans"],
                    "source_bounded_metrics": normalized.resonance_source["bounded_metrics"],
                }
            ),
        }
    )

    proto = PhaseCoherenceAuditReceipt(
        release_version=RELEASE_VERSION,
        audit_kind=AUDIT_KIND,
        input_summary=input_summary,
        trajectory_length=len(normalized.state_sequence),
        has_phase_sequence=True,
        has_drift_sequence=normalized.drift_sequence is not None,
        resonance_source_identity=resonance_identity,
        coherence_windows=windows,
        phase_break_spans=breaks,
        coherence_classification=decision.coherence_classification,
        recommendation=decision.recommendation_label,
        bounded_metrics=metrics,
        advisory_only=True,
        decoder_core_modified=False,
        replay_identity="",
    )
    replay_identity = _sha256_hex(proto.to_hash_payload_dict())

    return PhaseCoherenceAuditReceipt(
        release_version=proto.release_version,
        audit_kind=proto.audit_kind,
        input_summary=proto.input_summary,
        trajectory_length=proto.trajectory_length,
        has_phase_sequence=proto.has_phase_sequence,
        has_drift_sequence=proto.has_drift_sequence,
        resonance_source_identity=proto.resonance_source_identity,
        coherence_windows=proto.coherence_windows,
        phase_break_spans=proto.phase_break_spans,
        coherence_classification=proto.coherence_classification,
        recommendation=proto.recommendation,
        bounded_metrics=proto.bounded_metrics,
        advisory_only=proto.advisory_only,
        decoder_core_modified=proto.decoder_core_modified,
        replay_identity=replay_identity,
    )
