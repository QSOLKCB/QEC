"""v138.5.2 — E8 Topology Projection Experiment.

Deterministic, advisory-only E8-inspired projection for symbolic topology
experiments. This module emits replay-safe artifacts and remains strictly
experimental; it is not a physical or algebraic proof system.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Mapping

RELEASE_VERSION = "v138.5.2"
EXPERIMENT_KIND = "e8_topology_projection_experiment"

RESONANCE_RELEASE_VERSION = "v138.5.0"
RESONANCE_DIAGNOSTIC_KIND = "resonance_lock_diagnostic_kernel"
PHASE_RELEASE_VERSION = "v138.5.1"
PHASE_AUDIT_KIND = "phase_coherence_audit_layer"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]
StateIdentifier = int | str
PhaseValue = int | str | float

_COORDINATE_LABELS = (
    "c1_recurrence_axis",
    "c2_lock_axis",
    "c3_attractor_axis",
    "c4_phase_coherence_axis",
    "c5_break_suppression_axis",
    "c6_alignment_axis",
    "c7_dispersion_axis",
    "c8_balance_axis",
)


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


def _validate_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _normalize_replay_bound_receipt(payload_raw: Any, *, source_name: str) -> Mapping[str, _JSONValue]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    elif isinstance(payload_raw, Mapping):
        payload_raw = dict(payload_raw)
    else:
        raise ValueError(f"{source_name} must be a mapping or a receipt-like object")

    try:
        payload = _canonicalize_json(payload_raw)
    except ValueError as exc:
        raise ValueError(f"malformed {source_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"malformed {source_name}: payload must be a mapping")

    if "replay_identity" not in payload:
        raise ValueError(f"malformed {source_name}: missing field 'replay_identity'")
    replay_identity = payload["replay_identity"]
    if not isinstance(replay_identity, str) or replay_identity == "":
        raise ValueError(f"malformed {source_name}: replay_identity must be a non-empty string")

    hash_payload = dict(payload)
    hash_payload.pop("replay_identity", None)
    expected_hash = _sha256_hex(hash_payload)
    if expected_hash != replay_identity:
        raise ValueError(f"malformed {source_name}: replay_identity hash mismatch")

    return types.MappingProxyType({key: payload[key] for key in sorted(payload.keys())})


def _normalize_resonance_receipt(resonance_receipt: Any, trajectory_length: int) -> Mapping[str, _JSONValue]:
    payload = _normalize_replay_bound_receipt(resonance_receipt, source_name="resonance_receipt")
    if payload.get("release_version") != RESONANCE_RELEASE_VERSION:
        raise ValueError("resonance_receipt release_version must be 'v138.5.0'")
    if payload.get("diagnostic_kind") != RESONANCE_DIAGNOSTIC_KIND:
        raise ValueError("resonance_receipt diagnostic_kind must be 'resonance_lock_diagnostic_kernel'")

    required_fields = (
        "trajectory_length",
        "resonance_classification",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    )
    for field_name in required_fields:
        if field_name not in payload:
            raise ValueError(f"malformed resonance_receipt: missing field '{field_name}'")
    if "ordered_lock_spans" not in payload and "lock_spans" not in payload:
        raise ValueError(
            "malformed resonance_receipt: missing lock spans field ('ordered_lock_spans' or legacy 'lock_spans')"
        )

    if int(payload["trajectory_length"]) != trajectory_length:
        raise ValueError("resonance_receipt trajectory_length must match input trajectory length")
    if payload["advisory_only"] is not True:
        raise ValueError("resonance_receipt advisory_only must be True")
    if payload["decoder_core_modified"] is not False:
        raise ValueError("resonance_receipt decoder_core_modified must be False")

    return payload


def _normalize_phase_audit_receipt(phase_audit_receipt: Any, trajectory_length: int) -> Mapping[str, _JSONValue]:
    payload = _normalize_replay_bound_receipt(phase_audit_receipt, source_name="phase_audit_receipt")
    if payload.get("release_version") != PHASE_RELEASE_VERSION:
        raise ValueError("phase_audit_receipt release_version must be 'v138.5.1'")
    if payload.get("audit_kind") != PHASE_AUDIT_KIND:
        raise ValueError("phase_audit_receipt audit_kind must be 'phase_coherence_audit_layer'")

    required_fields = (
        "trajectory_length",
        "coherence_classification",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    )
    for field_name in required_fields:
        if field_name not in payload:
            raise ValueError(f"malformed phase_audit_receipt: missing field '{field_name}'")
    if "coherence_windows" not in payload:
        raise ValueError("malformed phase_audit_receipt: missing field 'coherence_windows'")

    if int(payload["trajectory_length"]) != trajectory_length:
        raise ValueError("phase_audit_receipt trajectory_length must match input trajectory length")
    if payload["advisory_only"] is not True:
        raise ValueError("phase_audit_receipt advisory_only must be True")
    if payload["decoder_core_modified"] is not False:
        raise ValueError("phase_audit_receipt decoder_core_modified must be False")

    return payload


@dataclass(frozen=True)
class E8TopologyProjectionPolicy:
    concentrated_threshold: float = 0.68
    balanced_threshold: float = 0.84
    dispersed_threshold: float = 0.58
    informative_confidence_threshold: float = 0.55
    weak_informative_threshold: float = 0.35

    def __post_init__(self) -> None:
        for field_name in (
            "concentrated_threshold",
            "balanced_threshold",
            "dispersed_threshold",
            "informative_confidence_threshold",
            "weak_informative_threshold",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"policy {field_name} must be a number in [0, 1]")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
                raise ValueError(f"policy {field_name} must be in [0, 1]")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "concentrated_threshold": self.concentrated_threshold,
            "balanced_threshold": self.balanced_threshold,
            "dispersed_threshold": self.dispersed_threshold,
            "informative_confidence_threshold": self.informative_confidence_threshold,
            "weak_informative_threshold": self.weak_informative_threshold,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8TopologyProjectionInput:
    state_sequence: tuple[StateIdentifier, ...]
    phase_sequence: tuple[PhaseValue, ...] | None
    phase_mode: str
    resonance_source: Mapping[str, _JSONValue] | None
    phase_audit_source: Mapping[str, _JSONValue] | None
    policy: E8TopologyProjectionPolicy

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "state_sequence": tuple(_state_token(value) for value in self.state_sequence),
            "phase_sequence": None if self.phase_sequence is None else tuple(_phase_token(v) for v in self.phase_sequence),
            "phase_mode": self.phase_mode,
            "resonance_source": None if self.resonance_source is None else dict(self.resonance_source),
            "phase_audit_source": None if self.phase_audit_source is None else dict(self.phase_audit_source),
            "policy": self.policy.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8ProjectionCoordinate:
    index: int
    label: str
    value: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {"index": self.index, "label": self.label, "value": self.value}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8TopologyFeatureProfile:
    feature_values: Mapping[str, float]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {"feature_values": dict(self.feature_values)}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8TopologyProjectionDecision:
    topology_classification: str
    structurally_informative: bool
    dominant_coordinate_label: str
    source_agreement_interpretation: str
    recommendation_label: str
    caution_text: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "topology_classification": self.topology_classification,
            "structurally_informative": self.structurally_informative,
            "dominant_coordinate_label": self.dominant_coordinate_label,
            "source_agreement_interpretation": self.source_agreement_interpretation,
            "recommendation_label": self.recommendation_label,
            "caution_text": self.caution_text,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8TopologyProjectionReceipt:
    release_version: str
    experiment_kind: str
    input_summary: Mapping[str, _JSONValue]
    trajectory_length: int
    resonance_source_identity: str | None
    phase_audit_source_identity: str | None
    ordered_coordinates: tuple[E8ProjectionCoordinate, ...]
    feature_profile: E8TopologyFeatureProfile
    topology_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    advisory_only: bool
    decoder_core_modified: bool
    replay_identity: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "experiment_kind": self.experiment_kind,
            "input_summary": dict(self.input_summary),
            "trajectory_length": self.trajectory_length,
            "resonance_source_identity": self.resonance_source_identity,
            "phase_audit_source_identity": self.phase_audit_source_identity,
            "ordered_coordinates": tuple(c.to_dict() for c in self.ordered_coordinates),
            "feature_profile": self.feature_profile.to_dict(),
            "topology_classification": self.topology_classification,
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
        return _sha256_hex(self.to_hash_payload_dict())


def _normalize_input(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    phase_sequence: tuple[PhaseValue, ...] | list[PhaseValue] | None,
    resonance_receipt: Any,
    phase_audit_receipt: Any,
    policy: E8TopologyProjectionPolicy,
) -> E8TopologyProjectionInput:
    if isinstance(state_sequence, (str, bytes, bytearray)):
        raise ValueError("state_sequence must be an ordered sequence of state identifiers")
    normalized_states = tuple(state_sequence)
    if len(normalized_states) == 0:
        raise ValueError("state_sequence must be non-empty")
    for state in normalized_states:
        _state_sort_key(state)

    normalized_phase: tuple[PhaseValue, ...] | None = None
    phase_mode = "none"
    if phase_sequence is not None:
        if isinstance(phase_sequence, (str, bytes, bytearray)):
            raise ValueError("phase_sequence must be a sequence aligned to state_sequence")
        normalized_phase = tuple(phase_sequence)
        if len(normalized_phase) != len(normalized_states):
            raise ValueError("phase_sequence length must match state_sequence length")
        numeric_candidate = True
        symbolic_candidate = True
        validated: list[PhaseValue] = []
        for idx, value in enumerate(normalized_phase):
            if isinstance(value, bool):
                raise ValueError(f"phase_sequence[{idx}] must be int, str, or finite float")
            if isinstance(value, (int, float)):
                _validate_finite_number(value, f"phase_sequence[{idx}]")
            else:
                numeric_candidate = False

            if isinstance(value, str):
                if value == "":
                    raise ValueError(f"phase_sequence[{idx}] must be non-empty when string")
            elif not isinstance(value, int):
                symbolic_candidate = False
            validated.append(value)

        if numeric_candidate:
            phase_mode = "numeric"
            normalized_phase = tuple(float(v) for v in validated)
        elif symbolic_candidate:
            phase_mode = "symbolic"
            normalized_phase = tuple(validated)
        else:
            raise ValueError("phase_sequence must be consistently numeric or symbolic")

    normalized_resonance = None
    if resonance_receipt is not None:
        normalized_resonance = _normalize_resonance_receipt(resonance_receipt, len(normalized_states))

    normalized_phase_audit = None
    if phase_audit_receipt is not None:
        normalized_phase_audit = _normalize_phase_audit_receipt(phase_audit_receipt, len(normalized_states))

    return E8TopologyProjectionInput(
        state_sequence=normalized_states,
        phase_sequence=normalized_phase,
        phase_mode=phase_mode,
        resonance_source=normalized_resonance,
        phase_audit_source=normalized_phase_audit,
        policy=policy,
    )


def _derive_phase_features(inp: E8TopologyProjectionInput) -> tuple[float, float]:
    if inp.phase_sequence is None or len(inp.phase_sequence) <= 1:
        return (0.0, 1.0)

    if inp.phase_mode == "numeric":
        deltas = [abs(float(inp.phase_sequence[i]) - float(inp.phase_sequence[i - 1])) for i in range(1, len(inp.phase_sequence))]
        mean_delta = sum(deltas) / float(len(deltas))
        phase_coherence = _clamp01(1.0 - mean_delta)
        phase_break_penalty = _clamp01(sum(_clamp01(delta) for delta in deltas) / float(len(deltas)))
        return (phase_coherence, phase_break_penalty)

    coherent = 0
    for idx in range(1, len(inp.phase_sequence)):
        if inp.phase_sequence[idx] == inp.phase_sequence[idx - 1] or (
            idx > 1 and inp.phase_sequence[idx] == inp.phase_sequence[idx - 2]
        ):
            coherent += 1
    denom = float(max(1, len(inp.phase_sequence) - 1))
    phase_coherence = _clamp01(coherent / denom)
    return (phase_coherence, _clamp01(1.0 - phase_coherence))


def _extract_resonance_features(source: Mapping[str, _JSONValue] | None, n: int) -> tuple[float, float, float]:
    if source is None:
        return (0.0, 0.0, 0.0)

    spans_key = "ordered_lock_spans" if "ordered_lock_spans" in source else "lock_spans"
    spans = source[spans_key] if spans_key in source else ()
    lock_count = float(len(spans)) if isinstance(spans, tuple) else 0.0
    lock_count_score = _clamp01(lock_count / float(max(1, n // 2)))

    metrics = source.get("bounded_metrics")
    strongest_lock = 0.0
    attractor_concentration = 0.0
    if isinstance(metrics, Mapping):
        if "lock_strength_score" in metrics:
            strongest_lock = _clamp01(_validate_finite_number(metrics["lock_strength_score"], "resonance lock_strength_score"))
        if "attractor_concentration_score" in metrics:
            attractor_concentration = _clamp01(
                _validate_finite_number(metrics["attractor_concentration_score"], "resonance attractor_concentration_score")
            )
    return (lock_count_score, strongest_lock, attractor_concentration)


def _extract_phase_audit_features(source: Mapping[str, _JSONValue] | None) -> tuple[float, float, float]:
    if source is None:
        return (0.0, 1.0, 0.0)
    metrics = source.get("bounded_metrics")
    if not isinstance(metrics, Mapping):
        return (0.0, 1.0, 0.0)

    coherence = _clamp01(_validate_finite_number(metrics.get("phase_coherence_score", 0.0), "phase_coherence_score"))
    break_penalty = _clamp01(_validate_finite_number(metrics.get("phase_break_penalty_score", 1.0), "phase_break_penalty_score"))
    alignment = _clamp01(_validate_finite_number(metrics.get("phase_lock_alignment_score", 0.0), "phase_lock_alignment_score"))
    return (coherence, break_penalty, alignment)


def _dominant_state_occupancy(states: tuple[StateIdentifier, ...]) -> float:
    counts: dict[StateIdentifier, int] = {}
    for value in states:
        counts[value] = counts.get(value, 0) + 1
    dominant = max(counts.values())
    return _clamp01(dominant / float(len(states)))


def _transition_diversity(states: tuple[StateIdentifier, ...]) -> float:
    if len(states) <= 1:
        return 0.0
    transitions = [(states[idx - 1], states[idx]) for idx in range(1, len(states))]
    unique = len(set(transitions))
    return _clamp01(unique / float(len(transitions)))


def _feature_profile(inp: E8TopologyProjectionInput) -> E8TopologyFeatureProfile:
    n = len(inp.state_sequence)
    unique_states = len(set(inp.state_sequence))
    recurrence = _clamp01((n - unique_states) / float(max(1, n - 1)))
    state_cardinality = _clamp01(unique_states / float(n))
    transition_diversity = _transition_diversity(inp.state_sequence)
    dominant_occupancy = _dominant_state_occupancy(inp.state_sequence)

    lock_count_score, strongest_lock_score, attractor_concentration = _extract_resonance_features(inp.resonance_source, n)
    phase_coherence, phase_break_penalty = _derive_phase_features(inp)
    audit_coherence, audit_break_penalty, audit_alignment = _extract_phase_audit_features(inp.phase_audit_source)

    coherence_score = audit_coherence if inp.phase_audit_source is not None else phase_coherence
    break_penalty = audit_break_penalty if inp.phase_audit_source is not None else phase_break_penalty

    lock_phase_alignment = audit_alignment
    if inp.phase_audit_source is None and inp.resonance_source is not None and inp.phase_sequence is not None:
        lock_phase_alignment = _clamp01(0.5 * lock_count_score + 0.5 * coherence_score)

    values = {
        "state_cardinality_score": state_cardinality,
        "trajectory_recurrence_score": recurrence,
        "resonance_lock_count_score": lock_count_score,
        "strongest_lock_score": strongest_lock_score,
        "attractor_concentration_score": attractor_concentration if attractor_concentration > 0.0 else dominant_occupancy,
        "phase_coherence_score": coherence_score,
        "phase_break_penalty_score": break_penalty,
        "lock_phase_alignment_score": lock_phase_alignment,
        "transition_diversity_score": transition_diversity,
        "dominant_state_occupancy_score": dominant_occupancy,
    }
    return E8TopologyFeatureProfile(feature_values=_immutable_mapping(values))


def _project_coordinates(features: E8TopologyFeatureProfile) -> tuple[E8ProjectionCoordinate, ...]:
    fv = features.feature_values
    values = (
        _clamp01(float(fv["trajectory_recurrence_score"])),
        _clamp01(0.65 * float(fv["resonance_lock_count_score"]) + 0.35 * float(fv["strongest_lock_score"])),
        _clamp01(float(fv["attractor_concentration_score"])),
        _clamp01(float(fv["phase_coherence_score"])),
        _clamp01(1.0 - float(fv["phase_break_penalty_score"])),
        _clamp01(float(fv["lock_phase_alignment_score"])),
        _clamp01(0.65 * float(fv["transition_diversity_score"]) + 0.35 * (1.0 - float(fv["dominant_state_occupancy_score"]))),
        _clamp01(
            1.0
            - abs(float(fv["trajectory_recurrence_score"]) - float(fv["phase_coherence_score"]))
            - 0.5 * abs(float(fv["resonance_lock_count_score"]) - float(fv["phase_coherence_score"]))
        ),
    )
    return tuple(
        E8ProjectionCoordinate(index=idx + 1, label=_COORDINATE_LABELS[idx], value=_clamp01(values[idx]))
        for idx in range(8)
    )


def _dominant_coordinate(coordinates: tuple[E8ProjectionCoordinate, ...]) -> E8ProjectionCoordinate:
    dominant = coordinates[0]
    for coordinate in coordinates[1:]:
        if coordinate.value > dominant.value:
            dominant = coordinate
    return dominant


def _bounded_metrics(
    coordinates: tuple[E8ProjectionCoordinate, ...],
    features: E8TopologyFeatureProfile,
    inp: E8TopologyProjectionInput,
) -> Mapping[str, float]:
    values = tuple(c.value for c in coordinates)
    mean_value = sum(values) / 8.0
    concentration = _clamp01(max(values))

    mean_abs_dev = sum(abs(v - mean_value) for v in values) / 8.0
    symmetry_balance = _clamp01(1.0 - (mean_abs_dev / 0.5))

    variance = sum((v - mean_value) ** 2 for v in values) / 8.0
    topology_dispersion = _clamp01(variance / 0.25)

    fv = features.feature_values
    consistency_terms = (
        1.0 - abs(values[0] - float(fv["trajectory_recurrence_score"])),
        1.0 - abs(values[1] - (0.65 * float(fv["resonance_lock_count_score"]) + 0.35 * float(fv["strongest_lock_score"]))),
        1.0 - abs(values[3] - float(fv["phase_coherence_score"])),
        1.0 - abs(values[4] - (1.0 - float(fv["phase_break_penalty_score"]))),
        1.0 - abs(values[6] - (0.65 * float(fv["transition_diversity_score"]) + 0.35 * (1.0 - float(fv["dominant_state_occupancy_score"])))),
    )
    projection_consistency = _clamp01(sum(_clamp01(v) for v in consistency_terms) / float(len(consistency_terms)))

    resonance_evidence = _clamp01(values[1] * 0.7 + values[2] * 0.3)
    phase_evidence = _clamp01(values[3] * 0.4 + values[5] * 0.6)
    if inp.resonance_source is not None and inp.phase_audit_source is not None:
        cross_source_stability = _clamp01(
            0.5 * (1.0 - abs(resonance_evidence - phase_evidence)) + 0.5 * values[5]
        )
    elif inp.resonance_source is not None:
        cross_source_stability = _clamp01(0.35 + 0.65 * resonance_evidence)
    elif inp.phase_audit_source is not None:
        cross_source_stability = _clamp01(0.35 + 0.65 * phase_evidence)
    else:
        cross_source_stability = _clamp01(0.2 + 0.4 * ((resonance_evidence + phase_evidence) / 2.0))

    confidence = _clamp01(
        (
            projection_consistency
            + concentration
            + symmetry_balance
            + (1.0 - topology_dispersion)
            + cross_source_stability
        )
        / 5.0
    )

    return _immutable_mapping(
        {
            "projection_consistency_score": projection_consistency,
            "topology_concentration_score": concentration,
            "symmetry_balance_score": symmetry_balance,
            "topology_dispersion_score": topology_dispersion,
            "cross_source_stability_score": cross_source_stability,
            "bounded_projection_confidence": confidence,
        }
    )


def _classification(
    coordinates: tuple[E8ProjectionCoordinate, ...],
    metrics: Mapping[str, float],
    policy: E8TopologyProjectionPolicy,
) -> str:
    dominant = _dominant_coordinate(coordinates)
    concentration = float(metrics["topology_concentration_score"])
    balance = float(metrics["symmetry_balance_score"])
    dispersion = float(metrics["topology_dispersion_score"])

    if balance >= policy.balanced_threshold and dispersion <= 0.35:
        return "balanced_topology_field"
    if dominant.label == "c2_lock_axis" and dominant.value >= policy.concentrated_threshold:
        return "lock_dominant_projection"
    if dominant.label == "c4_phase_coherence_axis" and dominant.value >= policy.concentrated_threshold:
        return "phase_shaped_projection"
    if dispersion >= policy.dispersed_threshold:
        return "dispersed_topology_cloud"
    if concentration >= policy.concentrated_threshold:
        return "concentrated_topology_cluster"
    return "dispersed_topology_cloud"


def _decision(
    *,
    classification: str,
    coordinates: tuple[E8ProjectionCoordinate, ...],
    metrics: Mapping[str, float],
    inp: E8TopologyProjectionInput,
) -> E8TopologyProjectionDecision:
    dominant = _dominant_coordinate(coordinates)
    stability = float(metrics["cross_source_stability_score"])
    confidence = float(metrics["bounded_projection_confidence"])

    if inp.resonance_source is not None and inp.phase_audit_source is not None:
        if stability >= 0.75:
            source_agreement = "resonance_and_phase_sources_are_aligned"
            recommendation = "projection_supports_joint_structure"
        elif stability >= 0.45:
            source_agreement = "resonance_and_phase_sources_are_partially_aligned"
            recommendation = "projection_supports_joint_structure"
        else:
            source_agreement = "resonance_and_phase_sources_conflict"
            recommendation = "projection_is_weakly_informative"
    elif inp.resonance_source is not None:
        source_agreement = "resonance_source_only"
        recommendation = "projection_supports_resonance_structure"
    elif inp.phase_audit_source is not None:
        source_agreement = "phase_source_only"
        recommendation = "projection_supports_phase_structure"
    else:
        source_agreement = "no_external_sources"
        recommendation = "projection_is_weakly_informative"

    structurally_informative = confidence >= inp.policy.informative_confidence_threshold
    caution = ""
    if confidence < inp.policy.weak_informative_threshold:
        caution = "low_projection_confidence"
    elif classification == "dispersed_topology_cloud":
        caution = "projection_indicates_diffuse_topology"

    return E8TopologyProjectionDecision(
        topology_classification=classification,
        structurally_informative=structurally_informative,
        dominant_coordinate_label=dominant.label,
        source_agreement_interpretation=source_agreement,
        recommendation_label=recommendation,
        caution_text=caution,
    )


def run_e8_topology_projection_experiment(
    *,
    state_sequence: tuple[StateIdentifier, ...] | list[StateIdentifier],
    phase_sequence: tuple[PhaseValue, ...] | list[PhaseValue] | None = None,
    resonance_receipt: Any = None,
    phase_audit_receipt: Any = None,
    policy: E8TopologyProjectionPolicy | None = None,
) -> E8TopologyProjectionReceipt:
    """Run a deterministic E8-inspired symbolic topology experiment."""
    effective_policy = policy if policy is not None else E8TopologyProjectionPolicy()
    normalized = _normalize_input(
        state_sequence=state_sequence,
        phase_sequence=phase_sequence,
        resonance_receipt=resonance_receipt,
        phase_audit_receipt=phase_audit_receipt,
        policy=effective_policy,
    )

    features = _feature_profile(normalized)
    coordinates = _project_coordinates(features)
    metrics = _bounded_metrics(coordinates, features, normalized)
    classification = _classification(coordinates, metrics, effective_policy)
    decision = _decision(
        classification=classification,
        coordinates=coordinates,
        metrics=metrics,
        inp=normalized,
    )

    resonance_identity = None if normalized.resonance_source is None else str(normalized.resonance_source["replay_identity"])
    phase_identity = None if normalized.phase_audit_source is None else str(normalized.phase_audit_source["replay_identity"])

    input_summary = _immutable_mapping(
        {
            "input_hash": normalized.stable_hash(),
            "policy_hash": effective_policy.stable_hash(),
            "state_cardinality": len(set(normalized.state_sequence)),
            "phase_mode": normalized.phase_mode,
            "resonance_source_identity": resonance_identity,
            "phase_audit_source_identity": phase_identity,
            "source_binding_hash": _sha256_hex(
                {
                    "resonance_identity": resonance_identity,
                    "resonance_classification": None if normalized.resonance_source is None else normalized.resonance_source["resonance_classification"],
                    "resonance_lock_spans": None
                    if normalized.resonance_source is None
                    else normalized.resonance_source[
                        "ordered_lock_spans" if "ordered_lock_spans" in normalized.resonance_source else "lock_spans"
                    ],
                    "resonance_bounded_metrics": None if normalized.resonance_source is None else normalized.resonance_source["bounded_metrics"],
                    "phase_identity": phase_identity,
                    "phase_classification": None if normalized.phase_audit_source is None else normalized.phase_audit_source["coherence_classification"],
                    "phase_windows": None if normalized.phase_audit_source is None else normalized.phase_audit_source["coherence_windows"],
                    "phase_bounded_metrics": None if normalized.phase_audit_source is None else normalized.phase_audit_source["bounded_metrics"],
                }
            ),
        }
    )

    proto = E8TopologyProjectionReceipt(
        release_version=RELEASE_VERSION,
        experiment_kind=EXPERIMENT_KIND,
        input_summary=input_summary,
        trajectory_length=len(normalized.state_sequence),
        resonance_source_identity=resonance_identity,
        phase_audit_source_identity=phase_identity,
        ordered_coordinates=coordinates,
        feature_profile=features,
        topology_classification=decision.topology_classification,
        recommendation=decision.recommendation_label,
        bounded_metrics=metrics,
        advisory_only=True,
        decoder_core_modified=False,
        replay_identity="",
    )
    replay_identity = _sha256_hex(proto.to_hash_payload_dict())

    return E8TopologyProjectionReceipt(
        release_version=proto.release_version,
        experiment_kind=proto.experiment_kind,
        input_summary=proto.input_summary,
        trajectory_length=proto.trajectory_length,
        resonance_source_identity=proto.resonance_source_identity,
        phase_audit_source_identity=proto.phase_audit_source_identity,
        ordered_coordinates=proto.ordered_coordinates,
        feature_profile=proto.feature_profile,
        topology_classification=decision.topology_classification,
        recommendation=decision.recommendation_label,
        bounded_metrics=proto.bounded_metrics,
        advisory_only=proto.advisory_only,
        decoder_core_modified=proto.decoder_core_modified,
        replay_identity=replay_identity,
    )
