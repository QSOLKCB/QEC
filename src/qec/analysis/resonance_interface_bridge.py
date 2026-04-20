"""v138.5.4 — Resonance Interface Bridge.

Deterministic additive bridge that unifies v138.5.0–v138.5.3 analysis receipts
into one replay-safe canonical interface artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Mapping

RELEASE_VERSION = "v138.5.4"
BRIDGE_KIND = "resonance_interface_bridge"

RESONANCE_RELEASE_VERSION = "v138.5.0"
RESONANCE_DIAGNOSTIC_KIND = "resonance_lock_diagnostic_kernel"
PHASE_RELEASE_VERSION = "v138.5.1"
PHASE_AUDIT_KIND = "phase_coherence_audit_layer"
TOPOLOGY_RELEASE_VERSION = "v138.5.2"
TOPOLOGY_EXPERIMENT_KIND = "e8_topology_projection_experiment"
FRACTAL_RELEASE_VERSION = "v138.5.3"
FRACTAL_EXPERIMENT_KIND = "fractal_field_invariant_mapper"

SOURCE_ORDER = ("resonance", "phase", "topology", "fractal")

INTERFACE_CLASSIFICATIONS = {
    "strongly_unified_interface",
    "partially_unified_interface",
    "weakly_supported_interface",
    "conflicted_interface",
}
RECOMMENDATIONS = {
    "interface_ready_for_runtime_binding",
    "interface_ready_for_partial_runtime_binding",
    "interface_requires_additional_sources",
    "interface_conflict_requires_review",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


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
    return types.MappingProxyType({key: _deep_freeze_json(_canonicalize_json(mapping[key])) for key in sorted(mapping.keys())})


def _deep_freeze_json(value: _JSONValue) -> _JSONValue:
    if isinstance(value, dict):
        return types.MappingProxyType({key: _deep_freeze_json(value[key]) for key in sorted(value.keys())})
    if isinstance(value, tuple):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _normalize_receipt_mapping(payload_raw: Any, *, source_name: str) -> Mapping[str, _JSONValue]:
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
    return _immutable_mapping(payload)


def _validate_bounded_metrics(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


def _validate_replay_identity(payload: Mapping[str, _JSONValue], *, source_name: str) -> str:
    replay_identity = payload.get("replay_identity")
    if not isinstance(replay_identity, str) or replay_identity == "":
        raise ValueError(f"malformed {source_name}: replay_identity must be a non-empty string")
    hash_payload = dict(payload)
    hash_payload.pop("replay_identity", None)
    expected = _sha256_hex(hash_payload)
    if expected != replay_identity:
        raise ValueError(f"malformed {source_name}: replay_identity hash mismatch")
    return replay_identity


def _ordered_tuple_of_mappings(value: Any, *, field_name: str, required_keys: tuple[str, ...]) -> tuple[Mapping[str, _JSONValue], ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    normalized: list[Mapping[str, _JSONValue]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{idx}] must be a mapping")
        for key in required_keys:
            if key not in item:
                raise ValueError(f"{field_name}[{idx}] missing field {key!r}")
        normalized.append(types.MappingProxyType({k: item[k] for k in sorted(item.keys())}))
    return tuple(normalized)


@dataclass(frozen=True)
class ResonanceInterfaceInput:
    resonance_receipt: Any = None
    phase_audit_receipt: Any = None
    topology_projection_receipt: Any = None
    fractal_invariant_receipt: Any = None


@dataclass(frozen=True)
class InterfaceSourceSummary:
    source_name: str
    source_release_version: str
    source_kind: str
    source_identity: str
    trajectory_length: int
    primary_classification: str
    support_label: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_name": self.source_name,
            "source_release_version": self.source_release_version,
            "source_kind": self.source_kind,
            "source_identity": self.source_identity,
            "trajectory_length": self.trajectory_length,
            "primary_classification": self.primary_classification,
            "support_label": self.support_label,
        }


@dataclass(frozen=True)
class ResonanceInterfaceAgreement:
    source_agreement_interpretation: str
    structural_consistency: float
    behavioral_consistency: float
    embedding_consistency: float
    multiscale_consistency: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_agreement_interpretation": self.source_agreement_interpretation,
            "structural_consistency": self.structural_consistency,
            "behavioral_consistency": self.behavioral_consistency,
            "embedding_consistency": self.embedding_consistency,
            "multiscale_consistency": self.multiscale_consistency,
        }


@dataclass(frozen=True)
class ResonanceInterfaceDecision:
    interface_classification: str
    source_coverage_summary: str
    dominant_source_support: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "interface_classification": self.interface_classification,
            "source_coverage_summary": self.source_coverage_summary,
            "dominant_source_support": self.dominant_source_support,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class ResonanceInterfaceBridgeReceipt:
    release_version: str
    bridge_kind: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    ordered_source_summaries: tuple[InterfaceSourceSummary, ...]
    structure_summary: Mapping[str, _JSONValue]
    behavior_summary: Mapping[str, _JSONValue]
    embedding_summary: Mapping[str, _JSONValue]
    agreement_summary: Mapping[str, _JSONValue]
    interface_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    decision: Mapping[str, _JSONValue]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "bridge_kind": self.bridge_kind,
            "trajectory_length": self.trajectory_length,
            "source_presence_flags": dict(self.source_presence_flags),
            "ordered_source_summaries": tuple(item.to_dict() for item in self.ordered_source_summaries),
            "structure_summary": dict(self.structure_summary),
            "behavior_summary": dict(self.behavior_summary),
            "embedding_summary": dict(self.embedding_summary),
            "agreement_summary": dict(self.agreement_summary),
            "interface_classification": self.interface_classification,
            "recommendation": self.recommendation,
            "bounded_metrics": dict(self.bounded_metrics),
            "decision": dict(self.decision),
            "advisory_only": self.advisory_only,
            "decoder_core_modified": self.decoder_core_modified,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class _NormalizedSource:
    source_name: str
    release_version: str
    source_kind: str
    source_identity: str
    trajectory_length: int
    classification: str
    support_label: str
    payload: Mapping[str, _JSONValue]


def _normalize_resonance_source(resonance_receipt: Any) -> _NormalizedSource:
    payload = _normalize_receipt_mapping(resonance_receipt, source_name="resonance_receipt")
    if payload.get("release_version") != RESONANCE_RELEASE_VERSION:
        raise ValueError("resonance_receipt release_version must be 'v138.5.0'")
    if payload.get("diagnostic_kind") != RESONANCE_DIAGNOSTIC_KIND:
        raise ValueError("resonance_receipt diagnostic_kind must be 'resonance_lock_diagnostic_kernel'")
    if payload.get("advisory_only") is not True or payload.get("decoder_core_modified") is not False:
        raise ValueError("resonance_receipt must be advisory_only=True and decoder_core_modified=False")

    replay_identity = _validate_replay_identity(payload, source_name="resonance_receipt")
    metrics = _validate_bounded_metrics(payload.get("bounded_metrics"), field_name="resonance_receipt.bounded_metrics")

    lock_spans = payload.get("ordered_lock_spans", payload.get("lock_spans"))
    _ordered_tuple_of_mappings(lock_spans, field_name="resonance lock spans", required_keys=("start_index", "end_index", "lock_strength"))

    classification = payload.get("resonance_classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError("malformed resonance_receipt: resonance_classification must be a non-empty string")

    trajectory_length = payload.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ValueError("malformed resonance_receipt: trajectory_length must be a positive int")

    _ = metrics
    return _NormalizedSource(
        source_name="resonance",
        release_version=RESONANCE_RELEASE_VERSION,
        source_kind=RESONANCE_DIAGNOSTIC_KIND,
        source_identity=replay_identity,
        trajectory_length=trajectory_length,
        classification=classification,
        support_label="structure",
        payload=payload,
    )


def _normalize_phase_audit_source(phase_audit_receipt: Any) -> _NormalizedSource:
    payload = _normalize_receipt_mapping(phase_audit_receipt, source_name="phase_audit_receipt")
    if payload.get("release_version") != PHASE_RELEASE_VERSION:
        raise ValueError("phase_audit_receipt release_version must be 'v138.5.1'")
    if payload.get("audit_kind") != PHASE_AUDIT_KIND:
        raise ValueError("phase_audit_receipt audit_kind must be 'phase_coherence_audit_layer'")
    if payload.get("advisory_only") is not True or payload.get("decoder_core_modified") is not False:
        raise ValueError("phase_audit_receipt must be advisory_only=True and decoder_core_modified=False")

    replay_identity = _validate_replay_identity(payload, source_name="phase_audit_receipt")
    _validate_bounded_metrics(payload.get("bounded_metrics"), field_name="phase_audit_receipt.bounded_metrics")
    _ordered_tuple_of_mappings(
        payload.get("coherence_windows"),
        field_name="phase_audit_receipt.coherence_windows",
        required_keys=("start_index", "end_index", "coherence_score"),
    )

    phase_breaks = payload.get("phase_break_spans")
    if not isinstance(phase_breaks, tuple):
        raise ValueError("phase_audit_receipt.phase_break_spans must be a tuple")

    classification = payload.get("coherence_classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError("malformed phase_audit_receipt: coherence_classification must be a non-empty string")

    trajectory_length = payload.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ValueError("malformed phase_audit_receipt: trajectory_length must be a positive int")

    return _NormalizedSource(
        source_name="phase",
        release_version=PHASE_RELEASE_VERSION,
        source_kind=PHASE_AUDIT_KIND,
        source_identity=replay_identity,
        trajectory_length=trajectory_length,
        classification=classification,
        support_label="behavior",
        payload=payload,
    )


def _normalize_topology_source(topology_projection_receipt: Any) -> _NormalizedSource:
    payload = _normalize_receipt_mapping(topology_projection_receipt, source_name="topology_projection_receipt")
    if payload.get("release_version") != TOPOLOGY_RELEASE_VERSION:
        raise ValueError("topology_projection_receipt release_version must be 'v138.5.2'")
    if payload.get("experiment_kind") != TOPOLOGY_EXPERIMENT_KIND:
        raise ValueError("topology_projection_receipt experiment_kind must be 'e8_topology_projection_experiment'")
    if payload.get("advisory_only") is not True or payload.get("decoder_core_modified") is not False:
        raise ValueError("topology_projection_receipt must be advisory_only=True and decoder_core_modified=False")

    replay_identity = _validate_replay_identity(payload, source_name="topology_projection_receipt")
    _validate_bounded_metrics(payload.get("bounded_metrics"), field_name="topology_projection_receipt.bounded_metrics")
    _ordered_tuple_of_mappings(
        payload.get("ordered_coordinates"),
        field_name="topology_projection_receipt.ordered_coordinates",
        required_keys=("index", "label", "value"),
    )

    classification = payload.get("topology_classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError("malformed topology_projection_receipt: topology_classification must be a non-empty string")

    trajectory_length = payload.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ValueError("malformed topology_projection_receipt: trajectory_length must be a positive int")

    return _NormalizedSource(
        source_name="topology",
        release_version=TOPOLOGY_RELEASE_VERSION,
        source_kind=TOPOLOGY_EXPERIMENT_KIND,
        source_identity=replay_identity,
        trajectory_length=trajectory_length,
        classification=classification,
        support_label="embedding",
        payload=payload,
    )


def _normalize_fractal_source(fractal_invariant_receipt: Any) -> _NormalizedSource:
    payload = _normalize_receipt_mapping(fractal_invariant_receipt, source_name="fractal_invariant_receipt")
    if payload.get("release_version") != FRACTAL_RELEASE_VERSION:
        raise ValueError("fractal_invariant_receipt release_version must be 'v138.5.3'")
    if payload.get("experiment_kind") != FRACTAL_EXPERIMENT_KIND:
        raise ValueError("fractal_invariant_receipt experiment_kind must be 'fractal_field_invariant_mapper'")
    if payload.get("advisory_only") is not True or payload.get("decoder_core_modified") is not False:
        raise ValueError("fractal_invariant_receipt must be advisory_only=True and decoder_core_modified=False")

    input_content_hash = payload.get("input_content_hash")
    if not isinstance(input_content_hash, str) or input_content_hash == "":
        raise ValueError("malformed fractal_invariant_receipt: input_content_hash must be a non-empty string")

    _validate_bounded_metrics(
        payload.get("bounded_metric_bundle"),
        field_name="fractal_invariant_receipt.bounded_metric_bundle",
    )
    _ordered_tuple_of_mappings(
        payload.get("ordered_scale_profiles"),
        field_name="fractal_invariant_receipt.ordered_scale_profiles",
        required_keys=("scale_size", "window_count", "dominant_motif"),
    )
    _ordered_tuple_of_mappings(
        payload.get("ordered_invariant_motifs"),
        field_name="fractal_invariant_receipt.ordered_invariant_motifs",
        required_keys=("motif", "canonical_signature", "scale_sizes"),
    )

    classification = payload.get("classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError("malformed fractal_invariant_receipt: classification must be a non-empty string")

    trajectory_length = payload.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ValueError("malformed fractal_invariant_receipt: trajectory_length must be a positive int")

    return _NormalizedSource(
        source_name="fractal",
        release_version=FRACTAL_RELEASE_VERSION,
        source_kind=FRACTAL_EXPERIMENT_KIND,
        source_identity=input_content_hash,
        trajectory_length=trajectory_length,
        classification=classification,
        support_label="multiscale",
        payload=payload,
    )


def _classification_strength(source_name: str, classification: str) -> float:
    if source_name == "resonance":
        if classification in {"single_attractor_lock", "multi_attractor_lock"}:
            return 1.0
        if classification in {"resonant_transient", "weak_lock_field"}:
            return 0.6
        return 0.2
    if source_name == "phase":
        if classification == "strong_phase_coherence":
            return 1.0
        if classification == "localized_phase_coherence":
            return 0.75
        if classification == "weak_phase_structure":
            return 0.5
        return 0.2
    if source_name == "topology":
        if classification in {"lock_dominant_projection", "phase_shaped_projection", "balanced_topology_field"}:
            return 0.9
        if classification == "concentrated_topology_cluster":
            return 0.75
        return 0.25
    if source_name == "fractal":
        if classification == "stable_recursive_field":
            return 1.0
        if classification in {"cross_scale_balanced_field", "localized_invariant_cluster"}:
            return 0.75
        if classification == "weak_invariant_structure":
            return 0.45
        return 0.2
    return 0.0


def _precompute_interface_agreement(normalized: Mapping[str, _NormalizedSource]) -> ResonanceInterfaceAgreement:
    structural = 0.5
    if "resonance" in normalized and "fractal" in normalized:
        rs = _classification_strength("resonance", normalized["resonance"].classification)
        fs = _classification_strength("fractal", normalized["fractal"].classification)
        structural = _clamp01(0.5 + 0.5 * min(rs, fs))

    behavioral = 0.5
    if "phase" in normalized and "resonance" in normalized:
        ps = _classification_strength("phase", normalized["phase"].classification)
        rs = _classification_strength("resonance", normalized["resonance"].classification)
        behavioral = _clamp01(0.5 + 0.5 * min(ps, rs))

    embedding = 0.5
    if "topology" in normalized and "phase" in normalized:
        ts = _classification_strength("topology", normalized["topology"].classification)
        ps = _classification_strength("phase", normalized["phase"].classification)
        embedding = _clamp01(0.5 + 0.5 * min(ts, ps))

    multiscale = 0.5
    if "topology" in normalized and "fractal" in normalized:
        ts = _classification_strength("topology", normalized["topology"].classification)
        fs = _classification_strength("fractal", normalized["fractal"].classification)
        multiscale = _clamp01(0.5 + 0.5 * min(ts, fs))

    mean_consistency = (structural + behavioral + embedding + multiscale) / 4.0
    conflict_pairs = []
    if "resonance" in normalized and "fractal" in normalized:
        conflict_pairs.append(structural)
    if "phase" in normalized and "resonance" in normalized:
        conflict_pairs.append(behavioral)
    if "topology" in normalized and "phase" in normalized:
        conflict_pairs.append(embedding)
    if "topology" in normalized and "fractal" in normalized:
        conflict_pairs.append(multiscale)

    if conflict_pairs and min(conflict_pairs) <= 0.6 and mean_consistency < 0.7:
        interpretation = "cross_source_conflict_detected"
    elif mean_consistency >= 0.85:
        interpretation = "high_cross_source_agreement"
    elif mean_consistency >= 0.65:
        interpretation = "moderate_cross_source_agreement"
    elif mean_consistency >= 0.45:
        interpretation = "limited_cross_source_agreement"
    else:
        interpretation = "cross_source_conflict_detected"

    return ResonanceInterfaceAgreement(
        source_agreement_interpretation=interpretation,
        structural_consistency=structural,
        behavioral_consistency=behavioral,
        embedding_consistency=embedding,
        multiscale_consistency=multiscale,
    )


def _build_decision(
    *,
    source_count: int,
    metrics: Mapping[str, float],
    agreement: ResonanceInterfaceAgreement,
) -> ResonanceInterfaceDecision:
    cross = metrics["cross_source_consistency_score"]
    confidence = metrics["bounded_interface_confidence"]

    if source_count >= 2 and cross >= 0.85 and confidence >= 0.8:
        classification = "strongly_unified_interface"
        recommendation = "interface_ready_for_runtime_binding"
    elif source_count >= 2 and cross >= 0.65 and confidence >= 0.6:
        classification = "partially_unified_interface"
        recommendation = "interface_ready_for_partial_runtime_binding"
    elif cross < 0.45 or (
        source_count >= 2
        and agreement.source_agreement_interpretation == "cross_source_conflict_detected"
        and cross < 0.65
    ):
        classification = "conflicted_interface"
        recommendation = "interface_conflict_requires_review"
    else:
        classification = "weakly_supported_interface"
        recommendation = "interface_requires_additional_sources"

    cautions: list[str] = []
    if source_count < 2:
        cautions.append("insufficient_source_domain_coverage")
    if agreement.source_agreement_interpretation == "cross_source_conflict_detected":
        cautions.append("cross_source_conflict_detected")
    if metrics["interface_completeness_score"] < 1.0:
        cautions.append("incomplete_source_coverage")

    if classification in {"strongly_unified_interface", "partially_unified_interface"} and source_count < 2:
        raise ValueError("unified interface classifications require at least 2 valid source domains")

    if classification not in INTERFACE_CLASSIFICATIONS:
        raise ValueError("invalid interface classification")
    if recommendation not in RECOMMENDATIONS:
        raise ValueError("invalid recommendation")

    return ResonanceInterfaceDecision(
        interface_classification=classification,
        source_coverage_summary=f"{source_count}/4_source_domains",
        dominant_source_support=agreement.source_agreement_interpretation,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _source_flags(normalized: Mapping[str, _NormalizedSource]) -> Mapping[str, bool]:
    return types.MappingProxyType({name: name in normalized for name in SOURCE_ORDER})


def _source_summaries(normalized: Mapping[str, _NormalizedSource]) -> tuple[InterfaceSourceSummary, ...]:
    return tuple(
        InterfaceSourceSummary(
            source_name=name,
            source_release_version=normalized[name].release_version,
            source_kind=normalized[name].source_kind,
            source_identity=normalized[name].source_identity,
            trajectory_length=normalized[name].trajectory_length,
            primary_classification=normalized[name].classification,
            support_label=normalized[name].support_label,
        )
        for name in SOURCE_ORDER
        if name in normalized
    )


def _validate_cross_source_bindings(normalized: Mapping[str, _NormalizedSource]) -> None:
    lengths = {src.trajectory_length for src in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("all supplied source receipts must share trajectory_length")

    resonance_identity = normalized.get("resonance").source_identity if "resonance" in normalized else None
    phase_identity = normalized.get("phase").source_identity if "phase" in normalized else None

    if "phase" in normalized and resonance_identity is not None:
        linked = normalized["phase"].payload.get("resonance_source_identity")
        if linked is not None and linked != resonance_identity:
            raise ValueError("phase_audit_receipt resonance_source_identity does not match resonance receipt")

    if "topology" in normalized:
        payload = normalized["topology"].payload
        linked_resonance = payload.get("resonance_source_identity")
        linked_phase = payload.get("phase_audit_source_identity")
        if resonance_identity is not None and linked_resonance is not None and linked_resonance != resonance_identity:
            raise ValueError("topology_projection_receipt resonance_source_identity mismatch")
        if phase_identity is not None and linked_phase is not None and linked_phase != phase_identity:
            raise ValueError("topology_projection_receipt phase_audit_source_identity mismatch")


def build_resonance_interface_bridge(
    *,
    resonance_receipt: Any = None,
    phase_audit_receipt: Any = None,
    topology_projection_receipt: Any = None,
    fractal_invariant_receipt: Any = None,
) -> ResonanceInterfaceBridgeReceipt:
    """Build a deterministic canonical interface bridge receipt."""
    bridge_input = ResonanceInterfaceInput(
        resonance_receipt=resonance_receipt,
        phase_audit_receipt=phase_audit_receipt,
        topology_projection_receipt=topology_projection_receipt,
        fractal_invariant_receipt=fractal_invariant_receipt,
    )

    normalized: dict[str, _NormalizedSource] = {}
    if bridge_input.resonance_receipt is not None:
        normalized["resonance"] = _normalize_resonance_source(bridge_input.resonance_receipt)
    if bridge_input.phase_audit_receipt is not None:
        normalized["phase"] = _normalize_phase_audit_source(bridge_input.phase_audit_receipt)
    if bridge_input.topology_projection_receipt is not None:
        normalized["topology"] = _normalize_topology_source(bridge_input.topology_projection_receipt)
    if bridge_input.fractal_invariant_receipt is not None:
        normalized["fractal"] = _normalize_fractal_source(bridge_input.fractal_invariant_receipt)

    if not normalized:
        raise ValueError("at least one source receipt must be provided")

    _validate_cross_source_bindings(normalized)
    ordered_summaries = _source_summaries(normalized)
    if tuple(summary.source_name for summary in ordered_summaries) != tuple(
        name for name in SOURCE_ORDER if name in normalized
    ):
        raise ValueError("ordered source summaries must follow canonical source ordering")

    agreement = _precompute_interface_agreement(normalized)

    present_count = len(normalized)
    completeness = _clamp01(present_count / 4.0)
    structural_alignment = _clamp01(agreement.structural_consistency)
    behavioral_alignment = _clamp01(agreement.behavioral_consistency)
    embedding_alignment = _clamp01((agreement.embedding_consistency + agreement.multiscale_consistency) / 2.0)
    cross_source_consistency = _clamp01(
        (agreement.structural_consistency + agreement.behavioral_consistency + agreement.embedding_consistency + agreement.multiscale_consistency)
        / 4.0
    )
    penalty = 0.0
    if agreement.source_agreement_interpretation == "cross_source_conflict_detected":
        penalty = 0.25
    bounded_confidence = _clamp01(
        0.35 * cross_source_consistency
        + 0.35 * completeness
    resonance_lock_spans = ()
    strongest_lock = None
    if "resonance" in normalized:
        resonance_lock_spans = normalized["resonance"].payload.get(
            "ordered_lock_spans",
            normalized["resonance"].payload.get("lock_spans", ()),
        )
        if len(resonance_lock_spans) > 0:
            strongest_lock = max(
                resonance_lock_spans,
                key=lambda item: (
                    float(item.get("lock_strength", 0.0)),
                    -int(item.get("start_index", 0)),
                    -int(item.get("end_index", 0)),
                ),
            )

    structure_summary = _immutable_mapping(
        {
            "resonance_classification": None if "resonance" not in normalized else normalized["resonance"].classification,
            "lock_count": None if "resonance" not in normalized else len(resonance_lock_spans),
            "strongest_lock": strongest_lock,
            "embedding_alignment_score": embedding_alignment,
            "bounded_interface_confidence": bounded_confidence,
        }
    )

    for name, value in metrics.items():
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    decision = _build_decision(source_count=present_count, metrics=metrics, agreement=agreement)

    trajectory_length = next(iter(normalized.values())).trajectory_length

    structure_summary = _immutable_mapping(
        {
    def _validated_topology_coordinate_key(item: Mapping[str, Any], position: int) -> tuple[float, int]:
        """Validate a topology coordinate entry and return the ordering key."""
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Invalid topology ordered_coordinates entry at index {position}: expected a mapping"
            )
        if "value" not in item or "index" not in item:
            raise ValueError(
                f"Invalid topology ordered_coordinates entry at index {position}: missing 'value' or 'index'"
            )

        raw_value = item["value"]
        raw_index = item["index"]

        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"Invalid topology ordered_coordinates entry at index {position}: 'index' must be an integer"
            )

        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"Invalid topology ordered_coordinates entry at index {position}: 'value' must be numeric"
            )

        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid topology ordered_coordinates entry at index {position}: 'value' must be finite"
            )

        return (value, -raw_index)

    embedding_summary = _immutable_mapping(
        {
            "topology_classification": None if "topology" not in normalized else normalized["topology"].classification,
            "dominant_coordinate": None
            if "topology" not in normalized or len(normalized["topology"].payload["ordered_coordinates"]) == 0
            else max(
                enumerate(normalized["topology"].payload["ordered_coordinates"]),
                key=lambda entry: _validated_topology_coordinate_key(entry[1], entry[0]),
            )[1],
                        -int(span["start_index"]),
                        -int(span["end_index"]),
                    ),
                )
                if len(normalized["resonance"].payload.get("ordered_lock_spans", normalized["resonance"].payload.get("lock_spans", ()))) > 0
                else None
            ),
            "fractal_classification": None if "fractal" not in normalized else normalized["fractal"].classification,
            "dominant_invariant_motif": None
            if "fractal" not in normalized or len(normalized["fractal"].payload["ordered_invariant_motifs"]) == 0
            else normalized["fractal"].payload["ordered_invariant_motifs"][0].get("motif"),
        }
    )

    behavior_summary = _immutable_mapping(
        {
            "phase_coherence_classification": None if "phase" not in normalized else normalized["phase"].classification,
            "coherence_window_count": None if "phase" not in normalized else len(normalized["phase"].payload.get("coherence_windows", ())),
            "phase_break_count": None if "phase" not in normalized else len(normalized["phase"].payload.get("phase_break_spans", ())),
        }
    )

    embedding_summary = _immutable_mapping(
        {
            "topology_classification": None if "topology" not in normalized else normalized["topology"].classification,
            "dominant_coordinate": None
            if "topology" not in normalized or len(normalized["topology"].payload["ordered_coordinates"]) == 0
            else max(
                normalized["topology"].payload["ordered_coordinates"],
                key=lambda item: (float(item["value"]), -int(item["index"])),
            ),
        }
    )

    agreement_summary = _immutable_mapping(agreement.to_dict())

    decision_mapping = _immutable_mapping(decision.to_dict())

    return ResonanceInterfaceBridgeReceipt(
        release_version=RELEASE_VERSION,
        bridge_kind=BRIDGE_KIND,
        trajectory_length=trajectory_length,
        source_presence_flags=_source_flags(normalized),
        ordered_source_summaries=ordered_summaries,
        structure_summary=structure_summary,
        behavior_summary=behavior_summary,
        embedding_summary=embedding_summary,
        agreement_summary=agreement_summary,
        interface_classification=decision.interface_classification,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        decision=decision_mapping,
        advisory_only=True,
        decoder_core_modified=False,
    )
