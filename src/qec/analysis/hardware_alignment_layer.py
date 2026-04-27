"""v149.2 — Deterministic hardware alignment analysis layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
import math
from typing import Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

HARDWARE_ALIGNMENT_MODULE_VERSION: Final[str] = "v149.2"

_ALLOWED_RECOMMENDATIONS: Final[tuple[str, ...]] = (
    "MAINTAIN_POLICY",
    "TIGHTEN_POLICY",
    "RELAX_POLICY",
    "ESCALATE_GOVERNANCE",
    "REVIEW_MEMORY",
)
_ALLOWED_LATENCY_CLASSES: Final[tuple[str, ...]] = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
_ALLOWED_ALIGNMENT_STATUSES: Final[tuple[str, ...]] = (
    "ALIGNED",
    "DEGRADED_ALIGNMENT",
    "REPLAY_UNSUPPORTED",
    "CAPABILITY_MISMATCH",
    "PRIORITY_EXCEEDED",
    "UNSTABLE_HARDWARE",
)
_ALLOWED_STATUS_REASON: Final[dict[str, str]] = {
    "ALIGNED": "aligned",
    "DEGRADED_ALIGNMENT": "degraded_but_usable",
    "REPLAY_UNSUPPORTED": "replay_not_supported",
    "CAPABILITY_MISMATCH": "capability_missing",
    "PRIORITY_EXCEEDED": "priority_exceeded",
    "UNSTABLE_HARDWARE": "stability_below_tolerance",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round_public_metric(value: float) -> float:
    return float(round(float(value), 12))


def _require_canonical_token(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty canonical string")
    token = value.strip()
    if not token or token != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return token


def _require_sha256_hex(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid SHA-256 hex") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    return value


def _require_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    return number


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalize_unique_sorted_tokens(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be tuple")
    normalized = tuple(sorted(_require_canonical_token(v, name=name) for v in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _bounded_mapping_score(control_confidence: float, stability_rating: float, risk_tolerance: float) -> float:
    score = (
        float(control_confidence) * 0.4
        + float(stability_rating) * 0.4
        + (1.0 - float(risk_tolerance)) * 0.2
    )
    score = min(1.0, max(0.0, score))
    return _round_public_metric(score)


@dataclass(frozen=True)
class ControlSignalIntent:
    signal_id: str
    source_receipt_hash: str
    recommendation: str
    control_priority: int
    control_confidence: float
    required_capability: str
    risk_tolerance: float
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.signal_id, name="signal_id")
        _require_sha256_hex(self.source_receipt_hash, name="source_receipt_hash")
        if self.recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("recommendation must be a supported governance label")
        _require_non_negative_int(self.control_priority, name="control_priority")
        object.__setattr__(self, "control_confidence", _round_public_metric(_require_probability(self.control_confidence, name="control_confidence")))
        _require_canonical_token(self.required_capability, name="required_capability")
        object.__setattr__(self, "risk_tolerance", _round_public_metric(_require_probability(self.risk_tolerance, name="risk_tolerance")))

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "signal_id": self.signal_id,
            "source_receipt_hash": self.source_receipt_hash,
            "recommendation": self.recommendation,
            "control_priority": int(self.control_priority),
            "control_confidence": _round_public_metric(float(self.control_confidence)),
            "required_capability": self.required_capability,
            "risk_tolerance": _round_public_metric(float(self.risk_tolerance)),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class HardwareConstraintProfile:
    hardware_id: str
    capabilities: tuple[str, ...]
    max_control_priority: int
    stability_rating: float
    latency_class: str
    replay_supported: bool
    constraint_hash: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.hardware_id, name="hardware_id")
        normalized_capabilities = _normalize_unique_sorted_tokens(self.capabilities, name="capabilities")
        _require_non_negative_int(self.max_control_priority, name="max_control_priority")
        object.__setattr__(self, "stability_rating", _round_public_metric(_require_probability(self.stability_rating, name="stability_rating")))
        if self.latency_class not in _ALLOWED_LATENCY_CLASSES:
            raise ValueError("latency_class must be one of LOW|MEDIUM|HIGH|UNKNOWN")
        if not isinstance(self.replay_supported, bool):
            raise ValueError("replay_supported must be bool")
        _require_sha256_hex(self.constraint_hash, name="constraint_hash")
        object.__setattr__(self, "capabilities", normalized_capabilities)

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "hardware_id": self.hardware_id,
            "capabilities": self.capabilities,
            "max_control_priority": int(self.max_control_priority),
            "stability_rating": _round_public_metric(float(self.stability_rating)),
            "latency_class": self.latency_class,
            "replay_supported": self.replay_supported,
            "constraint_hash": self.constraint_hash,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class HardwareAlignmentDecision:
    signal_id: str
    hardware_id: str
    alignment_status: str
    mapping_score: float
    constraint_violations: tuple[str, ...]
    selected_capability: str
    decision_reason: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.signal_id, name="signal_id")
        _require_canonical_token(self.hardware_id, name="hardware_id")
        if self.alignment_status not in _ALLOWED_ALIGNMENT_STATUSES:
            raise ValueError("alignment_status is not supported")
        object.__setattr__(self, "mapping_score", _round_public_metric(_require_probability(self.mapping_score, name="mapping_score")))
        violations = _normalize_unique_sorted_tokens(self.constraint_violations, name="constraint_violations")
        if self.selected_capability != "NONE":
            _require_canonical_token(self.selected_capability, name="selected_capability")
        _require_canonical_token(self.decision_reason, name="decision_reason")

        expected_reason = _ALLOWED_STATUS_REASON[self.alignment_status]
        if self.decision_reason != expected_reason:
            raise ValueError("decision_reason must match alignment_status")
        if self.alignment_status == "ALIGNED" and violations:
            raise ValueError("ALIGNED decisions cannot include constraint violations")

        if self.alignment_status == "DEGRADED_ALIGNMENT":
            allowed = {"high_latency", "low_stability"}
            if not violations or any(v not in allowed for v in violations):
                raise ValueError("DEGRADED_ALIGNMENT requires high_latency and/or low_stability violations")
        elif self.alignment_status == "REPLAY_UNSUPPORTED" and violations != ("replay_not_supported",):
            raise ValueError("REPLAY_UNSUPPORTED must include replay_not_supported")
        elif self.alignment_status == "CAPABILITY_MISMATCH" and violations != ("capability_missing",):
            raise ValueError("CAPABILITY_MISMATCH must include capability_missing")
        elif self.alignment_status == "PRIORITY_EXCEEDED" and violations != ("priority_exceeded",):
            raise ValueError("PRIORITY_EXCEEDED must include priority_exceeded")
        elif self.alignment_status == "UNSTABLE_HARDWARE" and violations != ("stability_below_tolerance",):
            raise ValueError("UNSTABLE_HARDWARE must include stability_below_tolerance")

        zero_score_statuses = {"REPLAY_UNSUPPORTED", "CAPABILITY_MISMATCH", "PRIORITY_EXCEEDED"}
        if self.alignment_status in zero_score_statuses and self.mapping_score != 0.0:
            raise ValueError("mapping_score must be 0.0 for blocked hard-failure statuses")

        none_capability_statuses = {"REPLAY_UNSUPPORTED", "CAPABILITY_MISMATCH"}
        if self.alignment_status in none_capability_statuses and self.selected_capability != "NONE":
            raise ValueError("selected_capability must be NONE for replay/capability blocked statuses")
        if self.alignment_status not in none_capability_statuses and self.selected_capability == "NONE":
            raise ValueError("selected_capability must be a capability token for this alignment_status")

        object.__setattr__(self, "constraint_violations", violations)

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "signal_id": self.signal_id,
            "hardware_id": self.hardware_id,
            "alignment_status": self.alignment_status,
            "mapping_score": _round_public_metric(float(self.mapping_score)),
            "constraint_violations": self.constraint_violations,
            "selected_capability": self.selected_capability,
            "decision_reason": self.decision_reason,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class HardwareAlignmentReceipt:
    module_version: str
    signal_count: int
    hardware_profile_count: int
    alignment_decisions: tuple[HardwareAlignmentDecision, ...]
    aligned_count: int
    degraded_count: int
    blocked_count: int
    overall_alignment_score: float
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.module_version != HARDWARE_ALIGNMENT_MODULE_VERSION:
            raise ValueError("module_version must match HARDWARE_ALIGNMENT_MODULE_VERSION")
        _require_non_negative_int(self.signal_count, name="signal_count")
        _require_non_negative_int(self.hardware_profile_count, name="hardware_profile_count")
        if not isinstance(self.alignment_decisions, tuple):
            raise ValueError("alignment_decisions must be tuple")

        for item in self.alignment_decisions:
            if not isinstance(item, HardwareAlignmentDecision):
                raise ValueError("alignment_decisions must contain HardwareAlignmentDecision")

        sorted_decisions = tuple(
            sorted(self.alignment_decisions, key=lambda d: (d.signal_id, d.hardware_id))
        )
        if sorted_decisions != self.alignment_decisions:
            raise ValueError("alignment_decisions must be sorted by (signal_id, hardware_id)")

        seen_pairs: set[tuple[str, str]] = set()
        for item in sorted_decisions:
            pair = (item.signal_id, item.hardware_id)
            if pair in seen_pairs:
                raise ValueError("duplicate (signal_id, hardware_id) is not allowed")
            seen_pairs.add(pair)

        expected_count = self.signal_count * self.hardware_profile_count
        if len(sorted_decisions) != expected_count:
            raise ValueError("alignment_decisions count must match signal_count * hardware_profile_count")

        _require_non_negative_int(self.aligned_count, name="aligned_count")
        _require_non_negative_int(self.degraded_count, name="degraded_count")
        _require_non_negative_int(self.blocked_count, name="blocked_count")
        if self.aligned_count + self.degraded_count + self.blocked_count != len(sorted_decisions):
            raise ValueError("aligned/degraded/blocked counts must match alignment_decisions length")

        aligned_count = sum(1 for d in sorted_decisions if d.alignment_status == "ALIGNED")
        degraded_count = sum(1 for d in sorted_decisions if d.alignment_status == "DEGRADED_ALIGNMENT")
        blocked_count = len(sorted_decisions) - aligned_count - degraded_count
        if (
            self.aligned_count != aligned_count
            or self.degraded_count != degraded_count
            or self.blocked_count != blocked_count
        ):
            raise ValueError("aligned/degraded/blocked counts must match decision statuses")

        object.__setattr__(self, "overall_alignment_score", _round_public_metric(_require_probability(self.overall_alignment_score, name="overall_alignment_score")))

        computed_score = (
            _round_public_metric(sum(d.mapping_score for d in sorted_decisions) / len(sorted_decisions))
            if sorted_decisions
            else 0.0
        )
        if float(self.overall_alignment_score) != computed_score:
            raise ValueError("overall_alignment_score must equal mean decision mapping_score")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "module_version": self.module_version,
            "signal_count": int(self.signal_count),
            "hardware_profile_count": int(self.hardware_profile_count),
            "alignment_decisions": tuple(decision.to_dict() for decision in self.alignment_decisions),
            "aligned_count": int(self.aligned_count),
            "degraded_count": int(self.degraded_count),
            "blocked_count": int(self.blocked_count),
            "overall_alignment_score": _round_public_metric(float(self.overall_alignment_score)),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def align_control_signals_to_hardware(
    control_signals: Sequence[ControlSignalIntent],
    hardware_profiles: Sequence[HardwareConstraintProfile],
) -> HardwareAlignmentReceipt:
    if isinstance(control_signals, (str, bytes, bytearray)) or not isinstance(control_signals, Sequence):
        raise ValueError("control_signals must be a sequence of ControlSignalIntent")
    if isinstance(hardware_profiles, (str, bytes, bytearray)) or not isinstance(hardware_profiles, Sequence):
        raise ValueError("hardware_profiles must be a sequence of HardwareConstraintProfile")

    canonical_signals = tuple(control_signals)
    canonical_hardware = tuple(hardware_profiles)

    seen_signal_ids: set[str] = set()
    for signal in canonical_signals:
        if not isinstance(signal, ControlSignalIntent):
            raise ValueError("control_signals must contain ControlSignalIntent")
        if signal.signal_id in seen_signal_ids:
            raise ValueError("duplicate signal_id is not allowed")
        seen_signal_ids.add(signal.signal_id)

    seen_hardware_ids: set[str] = set()
    for hardware in canonical_hardware:
        if not isinstance(hardware, HardwareConstraintProfile):
            raise ValueError("hardware_profiles must contain HardwareConstraintProfile")
        if hardware.hardware_id in seen_hardware_ids:
            raise ValueError("duplicate hardware_id is not allowed")
        seen_hardware_ids.add(hardware.hardware_id)

    sorted_signals = tuple(sorted(canonical_signals, key=lambda item: item.signal_id))
    sorted_hardware = tuple(sorted(canonical_hardware, key=lambda item: item.hardware_id))

    decisions: list[HardwareAlignmentDecision] = []
    for signal in sorted_signals:
        for hardware in sorted_hardware:
            if not hardware.replay_supported:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="REPLAY_UNSUPPORTED",
                        mapping_score=0.0,
                        constraint_violations=("replay_not_supported",),
                        selected_capability="NONE",
                        decision_reason="replay_not_supported",
                    )
                )
                continue

            if signal.required_capability not in hardware.capabilities:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="CAPABILITY_MISMATCH",
                        mapping_score=0.0,
                        constraint_violations=("capability_missing",),
                        selected_capability="NONE",
                        decision_reason="capability_missing",
                    )
                )
                continue

            if signal.control_priority > hardware.max_control_priority:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="PRIORITY_EXCEEDED",
                        mapping_score=0.0,
                        constraint_violations=("priority_exceeded",),
                        selected_capability=signal.required_capability,
                        decision_reason="priority_exceeded",
                    )
                )
                continue

            if hardware.stability_rating < signal.risk_tolerance:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="UNSTABLE_HARDWARE",
                        mapping_score=_round_public_metric(hardware.stability_rating),
                        constraint_violations=("stability_below_tolerance",),
                        selected_capability=signal.required_capability,
                        decision_reason="stability_below_tolerance",
                    )
                )
                continue

            score = _bounded_mapping_score(signal.control_confidence, hardware.stability_rating, signal.risk_tolerance)
            degraded_violations = tuple(
                reason
                for reason in ("high_latency", "low_stability")
                if (reason == "high_latency" and hardware.latency_class == "HIGH")
                or (reason == "low_stability" and hardware.stability_rating < 0.75)
            )
            if degraded_violations:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="DEGRADED_ALIGNMENT",
                        mapping_score=score,
                        constraint_violations=degraded_violations,
                        selected_capability=signal.required_capability,
                        decision_reason="degraded_but_usable",
                    )
                )
            else:
                decisions.append(
                    HardwareAlignmentDecision(
                        signal_id=signal.signal_id,
                        hardware_id=hardware.hardware_id,
                        alignment_status="ALIGNED",
                        mapping_score=score,
                        constraint_violations=tuple(),
                        selected_capability=signal.required_capability,
                        decision_reason="aligned",
                    )
                )

    decisions_tuple = tuple(decisions)
    aligned_count = sum(1 for d in decisions_tuple if d.alignment_status == "ALIGNED")
    degraded_count = sum(1 for d in decisions_tuple if d.alignment_status == "DEGRADED_ALIGNMENT")
    blocked_count = len(decisions_tuple) - aligned_count - degraded_count
    overall_score = (
        _round_public_metric(sum(d.mapping_score for d in decisions_tuple) / len(decisions_tuple))
        if decisions_tuple
        else 0.0
    )

    return HardwareAlignmentReceipt(
        module_version=HARDWARE_ALIGNMENT_MODULE_VERSION,
        signal_count=len(sorted_signals),
        hardware_profile_count=len(sorted_hardware),
        alignment_decisions=decisions_tuple,
        aligned_count=aligned_count,
        degraded_count=degraded_count,
        blocked_count=blocked_count,
        overall_alignment_score=overall_score,
    )


__all__ = [
    "HARDWARE_ALIGNMENT_MODULE_VERSION",
    "ControlSignalIntent",
    "HardwareConstraintProfile",
    "HardwareAlignmentDecision",
    "HardwareAlignmentReceipt",
    "align_control_signals_to_hardware",
]
