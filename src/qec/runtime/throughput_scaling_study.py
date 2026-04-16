# SPDX-License-Identifier: MIT
"""v138.2.2 — deterministic throughput scaling study layer.

This module is additive to v138.2.1 latency-budget enforcement and provides
mathematical throughput scaling analysis (no wall-clock timing, no async, no threads).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from qec.runtime.latency_budget_enforcement_hardware import (
    LatencyBudgetEnforcement,
    LatencyBudgetPolicy,
    LatencyBudgetReceipt,
    LatencyBudgetValidationReport,
    LatencyEnforcementDecision,
)

THROUGHPUT_SCALING_STUDY_VERSION = "v138.2.2"

_SUPPORTED_SCALING_MODES: Tuple[str, ...] = ("linear", "saturating", "bounded_mesh")
_SUPPORTED_DEGRADATION_MODES: Tuple[str, ...] = ("none", "soft_throttle", "hard_cap")
_SHA256_HEX_CHARS: frozenset = frozenset("0123456789abcdef")


class ThroughputScalingValidationError(ValueError):
    """Raised when throughput scaling study data violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if value is None or not isinstance(value, str):
        raise ThroughputScalingValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ThroughputScalingValidationError(f"{field} must be non-empty")
    return text


def _normalize_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThroughputScalingValidationError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ThroughputScalingValidationError(f"{field} must be >= {minimum}")
    return result


def _normalize_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ThroughputScalingValidationError(f"{field} must be a boolean")
    return value


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThroughputScalingValidationError(f"{field} must be a finite float")
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise ThroughputScalingValidationError(f"{field} must be a finite float")
    return result


def _normalize_sha256_hex(value: Any, *, field: str) -> str:
    text = _normalize_text(value, field=field).lower()
    if len(text) != 64:
        raise ThroughputScalingValidationError(f"{field} must be a 64-character SHA-256 hex string")
    if not frozenset(text) <= _SHA256_HEX_CHARS:
        raise ThroughputScalingValidationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ThroughputScalingValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise ThroughputScalingValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ThroughputScalingValidationError(f"{field} contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class ThroughputScalingPolicy:
    policy_id: str
    scaling_mode: str
    max_parallel_lanes: int
    target_ops_per_window: int
    degradation_mode: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "scaling_mode": self.scaling_mode,
            "max_parallel_lanes": int(self.max_parallel_lanes),
            "target_ops_per_window": int(self.target_ops_per_window),
            "degradation_mode": self.degradation_mode,
            "metadata": _canonicalize_value(dict(self.metadata), field="policy.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ThroughputSample:
    sample_id: str
    lane_count: int
    accepted_dispatches: int
    rejected_dispatches: int
    projected_ops_per_window: int
    effective_ops_per_window: int
    saturation_score: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "lane_count": int(self.lane_count),
            "accepted_dispatches": int(self.accepted_dispatches),
            "rejected_dispatches": int(self.rejected_dispatches),
            "projected_ops_per_window": int(self.projected_ops_per_window),
            "effective_ops_per_window": int(self.effective_ops_per_window),
            "saturation_score": float(self.saturation_score),
            "metadata": _canonicalize_value(dict(self.metadata), field="sample.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ThroughputScalingReceipt:
    policy_hash: str
    sample_set_hash: str
    scaling_hash: str
    study_valid: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "sample_set_hash": self.sample_set_hash,
            "scaling_hash": self.scaling_hash,
            "study_valid": bool(self.study_valid),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "sample_set_hash": self.sample_set_hash,
            "scaling_hash": self.scaling_hash,
            "study_valid": bool(self.study_valid),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ThroughputScalingValidationReport:
    valid: bool
    errors: Tuple[str, ...]
    error_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "errors": list(self.errors),
            "error_count": int(self.error_count),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ThroughputScalingStudy:
    policy: ThroughputScalingPolicy
    samples: Tuple[ThroughputSample, ...]
    receipt: ThroughputScalingReceipt
    validation: ThroughputScalingValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_policy(raw: ThroughputScalingPolicy | Mapping[str, Any]) -> ThroughputScalingPolicy:
    if isinstance(raw, ThroughputScalingPolicy):
        return raw
    if not isinstance(raw, Mapping):
        raise ThroughputScalingValidationError("policy must be mapping or ThroughputScalingPolicy")
    _meta = raw.get("metadata")
    return ThroughputScalingPolicy(
        policy_id=_normalize_text(raw.get("policy_id"), field="policy.policy_id"),
        scaling_mode=_normalize_text(raw.get("scaling_mode"), field="policy.scaling_mode"),
        max_parallel_lanes=_normalize_int(raw.get("max_parallel_lanes"), field="policy.max_parallel_lanes"),
        target_ops_per_window=_normalize_int(raw.get("target_ops_per_window"), field="policy.target_ops_per_window"),
        degradation_mode=_normalize_text(raw.get("degradation_mode"), field="policy.degradation_mode"),
        metadata=_canonicalize_value(_meta if isinstance(_meta, Mapping) else {}, field="policy.metadata"),
    )


def _normalize_enforcement_set(
    raw_set: Sequence[LatencyBudgetEnforcement | Mapping[str, Any]],
) -> Tuple[LatencyBudgetEnforcement, ...]:
    normalized = []
    for item in raw_set:
        if isinstance(item, LatencyBudgetEnforcement):
            normalized.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ThroughputScalingValidationError("enforcement_set entries must be mapping or LatencyBudgetEnforcement")
        normalized.append(
            LatencyBudgetEnforcement(
                policy=LatencyBudgetPolicy(
                    policy_id=_normalize_text(item["policy"]["policy_id"], field="enforcement.policy.policy_id"),
                    max_latency_ns=_normalize_int(item["policy"]["max_latency_ns"], field="enforcement.policy.max_latency_ns"),
                    hard_limit_ns=_normalize_int(item["policy"]["hard_limit_ns"], field="enforcement.policy.hard_limit_ns"),
                    violation_action=_normalize_text(
                        item["policy"]["violation_action"], field="enforcement.policy.violation_action"
                    ),
                    recovery_mode=_normalize_text(item["policy"]["recovery_mode"], field="enforcement.policy.recovery_mode"),
                    metadata=_canonicalize_value(item["policy"].get("metadata", {}), field="enforcement.policy.metadata"),
                ),
                decision=LatencyEnforcementDecision(
                    dispatch_id=_normalize_sha256_hex(item["decision"]["dispatch_id"], field="enforcement.decision.dispatch_id"),
                    projected_latency_ns=_normalize_int(
                        item["decision"]["projected_latency_ns"], field="enforcement.decision.projected_latency_ns", minimum=0
                    ),
                    within_budget=_normalize_bool(item["decision"]["within_budget"], field="enforcement.decision.within_budget"),
                    hard_limit_breached=_normalize_bool(
                        item["decision"]["hard_limit_breached"], field="enforcement.decision.hard_limit_breached"
                    ),
                    decision=_normalize_text(item["decision"]["decision"], field="enforcement.decision.decision"),
                    enforcement_reason=_normalize_text(
                        item["decision"]["enforcement_reason"], field="enforcement.decision.enforcement_reason"
                    ),
                    metadata=_canonicalize_value(
                        item["decision"].get("metadata", {}),
                        field="enforcement.decision.metadata",
                    ),
                ),
                receipt=LatencyBudgetReceipt(
                    policy_hash=_normalize_sha256_hex(item["receipt"]["policy_hash"], field="enforcement.receipt.policy_hash"),
                    dispatch_hash=_normalize_sha256_hex(item["receipt"]["dispatch_hash"], field="enforcement.receipt.dispatch_hash"),
                    decision_hash=_normalize_sha256_hex(item["receipt"]["decision_hash"], field="enforcement.receipt.decision_hash"),
                    within_budget=_normalize_bool(item["receipt"]["within_budget"], field="enforcement.receipt.within_budget"),
                    receipt_hash=_normalize_sha256_hex(item["receipt"]["receipt_hash"], field="enforcement.receipt.receipt_hash"),
                ),
                validation=LatencyBudgetValidationReport(
                    valid=_normalize_bool(
                        item.get("validation", {}).get("valid", True), field="enforcement.validation.valid"
                    ),
                    errors=tuple(str(v) for v in tuple(item.get("validation", {}).get("errors", ()))),
                    error_count=_normalize_int(
                        item.get("validation", {}).get("error_count", 0),
                        field="enforcement.validation.error_count",
                        minimum=0,
                    ),
                ),
                target_family=_normalize_text(item["target_family"], field="enforcement.target_family"),
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda e: (
                e.decision.dispatch_id,
                e.receipt.dispatch_hash,
                e.stable_hash(),
            ),
        )
    )


def _effective_ops_cap(policy: ThroughputScalingPolicy) -> int:
    if policy.scaling_mode == "linear":
        return policy.target_ops_per_window
    if policy.scaling_mode == "saturating":
        return policy.target_ops_per_window + (policy.max_parallel_lanes * 25)
    if policy.scaling_mode == "bounded_mesh":
        return policy.target_ops_per_window + (policy.max_parallel_lanes * 10)
    raise ThroughputScalingValidationError(f"unsupported policy.scaling_mode: {policy.scaling_mode!r}")


def _extract_lane_count(enforcement: LatencyBudgetEnforcement, fallback_lane_count: int) -> int:
    raw_lane = enforcement.decision.metadata.get("lane_count")
    if isinstance(raw_lane, int) and not isinstance(raw_lane, bool) and raw_lane >= 0:
        return int(raw_lane)
    return fallback_lane_count


def compute_throughput_profile(
    enforcement_set: Sequence[LatencyBudgetEnforcement],
    scaling_policy: ThroughputScalingPolicy | Mapping[str, Any],
) -> Tuple[ThroughputSample, ...]:
    """Deterministically compute analytical throughput samples from enforcement outcomes."""
    policy = _normalize_policy(scaling_policy)
    cap = _effective_ops_cap(policy)

    accepted_running = 0
    rejected_running = 0
    samples = []

    enriched = []
    for index, enforcement in enumerate(enforcement_set):
        lane_count = _extract_lane_count(enforcement, index + 1)
        enriched.append((lane_count, index, enforcement))

    ordered_enforcements = tuple(
        sorted(
            enriched,
            key=lambda item: (
                item[0],
                item[2].decision.dispatch_id,
                item[2].receipt.dispatch_hash,
                item[1],
            ),
        )
    )

    for lane_count, _index, enforcement in ordered_enforcements:

        if enforcement.decision.decision == "reject":
            rejected_running += 1
        else:
            accepted_running += 1

        projected_ops = accepted_running * 100
        effective_ops = min(projected_ops, cap)

        if policy.degradation_mode == "soft_throttle" and lane_count > policy.max_parallel_lanes:
            effective_ops = int(math.floor(float(effective_ops) * 0.9))

        saturation_score = min(1.0, float(projected_ops) / float(max(1, cap)))

        sample_payload = {
            "dispatch_id": enforcement.decision.dispatch_id,
            "dispatch_hash": enforcement.receipt.dispatch_hash,
            "lane_count": lane_count,
            "accepted_dispatches": accepted_running,
            "rejected_dispatches": rejected_running,
            "projected_ops_per_window": projected_ops,
            "effective_ops_per_window": effective_ops,
            "saturation_score": saturation_score,
            "scaling_mode": policy.scaling_mode,
            "degradation_mode": policy.degradation_mode,
        }
        sample_id = _stable_hash(sample_payload)

        samples.append(
            ThroughputSample(
                sample_id=sample_id,
                lane_count=lane_count,
                accepted_dispatches=accepted_running,
                rejected_dispatches=rejected_running,
                projected_ops_per_window=projected_ops,
                effective_ops_per_window=effective_ops,
                saturation_score=saturation_score,
                metadata={
                    "dispatch_id": enforcement.decision.dispatch_id,
                    "dispatch_hash": enforcement.receipt.dispatch_hash,
                    "target_family": enforcement.target_family,
                    "decision": enforcement.decision.decision,
                    "projected_latency_ns": enforcement.decision.projected_latency_ns,
                    "violation_class": enforcement.decision.metadata.get("violation_class", "unknown"),
                    "enforcement_hash": enforcement.stable_hash(),
                },
            )
        )

    return tuple(sorted(samples, key=lambda sample: (sample.lane_count, sample.sample_id)))


def _build_receipt(
    *,
    policy: ThroughputScalingPolicy,
    samples: Tuple[ThroughputSample, ...],
    study_valid: bool,
) -> ThroughputScalingReceipt:
    scaling_hash = _stable_hash(
        {
            "scaling_mode": policy.scaling_mode,
            "degradation_mode": policy.degradation_mode,
            "max_parallel_lanes": policy.max_parallel_lanes,
            "target_ops_per_window": policy.target_ops_per_window,
        }
    )
    sample_set_hash = _stable_hash([sample.stable_hash() for sample in samples])
    receipt = ThroughputScalingReceipt(
        policy_hash=policy.stable_hash(),
        sample_set_hash=sample_set_hash,
        scaling_hash=scaling_hash,
        study_valid=bool(study_valid),
        receipt_hash="",
    )
    return ThroughputScalingReceipt(
        policy_hash=receipt.policy_hash,
        sample_set_hash=receipt.sample_set_hash,
        scaling_hash=receipt.scaling_hash,
        study_valid=receipt.study_valid,
        receipt_hash=receipt.stable_hash(),
    )


def build_throughput_scaling_study(
    *,
    enforcement_set: Sequence[LatencyBudgetEnforcement | Mapping[str, Any]],
    scaling_policy: ThroughputScalingPolicy | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ThroughputScalingStudy:
    """Build deterministic throughput scaling study from v138.2.1 enforcement outputs."""
    del metadata  # additive extension point for grouped studies

    policy = _normalize_policy(scaling_policy)
    normalized_enforcements = _normalize_enforcement_set(enforcement_set)
    samples = compute_throughput_profile(normalized_enforcements, policy)

    provisional_receipt = _build_receipt(policy=policy, samples=samples, study_valid=True)
    provisional = ThroughputScalingStudy(
        policy=policy,
        samples=samples,
        receipt=provisional_receipt,
        validation=ThroughputScalingValidationReport(valid=True, errors=(), error_count=0),
    )
    validation = validate_throughput_scaling_study(provisional)
    receipt = _build_receipt(policy=policy, samples=samples, study_valid=validation.valid)

    final_study = ThroughputScalingStudy(
        policy=policy,
        samples=samples,
        receipt=receipt,
        validation=validation,
    )

    final_validation = validate_throughput_scaling_study(final_study)
    if final_validation != validation:
        return ThroughputScalingStudy(
            policy=policy,
            samples=samples,
            receipt=receipt,
            validation=final_validation,
        )
    return final_study


def validate_throughput_scaling_study(
    study_obj: ThroughputScalingStudy | Mapping[str, Any],
) -> ThroughputScalingValidationReport:
    """Validate deterministic throughput scaling semantics and receipt consistency."""
    errors = []

    try:
        if isinstance(study_obj, ThroughputScalingStudy):
            study = study_obj
        elif isinstance(study_obj, Mapping):
            policy = _normalize_policy(study_obj["policy"])
            sample_objs = []
            for sample_map in tuple(study_obj.get("samples", ())):
                sample_meta = sample_map.get("metadata")
                sample_objs.append(
                    ThroughputSample(
                        sample_id=_normalize_sha256_hex(sample_map["sample_id"], field="sample.sample_id"),
                        lane_count=_normalize_int(sample_map["lane_count"], field="sample.lane_count"),
                        accepted_dispatches=_normalize_int(
                            sample_map["accepted_dispatches"], field="sample.accepted_dispatches"
                        ),
                        rejected_dispatches=_normalize_int(
                            sample_map["rejected_dispatches"], field="sample.rejected_dispatches"
                        ),
                        projected_ops_per_window=_normalize_int(
                            sample_map["projected_ops_per_window"], field="sample.projected_ops_per_window"
                        ),
                        effective_ops_per_window=_normalize_int(
                            sample_map["effective_ops_per_window"], field="sample.effective_ops_per_window"
                        ),
                        saturation_score=_normalize_float(sample_map["saturation_score"], field="sample.saturation_score"),
                        metadata=_canonicalize_value(
                            sample_meta if isinstance(sample_meta, Mapping) else {},
                            field="sample.metadata",
                        ),
                    )
                )

            receipt_map = study_obj["receipt"]
            validation_map = study_obj.get("validation", {"valid": True, "errors": [], "error_count": 0})
            study = ThroughputScalingStudy(
                policy=policy,
                samples=tuple(sample_objs),
                receipt=ThroughputScalingReceipt(
                    policy_hash=_normalize_sha256_hex(receipt_map["policy_hash"], field="receipt.policy_hash"),
                    sample_set_hash=_normalize_sha256_hex(receipt_map["sample_set_hash"], field="receipt.sample_set_hash"),
                    scaling_hash=_normalize_sha256_hex(receipt_map["scaling_hash"], field="receipt.scaling_hash"),
                    study_valid=_normalize_bool(receipt_map["study_valid"], field="receipt.study_valid"),
                    receipt_hash=_normalize_sha256_hex(receipt_map["receipt_hash"], field="receipt.receipt_hash"),
                ),
                validation=ThroughputScalingValidationReport(
                    valid=_normalize_bool(validation_map.get("valid", True), field="validation.valid"),
                    errors=tuple(str(v) for v in tuple(validation_map.get("errors", ()))),
                    error_count=_normalize_int(validation_map.get("error_count", 0), field="validation.error_count", minimum=0),
                ),
            )
        else:
            raise ThroughputScalingValidationError("study_obj must be mapping or ThroughputScalingStudy")
    except Exception as exc:
        return ThroughputScalingValidationReport(valid=False, errors=(f"normalization_failed: {exc}",), error_count=1)

    if study.policy.max_parallel_lanes <= 0:
        errors.append("policy.max_parallel_lanes must be > 0")
    if study.policy.target_ops_per_window <= 0:
        errors.append("policy.target_ops_per_window must be > 0")
    if study.policy.scaling_mode not in _SUPPORTED_SCALING_MODES:
        errors.append(f"unsupported policy.scaling_mode: {study.policy.scaling_mode!r}")
    if study.policy.degradation_mode not in _SUPPORTED_DEGRADATION_MODES:
        errors.append(f"unsupported policy.degradation_mode: {study.policy.degradation_mode!r}")

    cap = _effective_ops_cap(study.policy) if study.policy.scaling_mode in _SUPPORTED_SCALING_MODES else 1

    previous_sort_key = None
    for sample in study.samples:
        if sample.lane_count < 0:
            errors.append(f"sample[{sample.sample_id}].lane_count must be >= 0")
        if sample.accepted_dispatches < 0:
            errors.append(f"sample[{sample.sample_id}].accepted_dispatches must be >= 0")
        if sample.rejected_dispatches < 0:
            errors.append(f"sample[{sample.sample_id}].rejected_dispatches must be >= 0")
        if sample.accepted_dispatches + sample.rejected_dispatches <= 0:
            errors.append(f"sample[{sample.sample_id}].accepted + rejected must be > 0")
        if study.policy.degradation_mode in ("soft_throttle", "hard_cap"):
            if sample.effective_ops_per_window > sample.projected_ops_per_window:
                errors.append(f"sample[{sample.sample_id}] effective_ops_per_window exceeds projected_ops_per_window")
        if math.isnan(sample.saturation_score) or math.isinf(sample.saturation_score):
            errors.append(f"sample[{sample.sample_id}].saturation_score must be finite")
        if not (0.0 <= sample.saturation_score <= 1.0):
            errors.append(f"sample[{sample.sample_id}].saturation_score must be within [0, 1]")

        expected_saturation = min(1.0, float(sample.projected_ops_per_window) / float(max(1, cap)))
        if abs(sample.saturation_score - expected_saturation) > 1e-12:
            errors.append(f"sample[{sample.sample_id}].saturation_score mismatch")

        sort_key = (sample.lane_count, sample.sample_id)
        if previous_sort_key is not None and sort_key < previous_sort_key:
            errors.append("samples are not in deterministic ordering")
        previous_sort_key = sort_key

    expected_receipt = _build_receipt(policy=study.policy, samples=study.samples, study_valid=not errors)

    if study.receipt.policy_hash != study.policy.stable_hash():
        errors.append("receipt.policy_hash mismatch")
    if study.receipt.sample_set_hash != _stable_hash([sample.stable_hash() for sample in study.samples]):
        errors.append("receipt.sample_set_hash mismatch")
    if study.receipt.scaling_hash != expected_receipt.scaling_hash:
        errors.append("receipt.scaling_hash mismatch")
    if study.receipt.study_valid != (not errors):
        errors.append("receipt.study_valid mismatch")
    if study.receipt.receipt_hash != _build_receipt(policy=study.policy, samples=study.samples, study_valid=study.receipt.study_valid).receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    if _canonical_json(study.to_dict()) != study.to_canonical_json():
        errors.append("canonical ordering mismatch")

    return ThroughputScalingValidationReport(valid=not errors, errors=tuple(errors), error_count=len(errors))


def throughput_replay_projection(
    study_obj: ThroughputScalingStudy,
) -> Dict[str, Any]:
    """Emit deterministic replay-safe throughput lineage for v138.2.3 integration."""
    payload = {
        "module_version": THROUGHPUT_SCALING_STUDY_VERSION,
        "policy_hash": study_obj.receipt.policy_hash,
        "sample_set_hash": study_obj.receipt.sample_set_hash,
        "scaling_hash": study_obj.receipt.scaling_hash,
        "receipt_hash": study_obj.receipt.receipt_hash,
        "study_hash": study_obj.stable_hash(),
        "sample_hashes": [sample.stable_hash() for sample in study_obj.samples],
        "study_valid": study_obj.validation.valid,
    }
    payload["projection_hash"] = _stable_hash(payload)
    return payload
