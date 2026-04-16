# SPDX-License-Identifier: MIT
"""v138.2.3 — deterministic thermal / power budget receipt pack layer.

This module is additive to v138.2.2 throughput scaling study and provides
mathematical thermal/power observability receipts (no wall-clock timing,
no hardware telemetry, no async, no threads).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

from qec.runtime.throughput_scaling_study import (
    ThroughputSample,
    ThroughputScalingPolicy,
    ThroughputScalingReceipt,
    ThroughputScalingStudy,
    ThroughputScalingValidationReport,
)

THERMAL_POWER_BUDGET_RECEIPT_PACK_VERSION = "v138.2.3"

_SUPPORTED_THERMAL_MODES: Tuple[str, ...] = ("linear", "saturating", "bounded_envelope")
_SUPPORTED_POWER_MODES: Tuple[str, ...] = ("proportional", "capped", "throttled")
_SHA256_HEX_CHARS: frozenset = frozenset("0123456789abcdef")

_THERMAL_DIVISOR: int = 10
_POWER_DIVISOR: int = 8
# Extra headroom allowed above the declared budget before the saturating/throttled cap takes effect.
_THERMAL_SATURATION_HEADROOM: int = 10
_POWER_THROTTLE_HEADROOM: int = 10


class ThermalPowerBudgetValidationError(ValueError):
    """Raised when thermal/power budget data violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if value is None or not isinstance(value, str):
        raise ThermalPowerBudgetValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ThermalPowerBudgetValidationError(f"{field} must be non-empty")
    return text


def _normalize_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThermalPowerBudgetValidationError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ThermalPowerBudgetValidationError(f"{field} must be >= {minimum}")
    return result


def _normalize_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ThermalPowerBudgetValidationError(f"{field} must be a boolean")
    return value


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThermalPowerBudgetValidationError(f"{field} must be a finite float")
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise ThermalPowerBudgetValidationError(f"{field} must be a finite float")
    return result


def _normalize_sha256_hex(value: Any, *, field: str) -> str:
    text = _normalize_text(value, field=field).lower()
    if len(text) != 64:
        raise ThermalPowerBudgetValidationError(f"{field} must be a 64-character SHA-256 hex string")
    if not frozenset(text) <= _SHA256_HEX_CHARS:
        raise ThermalPowerBudgetValidationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ThermalPowerBudgetValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise ThermalPowerBudgetValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ThermalPowerBudgetValidationError(f"{field} contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class ThermalPowerPolicy:
    policy_id: str
    max_thermal_units: int
    max_power_units: int
    thermal_mode: str
    power_mode: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_thermal_units": int(self.max_thermal_units),
            "max_power_units": int(self.max_power_units),
            "thermal_mode": self.thermal_mode,
            "power_mode": self.power_mode,
            "metadata": _canonicalize_value(dict(self.metadata), field="policy.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ThermalBudgetReceipt:
    sample_id: str
    lane_count: int
    projected_thermal_units: int
    within_thermal_budget: bool
    thermal_pressure_score: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "lane_count": int(self.lane_count),
            "projected_thermal_units": int(self.projected_thermal_units),
            "within_thermal_budget": bool(self.within_thermal_budget),
            "thermal_pressure_score": float(self.thermal_pressure_score),
            "metadata": _canonicalize_value(dict(self.metadata), field="thermal_receipt.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PowerBudgetReceipt:
    sample_id: str
    lane_count: int
    projected_power_units: int
    within_power_budget: bool
    power_pressure_score: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "lane_count": int(self.lane_count),
            "projected_power_units": int(self.projected_power_units),
            "within_power_budget": bool(self.within_power_budget),
            "power_pressure_score": float(self.power_pressure_score),
            "metadata": _canonicalize_value(dict(self.metadata), field="power_receipt.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ThermalPowerReceiptPack:
    policy_hash: str
    thermal_hash: str
    power_hash: str
    pack_valid: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "thermal_hash": self.thermal_hash,
            "power_hash": self.power_hash,
            "pack_valid": bool(self.pack_valid),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "thermal_hash": self.thermal_hash,
            "power_hash": self.power_hash,
            "pack_valid": bool(self.pack_valid),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ThermalPowerValidationReport:
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
class ThermalPowerBudgetStudy:
    policy: ThermalPowerPolicy
    thermal_receipts: Tuple[ThermalBudgetReceipt, ...]
    power_receipts: Tuple[PowerBudgetReceipt, ...]
    receipt_pack: ThermalPowerReceiptPack
    validation: ThermalPowerValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "thermal_receipts": [receipt.to_dict() for receipt in self.thermal_receipts],
            "power_receipts": [receipt.to_dict() for receipt in self.power_receipts],
            "receipt_pack": self.receipt_pack.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_policy(raw: ThermalPowerPolicy | Mapping[str, Any]) -> ThermalPowerPolicy:
    if isinstance(raw, ThermalPowerPolicy):
        return raw
    if not isinstance(raw, Mapping):
        raise ThermalPowerBudgetValidationError("policy must be mapping or ThermalPowerPolicy")
    _meta = raw.get("metadata")
    return ThermalPowerPolicy(
        policy_id=_normalize_text(raw.get("policy_id"), field="policy.policy_id"),
        max_thermal_units=_normalize_int(raw.get("max_thermal_units"), field="policy.max_thermal_units"),
        max_power_units=_normalize_int(raw.get("max_power_units"), field="policy.max_power_units"),
        thermal_mode=_normalize_text(raw.get("thermal_mode"), field="policy.thermal_mode"),
        power_mode=_normalize_text(raw.get("power_mode"), field="policy.power_mode"),
        metadata=_canonicalize_value(_meta if isinstance(_meta, Mapping) else {}, field="policy.metadata"),
    )


def _normalize_throughput_study(raw: ThroughputScalingStudy | Mapping[str, Any]) -> ThroughputScalingStudy:
    if isinstance(raw, ThroughputScalingStudy):
        return raw
    if not isinstance(raw, Mapping):
        raise ThermalPowerBudgetValidationError("throughput_study must be mapping or ThroughputScalingStudy")

    policy_map = raw.get("policy", {})
    normalized_policy = ThroughputScalingPolicy(
        policy_id=_normalize_text(policy_map.get("policy_id"), field="throughput.policy_id"),
        scaling_mode=_normalize_text(policy_map.get("scaling_mode"), field="throughput.scaling_mode"),
        max_parallel_lanes=_normalize_int(policy_map.get("max_parallel_lanes"), field="throughput.max_parallel_lanes"),
        target_ops_per_window=_normalize_int(
            policy_map.get("target_ops_per_window"), field="throughput.target_ops_per_window"
        ),
        degradation_mode=_normalize_text(policy_map.get("degradation_mode"), field="throughput.degradation_mode"),
        metadata=_canonicalize_value(policy_map.get("metadata", {}), field="throughput.policy.metadata"),
    )

    sample_objs = []
    for sample_map in tuple(raw.get("samples", ())):
        sample_meta = sample_map.get("metadata")
        sample_objs.append(
            ThroughputSample(
                sample_id=_normalize_sha256_hex(sample_map["sample_id"], field="throughput.sample.sample_id"),
                lane_count=_normalize_int(sample_map["lane_count"], field="throughput.sample.lane_count"),
                accepted_dispatches=_normalize_int(
                    sample_map["accepted_dispatches"], field="throughput.sample.accepted_dispatches"
                ),
                rejected_dispatches=_normalize_int(
                    sample_map["rejected_dispatches"], field="throughput.sample.rejected_dispatches"
                ),
                projected_ops_per_window=_normalize_int(
                    sample_map["projected_ops_per_window"], field="throughput.sample.projected_ops_per_window"
                ),
                effective_ops_per_window=_normalize_int(
                    sample_map["effective_ops_per_window"], field="throughput.sample.effective_ops_per_window"
                ),
                saturation_score=_normalize_float(sample_map["saturation_score"], field="throughput.sample.saturation_score"),
                metadata=_canonicalize_value(
                    sample_meta if isinstance(sample_meta, Mapping) else {},
                    field="throughput.sample.metadata",
                ),
            )
        )

    receipt_map = raw.get("receipt", {})
    validation_map = raw.get("validation", {"valid": True, "errors": [], "error_count": 0})
    return ThroughputScalingStudy(
        policy=normalized_policy,
        samples=tuple(sorted(sample_objs, key=lambda s: (s.lane_count, s.sample_id))),
        receipt=ThroughputScalingReceipt(
            policy_hash=_normalize_sha256_hex(receipt_map.get("policy_hash", "0" * 64), field="throughput.receipt.policy_hash"),
            sample_set_hash=_normalize_sha256_hex(
                receipt_map.get("sample_set_hash", "0" * 64), field="throughput.receipt.sample_set_hash"
            ),
            scaling_hash=_normalize_sha256_hex(receipt_map.get("scaling_hash", "0" * 64), field="throughput.receipt.scaling_hash"),
            study_valid=_normalize_bool(receipt_map.get("study_valid", True), field="throughput.receipt.study_valid"),
            receipt_hash=_normalize_sha256_hex(receipt_map.get("receipt_hash", "0" * 64), field="throughput.receipt.receipt_hash"),
        ),
        validation=ThroughputScalingValidationReport(
            valid=_normalize_bool(validation_map.get("valid", True), field="throughput.validation.valid"),
            errors=tuple(str(v) for v in tuple(validation_map.get("errors", ()))),
            error_count=_normalize_int(validation_map.get("error_count", 0), field="throughput.validation.error_count", minimum=0),
        ),
    )


def _apply_thermal_mode(*, ops: int, budget: int, mode: str) -> int:
    base = ops // _THERMAL_DIVISOR
    if mode == "linear":
        return base
    if mode == "saturating":
        return min(base, budget + _THERMAL_SATURATION_HEADROOM)
    if mode == "bounded_envelope":
        return min(base, budget)
    raise ThermalPowerBudgetValidationError(f"unsupported policy.thermal_mode: {mode!r}")


def _apply_power_mode(*, ops: int, budget: int, mode: str) -> int:
    base = ops // _POWER_DIVISOR
    if mode == "proportional":
        return base
    if mode == "capped":
        return min(base, budget)
    if mode == "throttled":
        return min(base, budget + _POWER_THROTTLE_HEADROOM)
    raise ThermalPowerBudgetValidationError(f"unsupported policy.power_mode: {mode!r}")


def compute_thermal_power_profile(
    throughput_study: ThroughputScalingStudy | Mapping[str, Any],
    policy: ThermalPowerPolicy | Mapping[str, Any],
) -> Tuple[Tuple[ThermalBudgetReceipt, ...], Tuple[PowerBudgetReceipt, ...]]:
    """Deterministically compute thermal/power receipts from throughput samples."""
    normalized_study = _normalize_throughput_study(throughput_study)
    normalized_policy = _normalize_policy(policy)

    ordered_samples = tuple(sorted(normalized_study.samples, key=lambda sample: (sample.lane_count, sample.sample_id)))
    thermal_receipts = []
    power_receipts = []
    for sample in ordered_samples:
        ops = _normalize_int(sample.effective_ops_per_window, field=f"sample[{sample.sample_id}].effective_ops_per_window")
        projected_thermal_units = _apply_thermal_mode(
            ops=ops,
            budget=normalized_policy.max_thermal_units,
            mode=normalized_policy.thermal_mode,
        )
        projected_power_units = _apply_power_mode(
            ops=ops,
            budget=normalized_policy.max_power_units,
            mode=normalized_policy.power_mode,
        )

        thermal_pressure = min(1.0, float(projected_thermal_units) / float(max(1, normalized_policy.max_thermal_units)))
        power_pressure = min(1.0, float(projected_power_units) / float(max(1, normalized_policy.max_power_units)))

        thermal_receipts.append(
            ThermalBudgetReceipt(
                sample_id=sample.sample_id,
                lane_count=sample.lane_count,
                projected_thermal_units=projected_thermal_units,
                within_thermal_budget=projected_thermal_units <= normalized_policy.max_thermal_units,
                thermal_pressure_score=thermal_pressure,
                metadata={
                    "lane_count": sample.lane_count,
                    "effective_ops_per_window": sample.effective_ops_per_window,
                    "sample_hash": sample.stable_hash(),
                    "thermal_mode": normalized_policy.thermal_mode,
                    "thermal_divisor": _THERMAL_DIVISOR,
                },
            )
        )
        power_receipts.append(
            PowerBudgetReceipt(
                sample_id=sample.sample_id,
                lane_count=sample.lane_count,
                projected_power_units=projected_power_units,
                within_power_budget=projected_power_units <= normalized_policy.max_power_units,
                power_pressure_score=power_pressure,
                metadata={
                    "lane_count": sample.lane_count,
                    "effective_ops_per_window": sample.effective_ops_per_window,
                    "sample_hash": sample.stable_hash(),
                    "power_mode": normalized_policy.power_mode,
                    "power_divisor": _POWER_DIVISOR,
                },
            )
        )

    return (
        tuple(sorted(thermal_receipts, key=lambda r: (r.lane_count, r.sample_id))),
        tuple(sorted(power_receipts, key=lambda r: (r.lane_count, r.sample_id))),
    )


def _build_receipt_pack(
    *,
    policy: ThermalPowerPolicy,
    thermal_receipts: Tuple[ThermalBudgetReceipt, ...],
    power_receipts: Tuple[PowerBudgetReceipt, ...],
    pack_valid: bool,
) -> ThermalPowerReceiptPack:
    receipt_pack = ThermalPowerReceiptPack(
        policy_hash=policy.stable_hash(),
        thermal_hash=_stable_hash([receipt.stable_hash() for receipt in thermal_receipts]),
        power_hash=_stable_hash([receipt.stable_hash() for receipt in power_receipts]),
        pack_valid=bool(pack_valid),
        receipt_hash="",
    )
    return ThermalPowerReceiptPack(
        policy_hash=receipt_pack.policy_hash,
        thermal_hash=receipt_pack.thermal_hash,
        power_hash=receipt_pack.power_hash,
        pack_valid=receipt_pack.pack_valid,
        receipt_hash=receipt_pack.stable_hash(),
    )


def build_thermal_power_budget_pack(
    *,
    throughput_study: ThroughputScalingStudy | Mapping[str, Any],
    thermal_power_policy: ThermalPowerPolicy | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ThermalPowerBudgetStudy:
    """Build deterministic thermal/power budget receipt pack from v138.2.2 study outputs."""
    del metadata

    policy = _normalize_policy(thermal_power_policy)
    normalized_study = _normalize_throughput_study(throughput_study)

    try:
        thermal_receipts, power_receipts = compute_thermal_power_profile(normalized_study, policy)
    except ThermalPowerBudgetValidationError as exc:
        error_msg = str(exc)
        invalid_receipt_pack = _build_receipt_pack(
            policy=policy,
            thermal_receipts=(),
            power_receipts=(),
            pack_valid=False,
        )
        return ThermalPowerBudgetStudy(
            policy=policy,
            thermal_receipts=(),
            power_receipts=(),
            receipt_pack=invalid_receipt_pack,
            validation=ThermalPowerValidationReport(valid=False, errors=(error_msg,), error_count=1),
        )

    provisional_pack = _build_receipt_pack(
        policy=policy,
        thermal_receipts=thermal_receipts,
        power_receipts=power_receipts,
        pack_valid=True,
    )
    provisional = ThermalPowerBudgetStudy(
        policy=policy,
        thermal_receipts=thermal_receipts,
        power_receipts=power_receipts,
        receipt_pack=provisional_pack,
        validation=ThermalPowerValidationReport(valid=True, errors=(), error_count=0),
    )

    validation = validate_thermal_power_budget_pack(provisional)
    receipt_pack = _build_receipt_pack(
        policy=policy,
        thermal_receipts=thermal_receipts,
        power_receipts=power_receipts,
        pack_valid=validation.valid,
    )

    final_study = ThermalPowerBudgetStudy(
        policy=policy,
        thermal_receipts=thermal_receipts,
        power_receipts=power_receipts,
        receipt_pack=receipt_pack,
        validation=validation,
    )

    final_validation = validate_thermal_power_budget_pack(final_study)
    if final_validation != validation:
        return ThermalPowerBudgetStudy(
            policy=policy,
            thermal_receipts=thermal_receipts,
            power_receipts=power_receipts,
            receipt_pack=receipt_pack,
            validation=final_validation,
        )
    return final_study


def validate_thermal_power_budget_pack(
    study_obj: ThermalPowerBudgetStudy | Mapping[str, Any],
) -> ThermalPowerValidationReport:
    """Validate deterministic thermal/power budget semantics and receipt consistency."""
    errors = []

    try:
        if isinstance(study_obj, ThermalPowerBudgetStudy):
            study = study_obj
        elif isinstance(study_obj, Mapping):
            policy = _normalize_policy(study_obj["policy"])

            thermal_receipts = []
            for entry in tuple(study_obj.get("thermal_receipts", ())):
                meta = entry.get("metadata")
                thermal_receipts.append(
                    ThermalBudgetReceipt(
                        sample_id=_normalize_sha256_hex(entry["sample_id"], field="thermal_receipt.sample_id"),
                        lane_count=_normalize_int(entry["lane_count"], field="thermal_receipt.lane_count", minimum=0),
                        projected_thermal_units=_normalize_int(
                            entry["projected_thermal_units"], field="thermal_receipt.projected_thermal_units"
                        ),
                        within_thermal_budget=_normalize_bool(
                            entry["within_thermal_budget"], field="thermal_receipt.within_thermal_budget"
                        ),
                        thermal_pressure_score=_normalize_float(
                            entry["thermal_pressure_score"], field="thermal_receipt.thermal_pressure_score"
                        ),
                        metadata=_canonicalize_value(meta if isinstance(meta, Mapping) else {}, field="thermal_receipt.metadata"),
                    )
                )

            power_receipts = []
            for entry in tuple(study_obj.get("power_receipts", ())):
                meta = entry.get("metadata")
                power_receipts.append(
                    PowerBudgetReceipt(
                        sample_id=_normalize_sha256_hex(entry["sample_id"], field="power_receipt.sample_id"),
                        lane_count=_normalize_int(entry["lane_count"], field="power_receipt.lane_count", minimum=0),
                        projected_power_units=_normalize_int(
                            entry["projected_power_units"], field="power_receipt.projected_power_units"
                        ),
                        within_power_budget=_normalize_bool(
                            entry["within_power_budget"], field="power_receipt.within_power_budget"
                        ),
                        power_pressure_score=_normalize_float(
                            entry["power_pressure_score"], field="power_receipt.power_pressure_score"
                        ),
                        metadata=_canonicalize_value(meta if isinstance(meta, Mapping) else {}, field="power_receipt.metadata"),
                    )
                )

            pack_map = study_obj["receipt_pack"]
            validation_map = study_obj.get("validation", {"valid": True, "errors": [], "error_count": 0})
            study = ThermalPowerBudgetStudy(
                policy=policy,
                thermal_receipts=tuple(thermal_receipts),
                power_receipts=tuple(power_receipts),
                receipt_pack=ThermalPowerReceiptPack(
                    policy_hash=_normalize_sha256_hex(pack_map["policy_hash"], field="receipt_pack.policy_hash"),
                    thermal_hash=_normalize_sha256_hex(pack_map["thermal_hash"], field="receipt_pack.thermal_hash"),
                    power_hash=_normalize_sha256_hex(pack_map["power_hash"], field="receipt_pack.power_hash"),
                    pack_valid=_normalize_bool(pack_map["pack_valid"], field="receipt_pack.pack_valid"),
                    receipt_hash=_normalize_sha256_hex(pack_map["receipt_hash"], field="receipt_pack.receipt_hash"),
                ),
                validation=ThermalPowerValidationReport(
                    valid=_normalize_bool(validation_map.get("valid", True), field="validation.valid"),
                    errors=tuple(str(v) for v in tuple(validation_map.get("errors", ()))),
                    error_count=_normalize_int(validation_map.get("error_count", 0), field="validation.error_count", minimum=0),
                ),
            )
        else:
            raise ThermalPowerBudgetValidationError("study_obj must be mapping or ThermalPowerBudgetStudy")
    except Exception as exc:
        return ThermalPowerValidationReport(valid=False, errors=(f"normalization_failed: {exc}",), error_count=1)

    if study.policy.max_thermal_units <= 0:
        errors.append("policy.max_thermal_units must be > 0")
    if study.policy.max_power_units <= 0:
        errors.append("policy.max_power_units must be > 0")
    if study.policy.thermal_mode not in _SUPPORTED_THERMAL_MODES:
        errors.append(f"unsupported policy.thermal_mode: {study.policy.thermal_mode!r}")
    if study.policy.power_mode not in _SUPPORTED_POWER_MODES:
        errors.append(f"unsupported policy.power_mode: {study.policy.power_mode!r}")

    previous_thermal_key = None
    for receipt in study.thermal_receipts:
        if receipt.projected_thermal_units < 0:
            errors.append(f"thermal_receipt[{receipt.sample_id}].projected_thermal_units must be >= 0")
        if math.isnan(receipt.thermal_pressure_score) or math.isinf(receipt.thermal_pressure_score):
            errors.append(f"thermal_receipt[{receipt.sample_id}].thermal_pressure_score must be finite")
        if not (0.0 <= receipt.thermal_pressure_score <= 1.0):
            errors.append(f"thermal_receipt[{receipt.sample_id}].thermal_pressure_score must be within [0, 1]")
        if (receipt.projected_thermal_units <= study.policy.max_thermal_units) != receipt.within_thermal_budget:
            errors.append(f"thermal_receipt[{receipt.sample_id}].within_thermal_budget mismatch")

        expected_pressure = min(1.0, float(receipt.projected_thermal_units) / float(max(1, study.policy.max_thermal_units)))
        if abs(receipt.thermal_pressure_score - expected_pressure) > 1e-12:
            errors.append(f"thermal_receipt[{receipt.sample_id}].thermal_pressure_score mismatch")

        key = (receipt.lane_count, receipt.sample_id)
        if previous_thermal_key is not None and key < previous_thermal_key:
            errors.append("thermal_receipts are not in deterministic ordering")
        previous_thermal_key = key

    previous_power_key = None
    for receipt in study.power_receipts:
        if receipt.projected_power_units < 0:
            errors.append(f"power_receipt[{receipt.sample_id}].projected_power_units must be >= 0")
        if math.isnan(receipt.power_pressure_score) or math.isinf(receipt.power_pressure_score):
            errors.append(f"power_receipt[{receipt.sample_id}].power_pressure_score must be finite")
        if not (0.0 <= receipt.power_pressure_score <= 1.0):
            errors.append(f"power_receipt[{receipt.sample_id}].power_pressure_score must be within [0, 1]")
        if (receipt.projected_power_units <= study.policy.max_power_units) != receipt.within_power_budget:
            errors.append(f"power_receipt[{receipt.sample_id}].within_power_budget mismatch")

        expected_pressure = min(1.0, float(receipt.projected_power_units) / float(max(1, study.policy.max_power_units)))
        if abs(receipt.power_pressure_score - expected_pressure) > 1e-12:
            errors.append(f"power_receipt[{receipt.sample_id}].power_pressure_score mismatch")

        key = (receipt.lane_count, receipt.sample_id)
        if previous_power_key is not None and key < previous_power_key:
            errors.append("power_receipts are not in deterministic ordering")
        previous_power_key = key

    thermal_ids = tuple(receipt.sample_id for receipt in study.thermal_receipts)
    power_ids = tuple(receipt.sample_id for receipt in study.power_receipts)
    if thermal_ids != power_ids:
        errors.append("thermal/power receipt sample_id sets differ")

    expected_pack = _build_receipt_pack(
        policy=study.policy,
        thermal_receipts=tuple(study.thermal_receipts),
        power_receipts=tuple(study.power_receipts),
        pack_valid=not errors,
    )
    if study.receipt_pack.policy_hash != study.policy.stable_hash():
        errors.append("receipt_pack.policy_hash mismatch")
    if study.receipt_pack.thermal_hash != _stable_hash([receipt.stable_hash() for receipt in study.thermal_receipts]):
        errors.append("receipt_pack.thermal_hash mismatch")
    if study.receipt_pack.power_hash != _stable_hash([receipt.stable_hash() for receipt in study.power_receipts]):
        errors.append("receipt_pack.power_hash mismatch")
    if study.receipt_pack.pack_valid != (not errors):
        errors.append("receipt_pack.pack_valid mismatch")
    if study.receipt_pack.receipt_hash != expected_pack.receipt_hash:
        errors.append("receipt_pack.receipt_hash mismatch")

    if _canonical_json(study.to_dict()) != study.to_canonical_json():
        errors.append("canonical ordering mismatch")

    return ThermalPowerValidationReport(valid=not errors, errors=tuple(errors), error_count=len(errors))


def budget_replay_projection(study_obj: ThermalPowerBudgetStudy) -> Dict[str, Any]:
    """Emit deterministic replay-safe thermal/power lineage for hardware validation epochs."""
    payload = {
        "module_version": THERMAL_POWER_BUDGET_RECEIPT_PACK_VERSION,
        "policy_hash": study_obj.receipt_pack.policy_hash,
        "thermal_hash": study_obj.receipt_pack.thermal_hash,
        "power_hash": study_obj.receipt_pack.power_hash,
        "receipt_hash": study_obj.receipt_pack.receipt_hash,
        "pack_hash": study_obj.stable_hash(),
        "thermal_receipt_hashes": [receipt.stable_hash() for receipt in study_obj.thermal_receipts],
        "power_receipt_hashes": [receipt.stable_hash() for receipt in study_obj.power_receipts],
        "pack_valid": study_obj.validation.valid,
    }
    payload["projection_hash"] = _stable_hash(payload)
    return payload
