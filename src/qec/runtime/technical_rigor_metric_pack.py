# SPDX-License-Identifier: MIT
"""v138.2.12 — deterministic technical rigor metric pack layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

TECHNICAL_RIGOR_METRIC_PACK_VERSION = "v138.2.12"

SUPPORTED_RIGOR_METRIC_NAMES: Tuple[str, ...] = (
    "constraint_coverage",
    "scope_adherence",
    "architecture_fidelity",
    "assumption_explicitness",
    "error_handling",
    "hallucination_risk",
)


class TechnicalRigorMetricValidationError(ValueError):
    """Raised when technical rigor metric input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TechnicalRigorMetricValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise TechnicalRigorMetricValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise TechnicalRigorMetricValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TechnicalRigorMetricValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TechnicalRigorMetricValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TechnicalRigorMetricValidationError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise TechnicalRigorMetricValidationError(f"{field} must be >= 0")
    return normalized


def _normalize_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TechnicalRigorMetricValidationError(f"{field} must be a finite float in [0.0, 1.0]")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise TechnicalRigorMetricValidationError(f"{field} must be finite")
    if normalized < 0.0 or normalized > 1.0:
        raise TechnicalRigorMetricValidationError(f"{field} must be within [0.0, 1.0]")
    return normalized


def _normalize_metric_name(value: Any, *, field: str) -> str:
    name = _normalize_required_text(value, field=field).lower()
    if name not in SUPPORTED_RIGOR_METRIC_NAMES:
        raise TechnicalRigorMetricValidationError(
            "metric.metric_name must be one of: " + ", ".join(SUPPORTED_RIGOR_METRIC_NAMES)
        )
    return name


def _normalize_metadata_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TechnicalRigorMetricValidationError(f"{field} must be a mapping")
    return _canonicalize_value(dict(value), field=field)


def _extract_prompt_hash(canonical_prompt_artifact: Any) -> str:
    if isinstance(canonical_prompt_artifact, Mapping):
        receipt = canonical_prompt_artifact.get("receipt", {})
        if isinstance(receipt, Mapping):
            prompt_hash = receipt.get("prompt_hash")
            if isinstance(prompt_hash, str) and prompt_hash:
                return prompt_hash
        prompt_hash = canonical_prompt_artifact.get("prompt_hash")
        if isinstance(prompt_hash, str) and prompt_hash:
            return prompt_hash
    prompt_receipt = getattr(canonical_prompt_artifact, "receipt", None)
    prompt_hash = getattr(prompt_receipt, "prompt_hash", None)
    if isinstance(prompt_hash, str) and prompt_hash:
        return prompt_hash
    prompt_hash = getattr(canonical_prompt_artifact, "prompt_hash", None)
    if isinstance(prompt_hash, str) and prompt_hash:
        return prompt_hash
    raise TechnicalRigorMetricValidationError("canonical_prompt_artifact must provide receipt.prompt_hash")


def _extract_matrix_hash(invocation_matrix: Any) -> str:
    if isinstance(invocation_matrix, Mapping):
        receipt = invocation_matrix.get("receipt", {})
        if isinstance(receipt, Mapping):
            matrix_hash = receipt.get("matrix_hash")
            if isinstance(matrix_hash, str) and matrix_hash:
                return matrix_hash
        matrix_hash = invocation_matrix.get("matrix_hash")
        if isinstance(matrix_hash, str) and matrix_hash:
            return matrix_hash
    matrix_receipt = getattr(invocation_matrix, "receipt", None)
    matrix_hash = getattr(matrix_receipt, "matrix_hash", None)
    if isinstance(matrix_hash, str) and matrix_hash:
        return matrix_hash
    matrix_hash = getattr(invocation_matrix, "matrix_hash", None)
    if isinstance(matrix_hash, str) and matrix_hash:
        return matrix_hash
    raise TechnicalRigorMetricValidationError("invocation_matrix must provide receipt.matrix_hash")


def _extract_invocation_ids(invocation_matrix: Any) -> Tuple[str, ...]:
    if isinstance(invocation_matrix, Mapping):
        specs = invocation_matrix.get("invocation_specs", ())
        if isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)):
            ids = []
            for spec in specs:
                if isinstance(spec, Mapping):
                    raw_id = spec.get("invocation_id")
                    if isinstance(raw_id, str) and raw_id.strip():
                        ids.append(raw_id.strip())
            if ids:
                return tuple(sorted(dict.fromkeys(ids)))
        records = invocation_matrix.get("records", ())
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            ids = []
            for record in records:
                if isinstance(record, Mapping):
                    raw_id = record.get("invocation_id")
                    if isinstance(raw_id, str) and raw_id.strip():
                        ids.append(raw_id.strip())
            if ids:
                return tuple(sorted(dict.fromkeys(ids)))
    specs_attr = getattr(invocation_matrix, "invocation_specs", None)
    if isinstance(specs_attr, Sequence):
        ids = tuple(
            sorted(
                dict.fromkeys(
                    s.invocation_id.strip() for s in specs_attr if isinstance(getattr(s, "invocation_id", None), str)
                )
            )
        )
        if ids:
            return ids
    records_attr = getattr(invocation_matrix, "records", None)
    if isinstance(records_attr, Sequence):
        ids = tuple(
            sorted(
                dict.fromkeys(
                    r.invocation_id.strip() for r in records_attr if isinstance(getattr(r, "invocation_id", None), str)
                )
            )
        )
        if ids:
            return ids
    return ()


def _metric_sort_key(metric: "TechnicalRigorMetric") -> Tuple[str, str, str]:
    return (metric.invocation_id, metric.metric_name, metric.metric_id)


def _ordered_metrics(metrics: Sequence["TechnicalRigorMetric"]) -> Tuple["TechnicalRigorMetric", ...]:
    return tuple(sorted(metrics, key=_metric_sort_key))


def _extract_metric_mapping(raw_metric: "TechnicalRigorMetric | Mapping[str, Any]") -> Dict[str, Any]:
    if isinstance(raw_metric, TechnicalRigorMetric):
        return raw_metric.to_dict()
    if isinstance(raw_metric, Mapping):
        return dict(raw_metric)
    raise TechnicalRigorMetricValidationError("metric must be TechnicalRigorMetric or mapping")


def _normalize_metric(raw_metric: "TechnicalRigorMetric | Mapping[str, Any]") -> "TechnicalRigorMetric":
    metric_map = _extract_metric_mapping(raw_metric)
    return TechnicalRigorMetric(
        metric_id=_normalize_required_text(metric_map.get("metric_id"), field="metric.metric_id"),
        invocation_id=_normalize_required_text(metric_map.get("invocation_id"), field="metric.invocation_id"),
        metric_name=_normalize_metric_name(metric_map.get("metric_name"), field="metric.metric_name"),
        score=_normalize_score(metric_map.get("score"), field="metric.score"),
        evidence_count=_normalize_non_negative_int(metric_map.get("evidence_count", 0), field="metric.evidence_count"),
        metadata=_normalize_metadata_mapping(metric_map.get("metadata", {}), field="metric.metadata"),
    )


def _build_metric_pack_hash(prompt_hash: str, matrix_hash: str, metrics: Sequence["TechnicalRigorMetric"], aggregate_score: float) -> str:
    return _stable_hash(
        {
            "prompt_hash": prompt_hash,
            "matrix_hash": matrix_hash,
            "metrics": [m.to_dict() for m in metrics],
            "aggregate_score": float(aggregate_score),
        }
    )


def _build_receipt(
    prompt_hash: str,
    matrix_hash: str,
    metrics: Sequence["TechnicalRigorMetric"],
    *,
    aggregate_score: float,
    validation_passed: bool,
) -> "RigorEvaluationReceipt":
    metric_pack_hash = _build_metric_pack_hash(prompt_hash, matrix_hash, metrics, aggregate_score)
    provisional = RigorEvaluationReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        metric_pack_hash=metric_pack_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return RigorEvaluationReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        metric_pack_hash=provisional.metric_pack_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


@dataclass(frozen=True)
class TechnicalRigorMetric:
    metric_id: str
    invocation_id: str
    metric_name: str
    score: float
    evidence_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "invocation_id": self.invocation_id,
            "metric_name": self.metric_name,
            "score": float(self.score),
            "evidence_count": int(self.evidence_count),
            "metadata": _canonicalize_value(dict(self.metadata), field="metric.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RigorEvaluationReceipt:
    prompt_hash: str
    matrix_hash: str
    metric_pack_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "metric_pack_hash": self.metric_pack_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "metric_pack_hash": self.metric_pack_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class RigorValidationReport:
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
class TechnicalRigorMetricPack:
    prompt_hash: str
    matrix_hash: str
    metrics: Tuple[TechnicalRigorMetric, ...]
    aggregate_score: float
    receipt: RigorEvaluationReceipt
    validation: RigorValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "aggregate_score": float(self.aggregate_score),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "metric_pack_version": TECHNICAL_RIGOR_METRIC_PACK_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_aggregate_rigor_score(metrics: Sequence[TechnicalRigorMetric]) -> float:
    if not metrics:
        return 0.0
    total = 0.0
    for metric in metrics:
        total += float(metric.score)
    mean = total / float(len(metrics))
    return min(1.0, max(0.0, float(mean)))


def validate_rigor_metric_pack(
    metric_pack: TechnicalRigorMetricPack | Mapping[str, Any],
    *,
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
) -> RigorValidationReport:
    errors: list[str] = []

    expected_prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    expected_matrix_hash = _extract_matrix_hash(invocation_matrix)
    matrix_invocation_ids = frozenset(_extract_invocation_ids(invocation_matrix))

    if isinstance(metric_pack, TechnicalRigorMetricPack):
        prompt_hash = metric_pack.prompt_hash
        matrix_hash = metric_pack.matrix_hash
        metrics = tuple(metric_pack.metrics)
        aggregate_score = float(metric_pack.aggregate_score)
        receipt = metric_pack.receipt
    elif isinstance(metric_pack, Mapping):
        prompt_hash_raw = metric_pack.get("prompt_hash")
        matrix_hash_raw = metric_pack.get("matrix_hash")
        aggregate_score_raw = metric_pack.get("aggregate_score", 0.0)
        prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
        matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
        try:
            aggregate_score = _normalize_score(aggregate_score_raw, field="metric_pack.aggregate_score")
        except TechnicalRigorMetricValidationError as exc:
            errors.append(str(exc))
            aggregate_score = 0.0

        normalized_metrics: list[TechnicalRigorMetric] = []
        for raw_metric in metric_pack.get("metrics", ()):  # type: ignore[arg-type]
            try:
                normalized_metrics.append(_normalize_metric(raw_metric))
            except TechnicalRigorMetricValidationError as exc:
                errors.append(str(exc))
        metrics = tuple(normalized_metrics)

        receipt_raw = metric_pack.get("receipt", {})
        if isinstance(receipt_raw, Mapping):
            r_ph_raw = receipt_raw.get("prompt_hash")
            r_mh_raw = receipt_raw.get("matrix_hash")
            r_mph_raw = receipt_raw.get("metric_pack_hash")
            r_rh_raw = receipt_raw.get("receipt_hash")
            receipt = RigorEvaluationReceipt(
                prompt_hash=r_ph_raw.strip() if isinstance(r_ph_raw, str) else "",
                matrix_hash=r_mh_raw.strip() if isinstance(r_mh_raw, str) else "",
                metric_pack_hash=r_mph_raw.strip() if isinstance(r_mph_raw, str) else "",
                receipt_hash=r_rh_raw.strip() if isinstance(r_rh_raw, str) else "",
                validation_passed=bool(receipt_raw.get("validation_passed", False)),
            )
        else:
            receipt = RigorEvaluationReceipt(
                prompt_hash="",
                matrix_hash="",
                metric_pack_hash="",
                receipt_hash="",
                validation_passed=False,
            )
    else:
        return RigorValidationReport(valid=False, errors=("metric_pack must be TechnicalRigorMetricPack or mapping",), error_count=1)

    if prompt_hash != expected_prompt_hash:
        errors.append("metric_pack.prompt_hash mismatch")
    if matrix_hash != expected_matrix_hash:
        errors.append("metric_pack.matrix_hash mismatch")

    metric_ids = [m.metric_id for m in metrics]
    if len(set(metric_ids)) != len(metric_ids):
        errors.append("metric_id must be unique")

    covered_invocation_ids = {m.invocation_id for m in metrics}
    missing_invocation_ids = sorted(matrix_invocation_ids - covered_invocation_ids)
    if missing_invocation_ids:
        errors.append("every matrix invocation must have at least one metric")

    for metric in metrics:
        if metric.invocation_id not in matrix_invocation_ids:
            errors.append("metric.invocation_id must exist in matrix")
        if metric.metric_name not in SUPPORTED_RIGOR_METRIC_NAMES:
            errors.append("metric.metric_name must be one of: " + ", ".join(SUPPORTED_RIGOR_METRIC_NAMES))
        if math.isnan(float(metric.score)) or math.isinf(float(metric.score)):
            errors.append("metric.score must be finite")
        if float(metric.score) < 0.0 or float(metric.score) > 1.0:
            errors.append("metric.score must be within [0.0, 1.0]")
        if int(metric.evidence_count) < 0:
            errors.append("metric.evidence_count must be >= 0")

    expected_order = _ordered_metrics(metrics)
    if expected_order != tuple(metrics):
        errors.append("metrics not in deterministic order")

    recomputed_aggregate = compute_aggregate_rigor_score(metrics)
    if abs(float(aggregate_score) - recomputed_aggregate) > 0.0:
        errors.append("metric_pack.aggregate_score mismatch")

    expected_validation_passed = not bool(errors)
    expected_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        metrics,
        aggregate_score=float(aggregate_score),
        validation_passed=expected_validation_passed,
    )

    if receipt.validation_passed != expected_validation_passed:
        errors.append("receipt.validation_passed mismatch")
    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.metric_pack_hash != expected_receipt.metric_pack_hash:
        errors.append("receipt.metric_pack_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return RigorValidationReport(valid=not deduped_errors, errors=deduped_errors, error_count=len(deduped_errors))


def build_technical_rigor_metric_pack(
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    evaluation_metric_mappings: Sequence[TechnicalRigorMetric | Mapping[str, Any]],
) -> TechnicalRigorMetricPack:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    matrix_hash = _extract_matrix_hash(invocation_matrix)

    validation_errors: list[str] = []
    normalized_metrics: list[TechnicalRigorMetric] = []
    for raw_metric in evaluation_metric_mappings:
        try:
            normalized_metrics.append(_normalize_metric(raw_metric))
        except TechnicalRigorMetricValidationError as exc:
            validation_errors.append(str(exc))

    ordered_metrics = _ordered_metrics(normalized_metrics)
    aggregate_score = compute_aggregate_rigor_score(ordered_metrics)

    provisional_report = RigorValidationReport(
        valid=not validation_errors,
        errors=tuple(dict.fromkeys(validation_errors)),
        error_count=len(tuple(dict.fromkeys(validation_errors))),
    )

    receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        ordered_metrics,
        aggregate_score=aggregate_score,
        validation_passed=provisional_report.valid,
    )

    metric_pack = TechnicalRigorMetricPack(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        metrics=ordered_metrics,
        aggregate_score=aggregate_score,
        receipt=receipt,
        validation=provisional_report,
    )

    final_report_base = validate_rigor_metric_pack(
        metric_pack,
        canonical_prompt_artifact=canonical_prompt_artifact,
        invocation_matrix=invocation_matrix,
    )
    merged_errors = tuple(dict.fromkeys((*provisional_report.errors, *final_report_base.errors)))
    final_report = RigorValidationReport(valid=not merged_errors, errors=merged_errors, error_count=len(merged_errors))
    final_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        ordered_metrics,
        aggregate_score=aggregate_score,
        validation_passed=final_report.valid,
    )
    return TechnicalRigorMetricPack(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        metrics=ordered_metrics,
        aggregate_score=aggregate_score,
        receipt=final_receipt,
        validation=final_report,
    )


def rigor_metric_projection(
    metric_pack_or_parts: TechnicalRigorMetricPack | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(metric_pack_or_parts, TechnicalRigorMetricPack):
        pack = metric_pack_or_parts
    elif isinstance(metric_pack_or_parts, Mapping):
        prompt_hash_raw = metric_pack_or_parts.get("prompt_hash")
        matrix_hash_raw = metric_pack_or_parts.get("matrix_hash")
        prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
        matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
        metrics = tuple(_normalize_metric(m) for m in metric_pack_or_parts.get("metrics", ()))  # type: ignore[arg-type]
        ordered_metrics = _ordered_metrics(metrics)
        aggregate_score = compute_aggregate_rigor_score(ordered_metrics)
        receipt = _build_receipt(
            prompt_hash,
            matrix_hash,
            ordered_metrics,
            aggregate_score=aggregate_score,
            validation_passed=True,
        )
        pack = TechnicalRigorMetricPack(
            prompt_hash=prompt_hash,
            matrix_hash=matrix_hash,
            metrics=ordered_metrics,
            aggregate_score=aggregate_score,
            receipt=receipt,
            validation=RigorValidationReport(valid=True, errors=(), error_count=0),
        )
    else:
        raise TechnicalRigorMetricValidationError("metric_pack_or_parts must be TechnicalRigorMetricPack or mapping")

    ordered_metric_names = tuple(dict.fromkeys(m.metric_name for m in pack.metrics))
    invocation_coverage_count = len({m.invocation_id for m in pack.metrics})

    return {
        "aggregate_score": float(pack.aggregate_score),
        "ordered_metric_names": list(ordered_metric_names),
        "invocation_coverage_count": int(invocation_coverage_count),
        "metric_pack_hash": pack.receipt.metric_pack_hash,
    }
