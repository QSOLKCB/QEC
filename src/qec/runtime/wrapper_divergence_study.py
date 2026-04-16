# SPDX-License-Identifier: MIT
"""v138.2.14 — deterministic wrapper divergence study layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

WRAPPER_DIVERGENCE_STUDY_VERSION = "v138.2.14"

SUPPORTED_WRAPPER_DIVERGENCE_AXES: Tuple[str, ...] = (
    "route_output_divergence",
    "wrapper_prompt_shift",
    "safety_layer_variance",
    "response_length_delta",
    "constraint_break_delta",
    "tone_shift",
    "rigor_score_delta",
    "drift_tensor_delta",
)


class WrapperDivergenceValidationError(ValueError):
    """Raised when wrapper divergence study input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise WrapperDivergenceValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise WrapperDivergenceValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise WrapperDivergenceValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise WrapperDivergenceValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise WrapperDivergenceValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WrapperDivergenceValidationError(f"{field} must be a finite float in [0.0, 1.0]")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise WrapperDivergenceValidationError(f"{field} must be finite")
    if normalized < 0.0 or normalized > 1.0:
        raise WrapperDivergenceValidationError(f"{field} must be within [0.0, 1.0]")
    return normalized


def _normalize_axis_name(value: Any, *, field: str) -> str:
    axis_name = _normalize_required_text(value, field=field).lower()
    if axis_name not in SUPPORTED_WRAPPER_DIVERGENCE_AXES:
        raise WrapperDivergenceValidationError(
            "metric.axis_name must be one of: " + ", ".join(SUPPORTED_WRAPPER_DIVERGENCE_AXES)
        )
    return axis_name


def _normalize_metadata_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WrapperDivergenceValidationError(f"{field} must be a mapping")
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
    raise WrapperDivergenceValidationError("canonical_prompt_artifact must provide receipt.prompt_hash")


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
    raise WrapperDivergenceValidationError("invocation_matrix must provide receipt.matrix_hash")


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


def _metric_sort_key(metric: "WrapperDivergenceMetric") -> Tuple[str, str, str, str]:
    return (metric.primary_invocation_id, metric.comparison_invocation_id, metric.axis_name, metric.metric_id)


def _ordered_metrics(metrics: Sequence["WrapperDivergenceMetric"]) -> Tuple["WrapperDivergenceMetric", ...]:
    return tuple(sorted(metrics, key=_metric_sort_key))


def _extract_metric_mapping(raw_metric: "WrapperDivergenceMetric | Mapping[str, Any]") -> Dict[str, Any]:
    if isinstance(raw_metric, WrapperDivergenceMetric):
        return raw_metric.to_dict()
    if isinstance(raw_metric, Mapping):
        return dict(raw_metric)
    raise WrapperDivergenceValidationError("metric must be WrapperDivergenceMetric or mapping")


def _normalize_metric(raw_metric: "WrapperDivergenceMetric | Mapping[str, Any]") -> "WrapperDivergenceMetric":
    metric_map = _extract_metric_mapping(raw_metric)
    return WrapperDivergenceMetric(
        metric_id=_normalize_required_text(metric_map.get("metric_id"), field="metric.metric_id"),
        primary_invocation_id=_normalize_required_text(
            metric_map.get("primary_invocation_id"), field="metric.primary_invocation_id"
        ),
        comparison_invocation_id=_normalize_required_text(
            metric_map.get("comparison_invocation_id"), field="metric.comparison_invocation_id"
        ),
        axis_name=_normalize_axis_name(metric_map.get("axis_name"), field="metric.axis_name"),
        score=_normalize_score(metric_map.get("score"), field="metric.score"),
        metadata=_normalize_metadata_mapping(metric_map.get("metadata", {}), field="metric.metadata"),
    )


def _comparison_count(metrics: Sequence["WrapperDivergenceMetric"]) -> int:
    pairs = {(m.primary_invocation_id, m.comparison_invocation_id) for m in metrics}
    return int(len(pairs))


def _build_study_hash(
    prompt_hash: str,
    matrix_hash: str,
    comparison_count: int,
    metrics: Sequence["WrapperDivergenceMetric"],
    aggregate_divergence_score: float,
) -> str:
    return _stable_hash(
        {
            "prompt_hash": prompt_hash,
            "matrix_hash": matrix_hash,
            "comparison_count": int(comparison_count),
            "metrics": [m.to_dict() for m in metrics],
            "aggregate_divergence_score": float(aggregate_divergence_score),
        }
    )


def _build_receipt(
    prompt_hash: str,
    matrix_hash: str,
    comparison_count: int,
    metrics: Sequence["WrapperDivergenceMetric"],
    *,
    aggregate_divergence_score: float,
    validation_passed: bool,
) -> "WrapperDivergenceReceipt":
    study_hash = _build_study_hash(
        prompt_hash,
        matrix_hash,
        comparison_count,
        metrics,
        aggregate_divergence_score,
    )
    provisional = WrapperDivergenceReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        study_hash=study_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return WrapperDivergenceReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        study_hash=provisional.study_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


@dataclass(frozen=True)
class WrapperDivergenceMetric:
    metric_id: str
    primary_invocation_id: str
    comparison_invocation_id: str
    axis_name: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "primary_invocation_id": self.primary_invocation_id,
            "comparison_invocation_id": self.comparison_invocation_id,
            "axis_name": self.axis_name,
            "score": float(self.score),
            "metadata": _canonicalize_value(dict(self.metadata), field="metric.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class WrapperDivergenceReceipt:
    prompt_hash: str
    matrix_hash: str
    study_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "study_hash": self.study_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "study_hash": self.study_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class WrapperDivergenceValidationReport:
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
class WrapperDivergenceStudy:
    prompt_hash: str
    matrix_hash: str
    comparison_count: int
    metrics: Tuple[WrapperDivergenceMetric, ...]
    aggregate_divergence_score: float
    receipt: WrapperDivergenceReceipt
    validation: WrapperDivergenceValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "comparison_count": int(self.comparison_count),
            "metrics": [m.to_dict() for m in self.metrics],
            "aggregate_divergence_score": float(self.aggregate_divergence_score),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "wrapper_divergence_study_version": WRAPPER_DIVERGENCE_STUDY_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_aggregate_divergence(metrics: Sequence[WrapperDivergenceMetric]) -> float:
    if not metrics:
        return 0.0
    total = 0.0
    for metric in metrics:
        total += float(metric.score)
    mean = total / float(len(metrics))
    return min(1.0, max(0.0, float(mean)))


def validate_wrapper_divergence_study(
    study: WrapperDivergenceStudy | Mapping[str, Any],
    *,
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
) -> WrapperDivergenceValidationReport:
    errors: list[str] = []

    expected_prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    expected_matrix_hash = _extract_matrix_hash(invocation_matrix)
    matrix_invocation_ids = frozenset(_extract_invocation_ids(invocation_matrix))

    if isinstance(study, WrapperDivergenceStudy):
        prompt_hash = study.prompt_hash
        matrix_hash = study.matrix_hash
        comparison_count = int(study.comparison_count)
        metrics = tuple(study.metrics)
        aggregate_divergence_score = float(study.aggregate_divergence_score)
        receipt = study.receipt
    elif isinstance(study, Mapping):
        prompt_hash_raw = study.get("prompt_hash")
        matrix_hash_raw = study.get("matrix_hash")
        comparison_count_raw = study.get("comparison_count", 0)
        aggregate_raw = study.get("aggregate_divergence_score", 0.0)

        prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
        matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
        comparison_count = (
            int(comparison_count_raw)
            if isinstance(comparison_count_raw, int) and not isinstance(comparison_count_raw, bool)
            else 0
        )

        try:
            aggregate_divergence_score = _normalize_score(aggregate_raw, field="study.aggregate_divergence_score")
        except WrapperDivergenceValidationError as exc:
            errors.append(str(exc))
            aggregate_divergence_score = 0.0

        metrics_raw = study.get("metrics", ())
        normalized_metrics: list[WrapperDivergenceMetric] = []
        if isinstance(metrics_raw, Sequence) and not isinstance(metrics_raw, (str, bytes, bytearray)):
            for raw_metric in metrics_raw:
                try:
                    normalized_metrics.append(_normalize_metric(raw_metric))
                except WrapperDivergenceValidationError as exc:
                    errors.append(str(exc))
        else:
            errors.append("study.metrics must be an iterable sequence")
        metrics = tuple(normalized_metrics)

        receipt_raw = study.get("receipt", {})
        if isinstance(receipt_raw, Mapping):
            rp_raw = receipt_raw.get("prompt_hash")
            rm_raw = receipt_raw.get("matrix_hash")
            rs_raw = receipt_raw.get("study_hash")
            rr_raw = receipt_raw.get("receipt_hash")
            receipt = WrapperDivergenceReceipt(
                prompt_hash=rp_raw.strip() if isinstance(rp_raw, str) else "",
                matrix_hash=rm_raw.strip() if isinstance(rm_raw, str) else "",
                study_hash=rs_raw.strip() if isinstance(rs_raw, str) else "",
                receipt_hash=rr_raw.strip() if isinstance(rr_raw, str) else "",
                validation_passed=bool(receipt_raw.get("validation_passed", False)),
            )
        else:
            receipt = WrapperDivergenceReceipt(
                prompt_hash="",
                matrix_hash="",
                study_hash="",
                receipt_hash="",
                validation_passed=False,
            )
    else:
        return WrapperDivergenceValidationReport(
            valid=False,
            errors=("study must be WrapperDivergenceStudy or mapping",),
            error_count=1,
        )

    if prompt_hash != expected_prompt_hash:
        errors.append("study.prompt_hash mismatch")
    if matrix_hash != expected_matrix_hash:
        errors.append("study.matrix_hash mismatch")

    metric_ids = [m.metric_id for m in metrics]
    if len(set(metric_ids)) != len(metric_ids):
        errors.append("metric_id must be unique")

    for metric in metrics:
        if metric.primary_invocation_id not in matrix_invocation_ids:
            errors.append("metric.primary_invocation_id must exist in matrix")
        if metric.comparison_invocation_id not in matrix_invocation_ids:
            errors.append("metric.comparison_invocation_id must exist in matrix")
        if metric.primary_invocation_id == metric.comparison_invocation_id:
            errors.append("metric.primary_invocation_id must differ from metric.comparison_invocation_id")
        if metric.axis_name not in SUPPORTED_WRAPPER_DIVERGENCE_AXES:
            errors.append("metric.axis_name must be one of: " + ", ".join(SUPPORTED_WRAPPER_DIVERGENCE_AXES))
        score = float(metric.score)
        if math.isnan(score) or math.isinf(score):
            errors.append("metric.score must be finite")
        if score < 0.0 or score > 1.0:
            errors.append("metric.score must be within [0.0, 1.0]")

    expected_order = _ordered_metrics(metrics)
    if expected_order != tuple(metrics):
        errors.append("metrics not in deterministic order")

    expected_comparison_count = _comparison_count(metrics)
    if comparison_count <= 0:
        errors.append("study.comparison_count must be > 0")
    if comparison_count != expected_comparison_count:
        errors.append("study.comparison_count mismatch")

    recomputed_aggregate = compute_aggregate_divergence(metrics)
    if abs(float(aggregate_divergence_score) - recomputed_aggregate) > 1e-12:
        errors.append("study.aggregate_divergence_score mismatch")

    expected_validation_passed = not bool(errors)
    expected_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        comparison_count,
        metrics,
        aggregate_divergence_score=float(aggregate_divergence_score),
        validation_passed=expected_validation_passed,
    )

    if receipt.validation_passed != expected_validation_passed:
        errors.append("receipt.validation_passed mismatch")
    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.study_hash != expected_receipt.study_hash:
        errors.append("receipt.study_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return WrapperDivergenceValidationReport(valid=not deduped_errors, errors=deduped_errors, error_count=len(deduped_errors))


def build_wrapper_divergence_study(
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    divergence_metric_mappings: Sequence[WrapperDivergenceMetric | Mapping[str, Any]],
) -> WrapperDivergenceStudy:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    matrix_hash = _extract_matrix_hash(invocation_matrix)

    validation_errors: list[str] = []
    normalized_metrics: list[WrapperDivergenceMetric] = []
    for raw_metric in divergence_metric_mappings:
        try:
            normalized_metrics.append(_normalize_metric(raw_metric))
        except WrapperDivergenceValidationError as exc:
            validation_errors.append(str(exc))

    ordered_metrics = _ordered_metrics(normalized_metrics)
    comparison_count = _comparison_count(ordered_metrics)
    aggregate_divergence_score = compute_aggregate_divergence(ordered_metrics)

    provisional_report = WrapperDivergenceValidationReport(
        valid=not validation_errors,
        errors=tuple(dict.fromkeys(validation_errors)),
        error_count=len(tuple(dict.fromkeys(validation_errors))),
    )

    receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        comparison_count,
        ordered_metrics,
        aggregate_divergence_score=aggregate_divergence_score,
        validation_passed=provisional_report.valid,
    )

    study = WrapperDivergenceStudy(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        comparison_count=comparison_count,
        metrics=ordered_metrics,
        aggregate_divergence_score=aggregate_divergence_score,
        receipt=receipt,
        validation=provisional_report,
    )

    report = validate_wrapper_divergence_study(
        study,
        canonical_prompt_artifact=canonical_prompt_artifact,
        invocation_matrix=invocation_matrix,
    )

    merged_errors = tuple(dict.fromkeys([*validation_errors, *report.errors]))
    final_report = WrapperDivergenceValidationReport(
        valid=not merged_errors,
        errors=merged_errors,
        error_count=len(merged_errors),
    )

    final_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        comparison_count,
        ordered_metrics,
        aggregate_divergence_score=aggregate_divergence_score,
        validation_passed=final_report.valid,
    )

    return WrapperDivergenceStudy(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        comparison_count=comparison_count,
        metrics=ordered_metrics,
        aggregate_divergence_score=aggregate_divergence_score,
        receipt=final_receipt,
        validation=final_report,
    )


def wrapper_divergence_projection(study_or_mapping: WrapperDivergenceStudy | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(study_or_mapping, WrapperDivergenceStudy):
        study = study_or_mapping
        ordered_axis_names = tuple(dict.fromkeys(m.axis_name for m in study.metrics))
        return {
            "aggregate_divergence_score": float(study.aggregate_divergence_score),
            "ordered_axis_names": list(ordered_axis_names),
            "comparison_count": int(study.comparison_count),
            "study_hash": study.receipt.study_hash,
        }

    metrics_raw = study_or_mapping.get("metrics", ())
    normalized_metrics: list[WrapperDivergenceMetric] = []
    if isinstance(metrics_raw, Sequence) and not isinstance(metrics_raw, (str, bytes, bytearray)):
        for metric in metrics_raw:
            normalized_metrics.append(_normalize_metric(metric))
    ordered_metrics = _ordered_metrics(normalized_metrics)
    ordered_axis_names = tuple(dict.fromkeys(m.axis_name for m in ordered_metrics))

    prompt_hash_raw = study_or_mapping.get("prompt_hash")
    matrix_hash_raw = study_or_mapping.get("matrix_hash")
    prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
    matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
    comparison_count = _comparison_count(ordered_metrics)
    aggregate_divergence_score = compute_aggregate_divergence(ordered_metrics)
    study_hash = _build_study_hash(
        prompt_hash,
        matrix_hash,
        comparison_count,
        ordered_metrics,
        aggregate_divergence_score,
    )

    return {
        "aggregate_divergence_score": float(aggregate_divergence_score),
        "ordered_axis_names": list(ordered_axis_names),
        "comparison_count": int(comparison_count),
        "study_hash": study_hash,
    }
