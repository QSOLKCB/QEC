# SPDX-License-Identifier: MIT
"""v138.2.13 — deterministic persona drift tensor layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

PERSONA_DRIFT_TENSOR_VERSION = "v138.2.13"

SUPPORTED_PERSONA_DRIFT_AXES: Tuple[str, ...] = (
    "lexical_stability",
    "thematic_persistence",
    "rhetorical_motif_recurrence",
    "emotional_arc_retention",
    "tone_drift",
    "callback_continuity",
    "wrapper_divergence",
)


class PersonaDriftTensorValidationError(ValueError):
    """Raised when persona drift tensor input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PersonaDriftTensorValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise PersonaDriftTensorValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise PersonaDriftTensorValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise PersonaDriftTensorValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise PersonaDriftTensorValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersonaDriftTensorValidationError(f"{field} must be a finite float in [0.0, 1.0]")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise PersonaDriftTensorValidationError(f"{field} must be finite")
    if normalized < 0.0 or normalized > 1.0:
        raise PersonaDriftTensorValidationError(f"{field} must be within [0.0, 1.0]")
    return normalized


def _normalize_axis_name(value: Any, *, field: str) -> str:
    axis_name = _normalize_required_text(value, field=field).lower()
    if axis_name not in SUPPORTED_PERSONA_DRIFT_AXES:
        raise PersonaDriftTensorValidationError(
            "metric.axis_name must be one of: " + ", ".join(SUPPORTED_PERSONA_DRIFT_AXES)
        )
    return axis_name


def _normalize_metadata_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PersonaDriftTensorValidationError(f"{field} must be a mapping")
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
    raise PersonaDriftTensorValidationError("canonical_prompt_artifact must provide receipt.prompt_hash")


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
    raise PersonaDriftTensorValidationError("invocation_matrix must provide receipt.matrix_hash")


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


def _metric_sort_key(metric: "PersonaDriftMetric") -> Tuple[str, str, str]:
    return (metric.invocation_id, metric.axis_name, metric.metric_id)


def _ordered_metrics(metrics: Sequence["PersonaDriftMetric"]) -> Tuple["PersonaDriftMetric", ...]:
    return tuple(sorted(metrics, key=_metric_sort_key))


def _extract_metric_mapping(raw_metric: "PersonaDriftMetric | Mapping[str, Any]") -> Dict[str, Any]:
    if isinstance(raw_metric, PersonaDriftMetric):
        return raw_metric.to_dict()
    if isinstance(raw_metric, Mapping):
        return dict(raw_metric)
    raise PersonaDriftTensorValidationError("metric must be PersonaDriftMetric or mapping")


def _normalize_metric(raw_metric: "PersonaDriftMetric | Mapping[str, Any]") -> "PersonaDriftMetric":
    metric_map = _extract_metric_mapping(raw_metric)
    return PersonaDriftMetric(
        metric_id=_normalize_required_text(metric_map.get("metric_id"), field="metric.metric_id"),
        invocation_id=_normalize_required_text(metric_map.get("invocation_id"), field="metric.invocation_id"),
        axis_name=_normalize_axis_name(metric_map.get("axis_name"), field="metric.axis_name"),
        score=_normalize_score(metric_map.get("score"), field="metric.score"),
        metadata=_normalize_metadata_mapping(metric_map.get("metadata", {}), field="metric.metadata"),
    )


def _build_tensor_hash(
    prompt_hash: str,
    matrix_hash: str,
    run_count: int,
    metrics: Sequence["PersonaDriftMetric"],
    aggregate_stability_score: float,
    aggregate_drift_magnitude: float,
) -> str:
    return _stable_hash(
        {
            "prompt_hash": prompt_hash,
            "matrix_hash": matrix_hash,
            "run_count": int(run_count),
            "metrics": [m.to_dict() for m in metrics],
            "aggregate_stability_score": float(aggregate_stability_score),
            "aggregate_drift_magnitude": float(aggregate_drift_magnitude),
        }
    )


def _build_receipt(
    prompt_hash: str,
    matrix_hash: str,
    run_count: int,
    metrics: Sequence["PersonaDriftMetric"],
    *,
    aggregate_stability_score: float,
    aggregate_drift_magnitude: float,
    validation_passed: bool,
) -> "PersonaDriftReceipt":
    tensor_hash = _build_tensor_hash(
        prompt_hash,
        matrix_hash,
        run_count,
        metrics,
        aggregate_stability_score,
        aggregate_drift_magnitude,
    )
    provisional = PersonaDriftReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        tensor_hash=tensor_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return PersonaDriftReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        tensor_hash=provisional.tensor_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


@dataclass(frozen=True)
class PersonaDriftMetric:
    metric_id: str
    invocation_id: str
    axis_name: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "invocation_id": self.invocation_id,
            "axis_name": self.axis_name,
            "score": float(self.score),
            "metadata": _canonicalize_value(dict(self.metadata), field="metric.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PersonaDriftReceipt:
    prompt_hash: str
    matrix_hash: str
    tensor_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "tensor_hash": self.tensor_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "tensor_hash": self.tensor_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class PersonaDriftValidationReport:
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
class PersonaDriftTensor:
    prompt_hash: str
    matrix_hash: str
    run_count: int
    metrics: Tuple[PersonaDriftMetric, ...]
    aggregate_stability_score: float
    aggregate_drift_magnitude: float
    receipt: PersonaDriftReceipt
    validation: PersonaDriftValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "run_count": int(self.run_count),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "aggregate_stability_score": float(self.aggregate_stability_score),
            "aggregate_drift_magnitude": float(self.aggregate_drift_magnitude),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "persona_drift_tensor_version": PERSONA_DRIFT_TENSOR_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_aggregate_stability(metrics: Sequence[PersonaDriftMetric]) -> float:
    if not metrics:
        return 0.0
    total = 0.0
    for metric in metrics:
        total += float(metric.score)
    mean = total / float(len(metrics))
    return min(1.0, max(0.0, float(mean)))


def compute_drift_magnitude(stability_score: float) -> float:
    value = float(stability_score)
    if math.isnan(value) or math.isinf(value):
        raise PersonaDriftTensorValidationError("stability_score must be finite")
    drift = 1.0 - value
    return min(1.0, max(0.0, float(drift)))


def validate_persona_drift_tensor(
    tensor: PersonaDriftTensor | Mapping[str, Any],
    *,
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
) -> PersonaDriftValidationReport:
    errors: list[str] = []

    expected_prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    expected_matrix_hash = _extract_matrix_hash(invocation_matrix)
    matrix_invocation_ids = frozenset(_extract_invocation_ids(invocation_matrix))

    if isinstance(tensor, PersonaDriftTensor):
        prompt_hash = tensor.prompt_hash
        matrix_hash = tensor.matrix_hash
        run_count = int(tensor.run_count)
        metrics = tuple(tensor.metrics)
        aggregate_stability_score = float(tensor.aggregate_stability_score)
        aggregate_drift_magnitude = float(tensor.aggregate_drift_magnitude)
        receipt = tensor.receipt
    elif isinstance(tensor, Mapping):
        prompt_hash_raw = tensor.get("prompt_hash")
        matrix_hash_raw = tensor.get("matrix_hash")
        run_count_raw = tensor.get("run_count", 0)
        agg_stability_raw = tensor.get("aggregate_stability_score", 0.0)
        agg_drift_raw = tensor.get("aggregate_drift_magnitude", 0.0)

        prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
        matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
        run_count = int(run_count_raw) if isinstance(run_count_raw, int) and not isinstance(run_count_raw, bool) else 0

        try:
            aggregate_stability_score = _normalize_score(agg_stability_raw, field="tensor.aggregate_stability_score")
        except PersonaDriftTensorValidationError as exc:
            errors.append(str(exc))
            aggregate_stability_score = 0.0

        try:
            aggregate_drift_magnitude = _normalize_score(agg_drift_raw, field="tensor.aggregate_drift_magnitude")
        except PersonaDriftTensorValidationError as exc:
            errors.append(str(exc))
            aggregate_drift_magnitude = 0.0

        metrics_raw = tensor.get("metrics", ())
        normalized_metrics: list[PersonaDriftMetric] = []
        if isinstance(metrics_raw, Sequence) and not isinstance(metrics_raw, (str, bytes, bytearray)):
            for raw_metric in metrics_raw:
                try:
                    normalized_metrics.append(_normalize_metric(raw_metric))
                except PersonaDriftTensorValidationError as exc:
                    errors.append(str(exc))
        else:
            errors.append("tensor.metrics must be an iterable sequence")
        metrics = tuple(normalized_metrics)

        receipt_raw = tensor.get("receipt", {})
        if isinstance(receipt_raw, Mapping):
            rp_raw = receipt_raw.get("prompt_hash")
            rm_raw = receipt_raw.get("matrix_hash")
            rt_raw = receipt_raw.get("tensor_hash")
            rr_raw = receipt_raw.get("receipt_hash")
            receipt = PersonaDriftReceipt(
                prompt_hash=rp_raw.strip() if isinstance(rp_raw, str) else "",
                matrix_hash=rm_raw.strip() if isinstance(rm_raw, str) else "",
                tensor_hash=rt_raw.strip() if isinstance(rt_raw, str) else "",
                receipt_hash=rr_raw.strip() if isinstance(rr_raw, str) else "",
                validation_passed=bool(receipt_raw.get("validation_passed", False)),
            )
        else:
            receipt = PersonaDriftReceipt(
                prompt_hash="",
                matrix_hash="",
                tensor_hash="",
                receipt_hash="",
                validation_passed=False,
            )
    else:
        return PersonaDriftValidationReport(
            valid=False,
            errors=("tensor must be PersonaDriftTensor or mapping",),
            error_count=1,
        )

    if prompt_hash != expected_prompt_hash:
        errors.append("tensor.prompt_hash mismatch")
    if matrix_hash != expected_matrix_hash:
        errors.append("tensor.matrix_hash mismatch")

    metric_ids = [m.metric_id for m in metrics]
    if len(set(metric_ids)) != len(metric_ids):
        errors.append("metric_id must be unique")

    if run_count <= 0:
        errors.append("tensor.run_count must be > 0")
    if run_count != len(matrix_invocation_ids):
        errors.append("tensor.run_count must equal matrix invocation count")

    covered_invocation_ids = {m.invocation_id for m in metrics}
    missing_invocation_ids = sorted(matrix_invocation_ids - covered_invocation_ids)
    if missing_invocation_ids:
        errors.append("every matrix invocation must have at least one drift metric")

    for metric in metrics:
        metric_id = metric.metric_id.strip() if isinstance(metric.metric_id, str) else ""
        if not metric_id:
            errors.append("metric.metric_id must be a non-empty string")
        if metric.invocation_id not in matrix_invocation_ids:
            errors.append("metric.invocation_id must exist in matrix")
        if metric.axis_name not in SUPPORTED_PERSONA_DRIFT_AXES:
            errors.append("metric.axis_name must be one of: " + ", ".join(SUPPORTED_PERSONA_DRIFT_AXES))
        score = float(metric.score)
        if math.isnan(score) or math.isinf(score):
            errors.append("metric.score must be finite")
        if score < 0.0 or score > 1.0:
            errors.append("metric.score must be within [0.0, 1.0]")

    expected_order = _ordered_metrics(metrics)
    if expected_order != tuple(metrics):
        errors.append("metrics not in deterministic order")

    recomputed_aggregate_stability = compute_aggregate_stability(metrics)
    if abs(float(aggregate_stability_score) - recomputed_aggregate_stability) > 1e-12:
        errors.append("tensor.aggregate_stability_score mismatch")

    recomputed_drift_magnitude = compute_drift_magnitude(aggregate_stability_score)
    if abs(float(aggregate_drift_magnitude) - recomputed_drift_magnitude) > 1e-12:
        errors.append("tensor.aggregate_drift_magnitude mismatch")

    if abs(float(aggregate_drift_magnitude) - (1.0 - float(aggregate_stability_score))) > 1e-12:
        errors.append("aggregate drift law mismatch")

    expected_validation_passed = not bool(errors)
    expected_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        run_count,
        metrics,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_drift_magnitude=aggregate_drift_magnitude,
        validation_passed=expected_validation_passed,
    )

    if receipt.validation_passed != expected_validation_passed:
        errors.append("receipt.validation_passed mismatch")
    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.tensor_hash != expected_receipt.tensor_hash:
        errors.append("receipt.tensor_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return PersonaDriftValidationReport(valid=not deduped_errors, errors=deduped_errors, error_count=len(deduped_errors))


def build_persona_drift_tensor(
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    repeated_run_drift_metric_mappings: Sequence[PersonaDriftMetric | Mapping[str, Any]],
) -> PersonaDriftTensor:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    matrix_hash = _extract_matrix_hash(invocation_matrix)
    run_count = len(_extract_invocation_ids(invocation_matrix))

    validation_errors: list[str] = []
    normalized_metrics: list[PersonaDriftMetric] = []
    for raw_metric in repeated_run_drift_metric_mappings:
        try:
            normalized_metrics.append(_normalize_metric(raw_metric))
        except PersonaDriftTensorValidationError as exc:
            validation_errors.append(str(exc))

    ordered_metrics = _ordered_metrics(normalized_metrics)
    aggregate_stability_score = compute_aggregate_stability(ordered_metrics)
    aggregate_drift_magnitude = compute_drift_magnitude(aggregate_stability_score)

    provisional_report = PersonaDriftValidationReport(
        valid=not validation_errors,
        errors=tuple(dict.fromkeys(validation_errors)),
        error_count=len(tuple(dict.fromkeys(validation_errors))),
    )
    receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        run_count,
        ordered_metrics,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_drift_magnitude=aggregate_drift_magnitude,
        validation_passed=provisional_report.valid,
    )

    tensor = PersonaDriftTensor(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        run_count=run_count,
        metrics=ordered_metrics,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_drift_magnitude=aggregate_drift_magnitude,
        receipt=receipt,
        validation=provisional_report,
    )

    final_report_base = validate_persona_drift_tensor(
        tensor,
        canonical_prompt_artifact=canonical_prompt_artifact,
        invocation_matrix=invocation_matrix,
    )
    merged_errors = tuple(dict.fromkeys((*provisional_report.errors, *final_report_base.errors)))
    final_report = PersonaDriftValidationReport(valid=not merged_errors, errors=merged_errors, error_count=len(merged_errors))
    final_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        run_count,
        ordered_metrics,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_drift_magnitude=aggregate_drift_magnitude,
        validation_passed=final_report.valid,
    )

    return PersonaDriftTensor(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        run_count=run_count,
        metrics=ordered_metrics,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_drift_magnitude=aggregate_drift_magnitude,
        receipt=final_receipt,
        validation=final_report,
    )


def persona_drift_projection(
    tensor_or_parts: PersonaDriftTensor | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(tensor_or_parts, PersonaDriftTensor):
        tensor = tensor_or_parts
    elif isinstance(tensor_or_parts, Mapping):
        prompt_hash_raw = tensor_or_parts.get("prompt_hash")
        matrix_hash_raw = tensor_or_parts.get("matrix_hash")
        run_count_raw = tensor_or_parts.get("run_count", 0)
        prompt_hash = prompt_hash_raw.strip() if isinstance(prompt_hash_raw, str) else ""
        matrix_hash = matrix_hash_raw.strip() if isinstance(matrix_hash_raw, str) else ""
        run_count = int(run_count_raw) if isinstance(run_count_raw, int) and not isinstance(run_count_raw, bool) else 0
        metrics = tuple(_normalize_metric(m) for m in tensor_or_parts.get("metrics", ()))  # type: ignore[arg-type]
        ordered_metrics = _ordered_metrics(metrics)
        aggregate_stability_score = compute_aggregate_stability(ordered_metrics)
        aggregate_drift_magnitude = compute_drift_magnitude(aggregate_stability_score)
        receipt = _build_receipt(
            prompt_hash,
            matrix_hash,
            run_count,
            ordered_metrics,
            aggregate_stability_score=aggregate_stability_score,
            aggregate_drift_magnitude=aggregate_drift_magnitude,
            validation_passed=True,
        )
        tensor = PersonaDriftTensor(
            prompt_hash=prompt_hash,
            matrix_hash=matrix_hash,
            run_count=run_count,
            metrics=ordered_metrics,
            aggregate_stability_score=aggregate_stability_score,
            aggregate_drift_magnitude=aggregate_drift_magnitude,
            receipt=receipt,
            validation=PersonaDriftValidationReport(valid=True, errors=(), error_count=0),
        )
    else:
        raise PersonaDriftTensorValidationError("tensor_or_parts must be PersonaDriftTensor or mapping")

    ordered_axis_names = tuple(dict.fromkeys(m.axis_name for m in tensor.metrics))
    invocation_coverage_count = len({m.invocation_id for m in tensor.metrics})

    return {
        "aggregate_stability_score": float(tensor.aggregate_stability_score),
        "aggregate_drift_magnitude": float(tensor.aggregate_drift_magnitude),
        "ordered_axis_names": list(ordered_axis_names),
        "invocation_coverage_count": int(invocation_coverage_count),
        "tensor_hash": tensor.receipt.tensor_hash,
    }
