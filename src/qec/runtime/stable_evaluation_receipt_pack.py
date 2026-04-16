# SPDX-License-Identifier: MIT
"""v138.2.15 — deterministic stable evaluation receipt pack layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

STABLE_EVALUATION_RECEIPT_PACK_VERSION = "v138.2.15"


class StableEvaluationReceiptPackValidationError(ValueError):
    """Raised when stable evaluation receipt pack input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_required_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise StableEvaluationReceiptPackValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise StableEvaluationReceiptPackValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StableEvaluationReceiptPackValidationError(f"{field} must be a finite float in [0.0, 1.0]")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise StableEvaluationReceiptPackValidationError(f"{field} must be finite")
    if normalized < 0.0 or normalized > 1.0:
        raise StableEvaluationReceiptPackValidationError(f"{field} must be within [0.0, 1.0]")
    return normalized


def _extract_hash(artifact: Any, *, receipt_field: str, top_level_field: str, artifact_name: str) -> str:
    if isinstance(artifact, Mapping):
        receipt = artifact.get("receipt", {})
        if isinstance(receipt, Mapping):
            value = receipt.get(receipt_field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = artifact.get(top_level_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    receipt_obj = getattr(artifact, "receipt", None)
    value = getattr(receipt_obj, receipt_field, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = getattr(artifact, top_level_field, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise StableEvaluationReceiptPackValidationError(
        f"{artifact_name} must provide receipt.{receipt_field} or {top_level_field}"
    )


def _extract_prompt_hash(canonical_prompt_artifact: Any) -> str:
    return _extract_hash(
        canonical_prompt_artifact,
        receipt_field="prompt_hash",
        top_level_field="prompt_hash",
        artifact_name="canonical_prompt_artifact",
    )


def _extract_matrix_hash(invocation_matrix: Any) -> str:
    return _extract_hash(
        invocation_matrix,
        receipt_field="matrix_hash",
        top_level_field="matrix_hash",
        artifact_name="invocation_matrix",
    )


def _extract_rigor_pack_hash(rigor_metric_pack: Any) -> str:
    return _extract_hash(
        rigor_metric_pack,
        receipt_field="metric_pack_hash",
        top_level_field="rigor_pack_hash",
        artifact_name="rigor_metric_pack",
    )


def _extract_drift_tensor_hash(drift_tensor: Any) -> str:
    return _extract_hash(
        drift_tensor,
        receipt_field="tensor_hash",
        top_level_field="drift_tensor_hash",
        artifact_name="drift_tensor",
    )


def _extract_wrapper_study_hash(wrapper_divergence_study: Any) -> str:
    return _extract_hash(
        wrapper_divergence_study,
        receipt_field="study_hash",
        top_level_field="wrapper_study_hash",
        artifact_name="wrapper_divergence_study",
    )


def _extract_score(artifact: Any, *, field: str, artifact_name: str) -> float:
    if isinstance(artifact, Mapping):
        if field in artifact:
            return _normalize_score(artifact.get(field), field=f"{artifact_name}.{field}")
    value = getattr(artifact, field, None)
    return _normalize_score(value, field=f"{artifact_name}.{field}")


def _build_pack_hash_payload(
    prompt_hash: str,
    matrix_hash: str,
    rigor_pack_hash: str,
    drift_tensor_hash: str,
    wrapper_study_hash: str,
    aggregate_rigor_score: float,
    aggregate_stability_score: float,
    aggregate_divergence_score: float,
) -> Dict[str, Any]:
    return {
        "prompt_hash": prompt_hash,
        "matrix_hash": matrix_hash,
        "rigor_pack_hash": rigor_pack_hash,
        "drift_tensor_hash": drift_tensor_hash,
        "wrapper_study_hash": wrapper_study_hash,
        "aggregate_rigor_score": float(aggregate_rigor_score),
        "aggregate_stability_score": float(aggregate_stability_score),
        "aggregate_divergence_score": float(aggregate_divergence_score),
    }


def _build_receipt(
    prompt_hash: str,
    matrix_hash: str,
    rigor_pack_hash: str,
    drift_tensor_hash: str,
    wrapper_study_hash: str,
    aggregate_rigor_score: float,
    aggregate_stability_score: float,
    aggregate_divergence_score: float,
    *,
    validation_passed: bool,
) -> "StableEvaluationReceipt":
    pack_hash = _stable_hash(
        _build_pack_hash_payload(
            prompt_hash,
            matrix_hash,
            rigor_pack_hash,
            drift_tensor_hash,
            wrapper_study_hash,
            aggregate_rigor_score,
            aggregate_stability_score,
            aggregate_divergence_score,
        )
    )
    provisional = StableEvaluationReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        rigor_pack_hash=rigor_pack_hash,
        drift_tensor_hash=drift_tensor_hash,
        wrapper_study_hash=wrapper_study_hash,
        pack_hash=pack_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return StableEvaluationReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        rigor_pack_hash=provisional.rigor_pack_hash,
        drift_tensor_hash=provisional.drift_tensor_hash,
        wrapper_study_hash=provisional.wrapper_study_hash,
        pack_hash=provisional.pack_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


@dataclass(frozen=True)
class StableEvaluationReceipt:
    prompt_hash: str
    matrix_hash: str
    rigor_pack_hash: str
    drift_tensor_hash: str
    wrapper_study_hash: str
    pack_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "rigor_pack_hash": self.rigor_pack_hash,
            "drift_tensor_hash": self.drift_tensor_hash,
            "wrapper_study_hash": self.wrapper_study_hash,
            "pack_hash": self.pack_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "rigor_pack_hash": self.rigor_pack_hash,
            "drift_tensor_hash": self.drift_tensor_hash,
            "wrapper_study_hash": self.wrapper_study_hash,
            "pack_hash": self.pack_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class StableEvaluationValidationReport:
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
class StableEvaluationReceiptPack:
    prompt_hash: str
    matrix_hash: str
    rigor_pack_hash: str
    drift_tensor_hash: str
    wrapper_study_hash: str
    aggregate_rigor_score: float
    aggregate_stability_score: float
    aggregate_divergence_score: float
    receipt: StableEvaluationReceipt
    validation: StableEvaluationValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "rigor_pack_hash": self.rigor_pack_hash,
            "drift_tensor_hash": self.drift_tensor_hash,
            "wrapper_study_hash": self.wrapper_study_hash,
            "aggregate_rigor_score": float(self.aggregate_rigor_score),
            "aggregate_stability_score": float(self.aggregate_stability_score),
            "aggregate_divergence_score": float(self.aggregate_divergence_score),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "stable_evaluation_receipt_pack_version": STABLE_EVALUATION_RECEIPT_PACK_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_composite_evaluation_score(
    rigor_score: float,
    stability_score: float,
    divergence_score: float,
) -> float:
    rigor = _normalize_score(rigor_score, field="rigor_score")
    stability = _normalize_score(stability_score, field="stability_score")
    divergence = _normalize_score(divergence_score, field="divergence_score")
    composite = (float(rigor) + float(stability) + float(1.0 - divergence)) / 3.0
    return min(1.0, max(0.0, float(composite)))


def validate_stable_evaluation_receipt_pack(
    receipt_pack: StableEvaluationReceiptPack | Mapping[str, Any],
    *,
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    rigor_metric_pack: Any,
    drift_tensor: Any,
    wrapper_divergence_study: Any,
) -> StableEvaluationValidationReport:
    errors: list[str] = []

    expected_prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    expected_matrix_hash = _extract_matrix_hash(invocation_matrix)
    expected_rigor_pack_hash = _extract_rigor_pack_hash(rigor_metric_pack)
    expected_drift_tensor_hash = _extract_drift_tensor_hash(drift_tensor)
    expected_wrapper_study_hash = _extract_wrapper_study_hash(wrapper_divergence_study)

    if isinstance(receipt_pack, StableEvaluationReceiptPack):
        try:
            receipt_pack_mapping = receipt_pack.to_dict()
        except (TypeError, ValueError, StableEvaluationReceiptPackValidationError) as exc:
            errors.append(f"receipt_pack.to_dict() failed: {exc}")
            receipt_pack = {}
        else:
            try:
                recomputed_json = _canonical_json(receipt_pack_mapping)
            except (TypeError, ValueError) as exc:
                errors.append(f"receipt_pack canonical JSON invalid: {exc}")
            else:
                try:
                    canonical_json = receipt_pack.to_canonical_json()
                except (TypeError, ValueError, StableEvaluationReceiptPackValidationError) as exc:
                    errors.append(f"receipt_pack.to_canonical_json() failed: {exc}")
                else:
                    if recomputed_json != canonical_json:
                        errors.append("canonical JSON mismatch")
            receipt_pack = receipt_pack_mapping

    if isinstance(receipt_pack, Mapping):
        try:
            prompt_hash = _normalize_required_hash(receipt_pack.get("prompt_hash"), field="receipt_pack.prompt_hash")
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            prompt_hash = ""
        try:
            matrix_hash = _normalize_required_hash(receipt_pack.get("matrix_hash"), field="receipt_pack.matrix_hash")
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            matrix_hash = ""
        try:
            rigor_pack_hash = _normalize_required_hash(
                receipt_pack.get("rigor_pack_hash"), field="receipt_pack.rigor_pack_hash"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            rigor_pack_hash = ""
        try:
            drift_tensor_hash = _normalize_required_hash(
                receipt_pack.get("drift_tensor_hash"), field="receipt_pack.drift_tensor_hash"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            drift_tensor_hash = ""
        try:
            wrapper_study_hash = _normalize_required_hash(
                receipt_pack.get("wrapper_study_hash"), field="receipt_pack.wrapper_study_hash"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            wrapper_study_hash = ""

        try:
            aggregate_rigor_score = _normalize_score(
                receipt_pack.get("aggregate_rigor_score"), field="receipt_pack.aggregate_rigor_score"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            aggregate_rigor_score = 0.0
        try:
            aggregate_stability_score = _normalize_score(
                receipt_pack.get("aggregate_stability_score"), field="receipt_pack.aggregate_stability_score"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            aggregate_stability_score = 0.0
        try:
            aggregate_divergence_score = _normalize_score(
                receipt_pack.get("aggregate_divergence_score"), field="receipt_pack.aggregate_divergence_score"
            )
        except StableEvaluationReceiptPackValidationError as exc:
            errors.append(str(exc))
            aggregate_divergence_score = 0.0

        receipt_raw = receipt_pack.get("receipt", {})
        if isinstance(receipt_raw, Mapping):
            receipt = StableEvaluationReceipt(
                prompt_hash=receipt_raw.get("prompt_hash", "").strip() if isinstance(receipt_raw.get("prompt_hash"), str) else "",
                matrix_hash=receipt_raw.get("matrix_hash", "").strip() if isinstance(receipt_raw.get("matrix_hash"), str) else "",
                rigor_pack_hash=receipt_raw.get("rigor_pack_hash", "").strip() if isinstance(receipt_raw.get("rigor_pack_hash"), str) else "",
                drift_tensor_hash=receipt_raw.get("drift_tensor_hash", "").strip() if isinstance(receipt_raw.get("drift_tensor_hash"), str) else "",
                wrapper_study_hash=receipt_raw.get("wrapper_study_hash", "").strip() if isinstance(receipt_raw.get("wrapper_study_hash"), str) else "",
                pack_hash=receipt_raw.get("pack_hash", "").strip() if isinstance(receipt_raw.get("pack_hash"), str) else "",
                receipt_hash=receipt_raw.get("receipt_hash", "").strip() if isinstance(receipt_raw.get("receipt_hash"), str) else "",
                validation_passed=bool(receipt_raw.get("validation_passed", False)),
            )
        else:
            receipt = StableEvaluationReceipt(
                prompt_hash="",
                matrix_hash="",
                rigor_pack_hash="",
                drift_tensor_hash="",
                wrapper_study_hash="",
                pack_hash="",
                receipt_hash="",
                validation_passed=False,
            )
    else:
        return StableEvaluationValidationReport(
            valid=False,
            errors=("receipt_pack must be StableEvaluationReceiptPack or mapping",),
            error_count=1,
        )

    if prompt_hash != expected_prompt_hash:
        errors.append("receipt_pack.prompt_hash mismatch")
    if matrix_hash != expected_matrix_hash:
        errors.append("receipt_pack.matrix_hash mismatch")
    if rigor_pack_hash != expected_rigor_pack_hash:
        errors.append("receipt_pack.rigor_pack_hash mismatch")
    if drift_tensor_hash != expected_drift_tensor_hash:
        errors.append("receipt_pack.drift_tensor_hash mismatch")
    if wrapper_study_hash != expected_wrapper_study_hash:
        errors.append("receipt_pack.wrapper_study_hash mismatch")

    expected_aggregate_rigor_score = _extract_score(
        rigor_metric_pack,
        field="aggregate_score",
        artifact_name="rigor_metric_pack",
    )
    expected_aggregate_stability_score = _extract_score(
        drift_tensor,
        field="aggregate_stability_score",
        artifact_name="drift_tensor",
    )
    expected_aggregate_divergence_score = _extract_score(
        wrapper_divergence_study,
        field="aggregate_divergence_score",
        artifact_name="wrapper_divergence_study",
    )

    if abs(float(aggregate_rigor_score) - float(expected_aggregate_rigor_score)) > 1e-12:
        errors.append("receipt_pack.aggregate_rigor_score mismatch")
    if abs(float(aggregate_stability_score) - float(expected_aggregate_stability_score)) > 1e-12:
        errors.append("receipt_pack.aggregate_stability_score mismatch")
    if abs(float(aggregate_divergence_score) - float(expected_aggregate_divergence_score)) > 1e-12:
        errors.append("receipt_pack.aggregate_divergence_score mismatch")

    expected_validation_passed = not bool(errors)
    expected_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        rigor_pack_hash,
        drift_tensor_hash,
        wrapper_study_hash,
        aggregate_rigor_score,
        aggregate_stability_score,
        aggregate_divergence_score,
        validation_passed=expected_validation_passed,
    )

    if receipt.validation_passed != expected_validation_passed:
        errors.append("receipt.validation_passed mismatch")
    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.rigor_pack_hash != rigor_pack_hash:
        errors.append("receipt.rigor_pack_hash mismatch")
    if receipt.drift_tensor_hash != drift_tensor_hash:
        errors.append("receipt.drift_tensor_hash mismatch")
    if receipt.wrapper_study_hash != wrapper_study_hash:
        errors.append("receipt.wrapper_study_hash mismatch")
    if receipt.pack_hash != expected_receipt.pack_hash:
        errors.append("receipt.pack_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return StableEvaluationValidationReport(valid=not deduped_errors, errors=deduped_errors, error_count=len(deduped_errors))


def build_stable_evaluation_receipt_pack(
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    rigor_metric_pack: Any,
    drift_tensor: Any,
    wrapper_divergence_study: Any,
) -> StableEvaluationReceiptPack:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    matrix_hash = _extract_matrix_hash(invocation_matrix)
    rigor_pack_hash = _extract_rigor_pack_hash(rigor_metric_pack)
    drift_tensor_hash = _extract_drift_tensor_hash(drift_tensor)
    wrapper_study_hash = _extract_wrapper_study_hash(wrapper_divergence_study)

    aggregate_rigor_score = _extract_score(rigor_metric_pack, field="aggregate_score", artifact_name="rigor_metric_pack")
    aggregate_stability_score = _extract_score(
        drift_tensor,
        field="aggregate_stability_score",
        artifact_name="drift_tensor",
    )
    aggregate_divergence_score = _extract_score(
        wrapper_divergence_study,
        field="aggregate_divergence_score",
        artifact_name="wrapper_divergence_study",
    )

    provisional_report = StableEvaluationValidationReport(valid=True, errors=(), error_count=0)
    receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        rigor_pack_hash,
        drift_tensor_hash,
        wrapper_study_hash,
        aggregate_rigor_score,
        aggregate_stability_score,
        aggregate_divergence_score,
        validation_passed=provisional_report.valid,
    )

    pack = StableEvaluationReceiptPack(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        rigor_pack_hash=rigor_pack_hash,
        drift_tensor_hash=drift_tensor_hash,
        wrapper_study_hash=wrapper_study_hash,
        aggregate_rigor_score=aggregate_rigor_score,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_divergence_score=aggregate_divergence_score,
        receipt=receipt,
        validation=provisional_report,
    )

    final_report = validate_stable_evaluation_receipt_pack(
        pack,
        canonical_prompt_artifact=canonical_prompt_artifact,
        invocation_matrix=invocation_matrix,
        rigor_metric_pack=rigor_metric_pack,
        drift_tensor=drift_tensor,
        wrapper_divergence_study=wrapper_divergence_study,
    )
    final_receipt = _build_receipt(
        prompt_hash,
        matrix_hash,
        rigor_pack_hash,
        drift_tensor_hash,
        wrapper_study_hash,
        aggregate_rigor_score,
        aggregate_stability_score,
        aggregate_divergence_score,
        validation_passed=final_report.valid,
    )
    return StableEvaluationReceiptPack(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        rigor_pack_hash=rigor_pack_hash,
        drift_tensor_hash=drift_tensor_hash,
        wrapper_study_hash=wrapper_study_hash,
        aggregate_rigor_score=aggregate_rigor_score,
        aggregate_stability_score=aggregate_stability_score,
        aggregate_divergence_score=aggregate_divergence_score,
        receipt=final_receipt,
        validation=final_report,
    )


def stable_evaluation_projection(
    receipt_pack_or_mapping: StableEvaluationReceiptPack | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(receipt_pack_or_mapping, StableEvaluationReceiptPack):
        pack = receipt_pack_or_mapping
    elif isinstance(receipt_pack_or_mapping, Mapping):
        prompt_hash = _normalize_required_hash(receipt_pack_or_mapping.get("prompt_hash"), field="prompt_hash")
        matrix_hash = _normalize_required_hash(receipt_pack_or_mapping.get("matrix_hash"), field="matrix_hash")
        rigor_pack_hash = _normalize_required_hash(receipt_pack_or_mapping.get("rigor_pack_hash"), field="rigor_pack_hash")
        drift_tensor_hash = _normalize_required_hash(
            receipt_pack_or_mapping.get("drift_tensor_hash"), field="drift_tensor_hash"
        )
        wrapper_study_hash = _normalize_required_hash(
            receipt_pack_or_mapping.get("wrapper_study_hash"), field="wrapper_study_hash"
        )
        aggregate_rigor_score = _normalize_score(
            receipt_pack_or_mapping.get("aggregate_rigor_score"), field="aggregate_rigor_score"
        )
        aggregate_stability_score = _normalize_score(
            receipt_pack_or_mapping.get("aggregate_stability_score"), field="aggregate_stability_score"
        )
        aggregate_divergence_score = _normalize_score(
            receipt_pack_or_mapping.get("aggregate_divergence_score"), field="aggregate_divergence_score"
        )
        receipt = _build_receipt(
            prompt_hash,
            matrix_hash,
            rigor_pack_hash,
            drift_tensor_hash,
            wrapper_study_hash,
            aggregate_rigor_score,
            aggregate_stability_score,
            aggregate_divergence_score,
            validation_passed=True,
        )
        pack = StableEvaluationReceiptPack(
            prompt_hash=prompt_hash,
            matrix_hash=matrix_hash,
            rigor_pack_hash=rigor_pack_hash,
            drift_tensor_hash=drift_tensor_hash,
            wrapper_study_hash=wrapper_study_hash,
            aggregate_rigor_score=aggregate_rigor_score,
            aggregate_stability_score=aggregate_stability_score,
            aggregate_divergence_score=aggregate_divergence_score,
            receipt=receipt,
            validation=StableEvaluationValidationReport(valid=True, errors=(), error_count=0),
        )
    else:
        raise StableEvaluationReceiptPackValidationError(
            "receipt_pack_or_mapping must be StableEvaluationReceiptPack or mapping"
        )

    return {
        "aggregate_rigor_score": float(pack.aggregate_rigor_score),
        "aggregate_stability_score": float(pack.aggregate_stability_score),
        "aggregate_divergence_score": float(pack.aggregate_divergence_score),
        "pack_hash": pack.receipt.pack_hash,
        "receipt_hash": pack.receipt.receipt_hash,
    }
