# SPDX-License-Identifier: MIT
"""v138.2.16 — deterministic Zenodo dataset export layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import platform
from typing import Any, Dict, Mapping, Tuple

ZENODO_DATASET_EXPORT_LAYER_VERSION = "v138.2.16"

DEFAULT_ZENODO_KEYWORDS: Tuple[str, ...] = (
    "quantum error correction",
    "determinism",
    "comparative llm evaluation",
    "wrapper divergence",
    "replay-safe systems",
)

RESERVED_REPRODUCIBILITY_METADATA_KEYS: Tuple[str, ...] = (
    "hash_lineage",
    "canonical_manifest_checksum",
    "python_version",
    "module_version",
    "reproducibility_note",
)


class ZenodoDatasetExportValidationError(ValueError):
    """Raised when Zenodo dataset export input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ZenodoDatasetExportValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise ZenodoDatasetExportValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ZenodoDatasetExportValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _validate_reproducibility_metadata_overrides(extra_metadata: Mapping[str, Any]) -> None:
    """Reject caller metadata that attempts to override generated lineage fields."""
    collisions = sorted(
        key for key in RESERVED_REPRODUCIBILITY_METADATA_KEYS if key in extra_metadata
    )
    if collisions:
        collision_list = ", ".join(collisions)
        raise ZenodoDatasetExportValidationError(
            "reproducibility_metadata contains reserved generated keys: "
            f"{collision_list}"
        )


def _normalize_required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ZenodoDatasetExportValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ZenodoDatasetExportValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_keywords(keywords: Any) -> Tuple[str, ...]:
    if keywords is None:
        raw_keywords = DEFAULT_ZENODO_KEYWORDS
    elif isinstance(keywords, str):
        raw_keywords = (keywords,)
    elif isinstance(keywords, (tuple, list)):
        raw_keywords = tuple(keywords)
    else:
        raise ZenodoDatasetExportValidationError("keywords must be a string or sequence of strings")

    normalized = []
    for raw in raw_keywords:
        if not isinstance(raw, str):
            raise ZenodoDatasetExportValidationError("keywords entries must be strings")
        text = raw.strip().lower()
        if not text:
            raise ZenodoDatasetExportValidationError("keywords entries must be non-empty")
        normalized.append(text)

    return tuple(sorted(dict.fromkeys(normalized)))


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
    raise ZenodoDatasetExportValidationError(
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


def _extract_receipt_pack_hash(stable_evaluation_receipt_pack: Any) -> str:
    return _extract_hash(
        stable_evaluation_receipt_pack,
        receipt_field="pack_hash",
        top_level_field="pack_hash",
        artifact_name="stable_evaluation_receipt_pack",
    )


def _extract_artifact_valid(artifact: Any) -> bool:
    if isinstance(artifact, Mapping):
        validation = artifact.get("validation")
        if isinstance(validation, Mapping):
            return bool(validation.get("valid", True))
    validation_obj = getattr(artifact, "validation", None)
    if validation_obj is not None:
        valid_flag = getattr(validation_obj, "valid", None)
        if valid_flag is not None:
            return bool(valid_flag)
    return True


def _module_version() -> str:
    try:
        return importlib_metadata.version("qec")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _manifest_checksum_payload(
    dataset_id: str,
    title: str,
    version_tag: str,
    author: str,
    affiliation: str,
    keywords: Tuple[str, ...],
    methodology_notes: str,
    reproducibility_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "title": title,
        "version_tag": version_tag,
        "author": author,
        "affiliation": affiliation,
        "keywords": list(keywords),
        "methodology_notes": methodology_notes,
        "reproducibility_metadata": _canonicalize_value(dict(reproducibility_metadata), field="reproducibility_metadata"),
    }


def _bundle_hash_payload(bundle: "ZenodoDatasetExportBundle") -> Dict[str, Any]:
    manifest_dict = bundle.manifest.to_dict()
    manifest_dict["export_hash"] = ""
    return {
        "manifest": manifest_dict,
        "prompt_artifact_hash": bundle.prompt_artifact_hash,
        "invocation_matrix_hash": bundle.invocation_matrix_hash,
        "rigor_metric_pack_hash": bundle.rigor_metric_pack_hash,
        "drift_tensor_hash": bundle.drift_tensor_hash,
        "wrapper_study_hash": bundle.wrapper_study_hash,
        "stable_receipt_pack_hash": bundle.stable_receipt_pack_hash,
    }


def _build_receipt(
    prompt_hash: str,
    matrix_hash: str,
    receipt_pack_hash: str,
    bundle_hash: str,
    *,
    validation_passed: bool,
) -> "ZenodoExportReceipt":
    provisional = ZenodoExportReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        receipt_pack_hash=receipt_pack_hash,
        bundle_hash=bundle_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return ZenodoExportReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        receipt_pack_hash=provisional.receipt_pack_hash,
        bundle_hash=provisional.bundle_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


@dataclass(frozen=True)
class ZenodoDatasetManifest:
    dataset_id: str
    title: str
    version_tag: str
    author: str
    affiliation: str
    keywords: Tuple[str, ...]
    methodology_notes: str
    reproducibility_metadata: Dict[str, Any]
    export_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "version_tag": self.version_tag,
            "author": self.author,
            "affiliation": self.affiliation,
            "keywords": list(self.keywords),
            "methodology_notes": self.methodology_notes,
            "reproducibility_metadata": _canonicalize_value(
                dict(self.reproducibility_metadata), field="manifest.reproducibility_metadata"
            ),
            "export_hash": self.export_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ZenodoExportReceipt:
    prompt_hash: str
    matrix_hash: str
    receipt_pack_hash: str
    bundle_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "receipt_pack_hash": self.receipt_pack_hash,
            "bundle_hash": self.bundle_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "receipt_pack_hash": self.receipt_pack_hash,
            "bundle_hash": self.bundle_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ZenodoExportValidationReport:
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
class ZenodoDatasetExportBundle:
    manifest: ZenodoDatasetManifest
    prompt_artifact_hash: str
    invocation_matrix_hash: str
    rigor_metric_pack_hash: str
    drift_tensor_hash: str
    wrapper_study_hash: str
    stable_receipt_pack_hash: str
    receipt: ZenodoExportReceipt
    validation: ZenodoExportValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "prompt_artifact_hash": self.prompt_artifact_hash,
            "invocation_matrix_hash": self.invocation_matrix_hash,
            "rigor_metric_pack_hash": self.rigor_metric_pack_hash,
            "drift_tensor_hash": self.drift_tensor_hash,
            "wrapper_study_hash": self.wrapper_study_hash,
            "stable_receipt_pack_hash": self.stable_receipt_pack_hash,
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "zenodo_dataset_export_layer_version": ZENODO_DATASET_EXPORT_LAYER_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def build_reproducibility_metadata(
    *,
    prompt_hash: str,
    matrix_hash: str,
    rigor_metric_pack_hash: str,
    drift_tensor_hash: str,
    wrapper_study_hash: str,
    stable_receipt_pack_hash: str,
    canonical_manifest_payload: Mapping[str, Any] | None = None,
    reproducibility_note: str = "Deterministic Zenodo export bundle generated with canonical JSON and stable SHA-256 lineage.",
) -> Dict[str, Any]:
    if canonical_manifest_payload is None:
        canonical_manifest_payload = {}
    checksum = _stable_hash(_canonicalize_value(dict(canonical_manifest_payload), field="canonical_manifest_payload"))
    lineage = {
        "prompt_hash": _normalize_required_text(prompt_hash, field="prompt_hash"),
        "matrix_hash": _normalize_required_text(matrix_hash, field="matrix_hash"),
        "rigor_metric_pack_hash": _normalize_required_text(rigor_metric_pack_hash, field="rigor_metric_pack_hash"),
        "drift_tensor_hash": _normalize_required_text(drift_tensor_hash, field="drift_tensor_hash"),
        "wrapper_study_hash": _normalize_required_text(wrapper_study_hash, field="wrapper_study_hash"),
        "stable_receipt_pack_hash": _normalize_required_text(stable_receipt_pack_hash, field="stable_receipt_pack_hash"),
    }
    return {
        "python_version": platform.python_version(),
        "module_version": _module_version(),
        "hash_lineage": lineage,
        "canonical_manifest_checksum": checksum,
        "reproducibility_note": _normalize_required_text(reproducibility_note, field="reproducibility_note"),
    }


def validate_zenodo_dataset_export_bundle(
    bundle: ZenodoDatasetExportBundle | Mapping[str, Any],
    *,
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    rigor_metric_pack: Any,
    drift_tensor: Any,
    wrapper_divergence_study: Any,
    stable_evaluation_receipt_pack: Any,
) -> ZenodoExportValidationReport:
    errors: list[str] = []

    expected_prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    expected_matrix_hash = _extract_matrix_hash(invocation_matrix)
    expected_rigor_hash = _extract_rigor_pack_hash(rigor_metric_pack)
    expected_drift_hash = _extract_drift_tensor_hash(drift_tensor)
    expected_wrapper_hash = _extract_wrapper_study_hash(wrapper_divergence_study)
    expected_receipt_pack_hash = _extract_receipt_pack_hash(stable_evaluation_receipt_pack)

    for name, artifact in (
        ("invocation_matrix", invocation_matrix),
        ("rigor_metric_pack", rigor_metric_pack),
        ("drift_tensor", drift_tensor),
        ("wrapper_divergence_study", wrapper_divergence_study),
        ("stable_evaluation_receipt_pack", stable_evaluation_receipt_pack),
    ):
        if not _extract_artifact_valid(artifact):
            errors.append(f"{name}.validation.valid is False")

    if isinstance(bundle, ZenodoDatasetExportBundle):
        try:
            payload = bundle.to_dict()
        except (TypeError, ValueError) as exc:
            errors.append(f"bundle.to_dict() failed: {exc}")
            payload = {}
        else:
            payload_canonical_json: str | None = None
            bundle_canonical_json: str | None = None
            try:
                payload_canonical_json = _canonical_json(payload)
            except (TypeError, ValueError) as exc:
                errors.append(f"_canonical_json(payload) failed: {exc}")
            try:
                bundle_canonical_json = bundle.to_canonical_json()
            except (TypeError, ValueError) as exc:
                errors.append(f"bundle.to_canonical_json() failed: {exc}")
            if (
                payload_canonical_json is not None
                and bundle_canonical_json is not None
                and payload_canonical_json != bundle_canonical_json
            ):
                errors.append("canonical JSON mismatch")
    elif isinstance(bundle, Mapping):
        payload = dict(bundle)
    else:
        return ZenodoExportValidationReport(
            valid=False,
            errors=("bundle must be ZenodoDatasetExportBundle or mapping",),
            error_count=1,
        )

    manifest_raw = payload.get("manifest", {})
    receipt_raw = payload.get("receipt", {})

    if not isinstance(manifest_raw, Mapping):
        errors.append("manifest must be a mapping")
        manifest_raw = {}
    if not isinstance(receipt_raw, Mapping):
        errors.append("receipt must be a mapping")
        receipt_raw = {}

    def _manifest_text(field_name: str) -> str:
        try:
            return _normalize_required_text(manifest_raw.get(field_name), field=f"manifest.{field_name}")
        except ZenodoDatasetExportValidationError as exc:
            errors.append(str(exc))
            return ""

    dataset_id = _manifest_text("dataset_id")
    title = _manifest_text("title")
    version_tag = _manifest_text("version_tag")
    author = _manifest_text("author")
    affiliation = _manifest_text("affiliation")
    methodology_notes = _manifest_text("methodology_notes")

    try:
        keywords = _normalize_keywords(manifest_raw.get("keywords"))
    except ZenodoDatasetExportValidationError as exc:
        errors.append(str(exc))
        keywords = ()

    raw_metadata = manifest_raw.get("reproducibility_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        errors.append("manifest.reproducibility_metadata must be a mapping")
        metadata = {}
    else:
        try:
            metadata = _canonicalize_value(dict(raw_metadata), field="manifest.reproducibility_metadata")
        except ZenodoDatasetExportValidationError as exc:
            errors.append(str(exc))
            metadata = {}

    export_hash = manifest_raw.get("export_hash") if isinstance(manifest_raw.get("export_hash"), str) else ""
    if not export_hash.strip():
        errors.append("manifest.export_hash must be non-empty")

    prompt_hash = payload.get("prompt_artifact_hash") if isinstance(payload.get("prompt_artifact_hash"), str) else ""
    matrix_hash = payload.get("invocation_matrix_hash") if isinstance(payload.get("invocation_matrix_hash"), str) else ""
    rigor_hash = payload.get("rigor_metric_pack_hash") if isinstance(payload.get("rigor_metric_pack_hash"), str) else ""
    drift_hash = payload.get("drift_tensor_hash") if isinstance(payload.get("drift_tensor_hash"), str) else ""
    wrapper_hash = payload.get("wrapper_study_hash") if isinstance(payload.get("wrapper_study_hash"), str) else ""
    receipt_pack_hash = payload.get("stable_receipt_pack_hash") if isinstance(payload.get("stable_receipt_pack_hash"), str) else ""

    if prompt_hash != expected_prompt_hash:
        errors.append("bundle.prompt_artifact_hash mismatch")
    if matrix_hash != expected_matrix_hash:
        errors.append("bundle.invocation_matrix_hash mismatch")
    if rigor_hash != expected_rigor_hash:
        errors.append("bundle.rigor_metric_pack_hash mismatch")
    if drift_hash != expected_drift_hash:
        errors.append("bundle.drift_tensor_hash mismatch")
    if wrapper_hash != expected_wrapper_hash:
        errors.append("bundle.wrapper_study_hash mismatch")
    if receipt_pack_hash != expected_receipt_pack_hash:
        errors.append("bundle.stable_receipt_pack_hash mismatch")

    manifest_obj = ZenodoDatasetManifest(
        dataset_id=dataset_id,
        title=title,
        version_tag=version_tag,
        author=author,
        affiliation=affiliation,
        keywords=keywords,
        methodology_notes=methodology_notes,
        reproducibility_metadata=metadata,
        export_hash=export_hash.strip(),
    )

    expected_bundle = ZenodoDatasetExportBundle(
        manifest=manifest_obj,
        prompt_artifact_hash=prompt_hash,
        invocation_matrix_hash=matrix_hash,
        rigor_metric_pack_hash=rigor_hash,
        drift_tensor_hash=drift_hash,
        wrapper_study_hash=wrapper_hash,
        stable_receipt_pack_hash=receipt_pack_hash,
        receipt=ZenodoExportReceipt(prompt_hash="", matrix_hash="", receipt_pack_hash="", bundle_hash="", receipt_hash="", validation_passed=False),
        validation=ZenodoExportValidationReport(valid=True, errors=(), error_count=0),
    )
    recomputed_bundle_hash = _stable_hash(_bundle_hash_payload(expected_bundle))
    if export_hash.strip() != recomputed_bundle_hash:
        errors.append("manifest.export_hash mismatch")

    expected_validation_passed = not errors
    expected_receipt = _build_receipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        receipt_pack_hash=receipt_pack_hash,
        bundle_hash=recomputed_bundle_hash,
        validation_passed=expected_validation_passed,
    )

    receipt = ZenodoExportReceipt(
        prompt_hash=receipt_raw.get("prompt_hash", "") if isinstance(receipt_raw.get("prompt_hash"), str) else "",
        matrix_hash=receipt_raw.get("matrix_hash", "") if isinstance(receipt_raw.get("matrix_hash"), str) else "",
        receipt_pack_hash=receipt_raw.get("receipt_pack_hash", "") if isinstance(receipt_raw.get("receipt_pack_hash"), str) else "",
        bundle_hash=receipt_raw.get("bundle_hash", "") if isinstance(receipt_raw.get("bundle_hash"), str) else "",
        receipt_hash=receipt_raw.get("receipt_hash", "") if isinstance(receipt_raw.get("receipt_hash"), str) else "",
        validation_passed=bool(receipt_raw.get("validation_passed", False)),
    )

    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.receipt_pack_hash != receipt_pack_hash:
        errors.append("receipt.receipt_pack_hash mismatch")
    if receipt.bundle_hash != recomputed_bundle_hash:
        errors.append("receipt.bundle_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")
    if receipt.validation_passed != (not errors):
        errors.append("receipt.validation_passed mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return ZenodoExportValidationReport(valid=not deduped_errors, errors=deduped_errors, error_count=len(deduped_errors))


def build_zenodo_dataset_export_bundle(
    canonical_prompt_artifact: Any,
    invocation_matrix: Any,
    rigor_metric_pack: Any,
    drift_tensor: Any,
    wrapper_divergence_study: Any,
    stable_evaluation_receipt_pack: Any,
    *,
    dataset_id: str,
    title: str,
    version_tag: str,
    author: str,
    affiliation: str,
    keywords: Tuple[str, ...] | list[str] | str | None = None,
    methodology_notes: str = "Deterministic comparative evaluation harness export for archival publication.",
    reproducibility_metadata: Mapping[str, Any] | None = None,
) -> ZenodoDatasetExportBundle:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)
    matrix_hash = _extract_matrix_hash(invocation_matrix)
    rigor_hash = _extract_rigor_pack_hash(rigor_metric_pack)
    drift_hash = _extract_drift_tensor_hash(drift_tensor)
    wrapper_hash = _extract_wrapper_study_hash(wrapper_divergence_study)
    receipt_pack_hash = _extract_receipt_pack_hash(stable_evaluation_receipt_pack)

    dataset_id_text = _normalize_required_text(dataset_id, field="dataset_id")
    title_text = _normalize_required_text(title, field="title")
    version_tag_text = _normalize_required_text(version_tag, field="version_tag")
    author_text = _normalize_required_text(author, field="author")
    affiliation_text = _normalize_required_text(affiliation, field="affiliation")
    methodology_notes_text = _normalize_required_text(methodology_notes, field="methodology_notes")
    normalized_keywords = _normalize_keywords(keywords)

    extra_metadata = {} if reproducibility_metadata is None else _canonicalize_value(
        dict(reproducibility_metadata), field="reproducibility_metadata"
    )
    _validate_reproducibility_metadata_overrides(extra_metadata)

    manifest_payload = _manifest_checksum_payload(
        dataset_id_text,
        title_text,
        version_tag_text,
        author_text,
        affiliation_text,
        normalized_keywords,
        methodology_notes_text,
        extra_metadata,
    )
    generated_metadata = build_reproducibility_metadata(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        rigor_metric_pack_hash=rigor_hash,
        drift_tensor_hash=drift_hash,
        wrapper_study_hash=wrapper_hash,
        stable_receipt_pack_hash=receipt_pack_hash,
        canonical_manifest_payload=manifest_payload,
    )
    combined_metadata = _canonicalize_value(
        {**generated_metadata, **extra_metadata}, field="manifest.reproducibility_metadata"
    )

    manifest = ZenodoDatasetManifest(
        dataset_id=dataset_id_text,
        title=title_text,
        version_tag=version_tag_text,
        author=author_text,
        affiliation=affiliation_text,
        keywords=normalized_keywords,
        methodology_notes=methodology_notes_text,
        reproducibility_metadata=combined_metadata,
        export_hash="",
    )

    provisional = ZenodoDatasetExportBundle(
        manifest=manifest,
        prompt_artifact_hash=prompt_hash,
        invocation_matrix_hash=matrix_hash,
        rigor_metric_pack_hash=rigor_hash,
        drift_tensor_hash=drift_hash,
        wrapper_study_hash=wrapper_hash,
        stable_receipt_pack_hash=receipt_pack_hash,
        receipt=ZenodoExportReceipt(prompt_hash="", matrix_hash="", receipt_pack_hash="", bundle_hash="", receipt_hash="", validation_passed=True),
        validation=ZenodoExportValidationReport(valid=True, errors=(), error_count=0),
    )

    bundle_hash = _stable_hash(_bundle_hash_payload(provisional))
    manifest_with_export_hash = ZenodoDatasetManifest(
        dataset_id=manifest.dataset_id,
        title=manifest.title,
        version_tag=manifest.version_tag,
        author=manifest.author,
        affiliation=manifest.affiliation,
        keywords=manifest.keywords,
        methodology_notes=manifest.methodology_notes,
        reproducibility_metadata=manifest.reproducibility_metadata,
        export_hash=bundle_hash,
    )

    initial_receipt = _build_receipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        receipt_pack_hash=receipt_pack_hash,
        bundle_hash=bundle_hash,
        validation_passed=True,
    )

    initial_bundle = ZenodoDatasetExportBundle(
        manifest=manifest_with_export_hash,
        prompt_artifact_hash=prompt_hash,
        invocation_matrix_hash=matrix_hash,
        rigor_metric_pack_hash=rigor_hash,
        drift_tensor_hash=drift_hash,
        wrapper_study_hash=wrapper_hash,
        stable_receipt_pack_hash=receipt_pack_hash,
        receipt=initial_receipt,
        validation=ZenodoExportValidationReport(valid=True, errors=(), error_count=0),
    )

    final_report = validate_zenodo_dataset_export_bundle(
        initial_bundle,
        canonical_prompt_artifact=canonical_prompt_artifact,
        invocation_matrix=invocation_matrix,
        rigor_metric_pack=rigor_metric_pack,
        drift_tensor=drift_tensor,
        wrapper_divergence_study=wrapper_divergence_study,
        stable_evaluation_receipt_pack=stable_evaluation_receipt_pack,
    )
    final_receipt = _build_receipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        receipt_pack_hash=receipt_pack_hash,
        bundle_hash=bundle_hash,
        validation_passed=final_report.valid,
    )

    return ZenodoDatasetExportBundle(
        manifest=manifest_with_export_hash,
        prompt_artifact_hash=prompt_hash,
        invocation_matrix_hash=matrix_hash,
        rigor_metric_pack_hash=rigor_hash,
        drift_tensor_hash=drift_hash,
        wrapper_study_hash=wrapper_hash,
        stable_receipt_pack_hash=receipt_pack_hash,
        receipt=final_receipt,
        validation=final_report,
    )


def zenodo_export_projection(bundle_or_mapping: ZenodoDatasetExportBundle | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(bundle_or_mapping, ZenodoDatasetExportBundle):
        bundle = bundle_or_mapping
    elif isinstance(bundle_or_mapping, Mapping):
        manifest = bundle_or_mapping.get("manifest", {})
        if not isinstance(manifest, Mapping):
            raise ZenodoDatasetExportValidationError("manifest must be a mapping")
        dataset_id = _normalize_required_text(manifest.get("dataset_id"), field="manifest.dataset_id")
        title = _normalize_required_text(manifest.get("title"), field="manifest.title")
        version_tag = _normalize_required_text(manifest.get("version_tag"), field="manifest.version_tag")
        bundle_hash = _normalize_required_text(manifest.get("export_hash"), field="manifest.export_hash")
        receipt = _build_receipt(
            prompt_hash="projection",
            matrix_hash="projection",
            receipt_pack_hash="projection",
            bundle_hash=bundle_hash,
            validation_passed=True,
        )
        receipt_raw = bundle_or_mapping.get("receipt", {})
        if isinstance(receipt_raw, Mapping) and isinstance(receipt_raw.get("receipt_hash"), str) and receipt_raw.get("receipt_hash").strip():
            receipt_hash = receipt_raw.get("receipt_hash").strip()
        else:
            receipt_hash = receipt.receipt_hash
        return {
            "dataset_id": dataset_id,
            "title": title,
            "version_tag": version_tag,
            "bundle_hash": bundle_hash,
            "receipt_hash": receipt_hash,
        }
    else:
        raise ZenodoDatasetExportValidationError("bundle_or_mapping must be ZenodoDatasetExportBundle or mapping")

    return {
        "dataset_id": bundle.manifest.dataset_id,
        "title": bundle.manifest.title,
        "version_tag": bundle.manifest.version_tag,
        "bundle_hash": bundle.manifest.export_hash,
        "receipt_hash": bundle.receipt.receipt_hash,
    }
