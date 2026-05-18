from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from qec.analysis.dataframe_backend_manifest import validate_dataframe_backend_manifest
from qec.analysis.lazy_plan_canonical_receipts import validate_lazy_plan_canonical_receipt

_SCHEMA_VERSION = "POLARS_PANDAS_EQUIVALENCE_RECEIPT_V1"
_MAX_MISMATCHES = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_EQUIVALENCE_MODES = {"EXACT_CANONICAL_BYTES", "EXACT_CANONICAL_JSON", "EXACT_DIGEST", "SCHEMA_ONLY", "DECLARED_ROUNDING_BOUND"}
_ALLOWED_DTYPE_POLICIES = {"EXACT_DTYPE_MATCH", "DECLARED_CAST_EQUIVALENCE", "NUMERIC_WIDTH_EQUIVALENCE", "STRING_CATEGORY_EQUIVALENCE"}
_ALLOWED_ROW_ORDER_POLICIES = {"PRESERVE_INPUT_ORDER", "DECLARED_SORT_KEYS", "ROW_ORDER_IGNORED_WITH_SORT_KEYS"}
_ALLOWED_NULL_POLICIES = {"NULL_EQUALS_NULL", "NULL_NOT_EQUAL", "NULLS_FORBIDDEN", "DECLARED_NULLABLE"}
_ALLOWED_ROUNDING_POLICIES = {"EXACT", "IEEE754", "FIXED_PRECISION", "ROUND_HALF_EVEN", "ROUND_HALF_UP", "DECLARED_EXTERNAL"}
_ALLOWED_MISMATCH_KINDS = {"SCHEMA_HASH_MISMATCH", "ROW_COUNT_MISMATCH", "COLUMN_COUNT_MISMATCH", "OUTPUT_DIGEST_MISMATCH", "DTYPE_POLICY_MISMATCH", "NULL_POLICY_MISMATCH", "ROUNDING_POLICY_MISMATCH", "ROW_ORDER_POLICY_MISMATCH", "SORT_KEY_MISMATCH"}
_FORBIDDEN_RUNTIME_TOKENS = ("query executed", "backend executed", "runtime dataframe", "automatic optimization", "benchmark executed", "speedup proven", "polars is faster", "pandas authority", "polars authority")


@dataclass(frozen=True)
class DataframeOutputDigest:
    backend_name: str
    backend_manifest_hash: str
    canonical_output_hash: str
    row_count: int
    column_count: int
    schema_manifest_hash: str
    output_digest_hash: str


@dataclass(frozen=True)
class DataframeEquivalencePolicy:
    equivalence_mode: str
    dtype_policy: str
    row_order_policy: str
    declared_sort_keys: tuple[str, ...] = field(default_factory=tuple)
    null_policy: str = "NULL_EQUALS_NULL"
    rounding_policy: str = "EXACT"
    equivalence_policy_hash: str = ""


@dataclass(frozen=True)
class DataframeMismatchRecord:
    mismatch_index: int
    mismatch_kind: str
    left_value_hash: str
    right_value_hash: str
    reason: str
    mismatch_record_hash: str


@dataclass(frozen=True)
class DataframeSchemaComparison:
    left_schema_manifest_hash: str
    right_schema_manifest_hash: str
    schemas_match: bool
    schema_comparison_hash: str


@dataclass(frozen=True)
class PolarsPandasEquivalenceReceipt:
    schema_version: str
    left_backend_name: str
    right_backend_name: str
    left_output_digest: DataframeOutputDigest
    right_output_digest: DataframeOutputDigest
    equivalence_policy: DataframeEquivalencePolicy
    schema_comparison: DataframeSchemaComparison
    mismatches: tuple[DataframeMismatchRecord, ...] = field(default_factory=tuple)
    mismatch_count: int = 0
    lazy_plan_canonical_receipt_hash: str | None = None
    equivalence_passed: bool = False
    adapter_only: bool = True
    polars_pandas_equivalence_receipt_hash: str = ""


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, list):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _validate_dense_mismatch_indices(mismatches: Sequence[DataframeMismatchRecord]) -> None:
    if sorted(m.mismatch_index for m in mismatches) != list(range(len(mismatches))):
        raise ValueError("mismatch indices must be dense and zero-indexed")


def _validate_unique_mismatch_indices(mismatches: Sequence[DataframeMismatchRecord]) -> None:
    idx = [m.mismatch_index for m in mismatches]
    if len(idx) != len(set(idx)):
        raise ValueError("duplicate mismatch indices are not allowed")


def _validate_policy_consistency(policy: DataframeEquivalencePolicy) -> None:
    if policy.row_order_policy == "DECLARED_SORT_KEYS" and len(policy.declared_sort_keys) == 0:
        raise ValueError("declared_sort_keys required for DECLARED_SORT_KEYS")
    if policy.row_order_policy == "PRESERVE_INPUT_ORDER" and len(policy.declared_sort_keys) != 0:
        raise ValueError("declared_sort_keys forbidden for PRESERVE_INPUT_ORDER")
    if policy.row_order_policy == "ROW_ORDER_IGNORED_WITH_SORT_KEYS" and len(policy.declared_sort_keys) == 0:
        raise ValueError("declared_sort_keys required for ROW_ORDER_IGNORED_WITH_SORT_KEYS")
    if policy.equivalence_mode == "DECLARED_ROUNDING_BOUND" and policy.rounding_policy == "EXACT":
        raise ValueError("DECLARED_ROUNDING_BOUND requires non-EXACT rounding policy")


def _validate_equivalence_semantics(receipt: PolarsPandasEquivalenceReceipt) -> tuple[int, bool]:
    mismatch_count = len(receipt.mismatches)
    same_shape = receipt.left_output_digest.row_count == receipt.right_output_digest.row_count and receipt.left_output_digest.column_count == receipt.right_output_digest.column_count
    digest_match = receipt.left_output_digest.canonical_output_hash == receipt.right_output_digest.canonical_output_hash
    if receipt.equivalence_policy.equivalence_mode == "SCHEMA_ONLY":
        digest_condition = True
    else:
        digest_condition = digest_match
    passed = receipt.schema_comparison.schemas_match and same_shape and digest_condition and mismatch_count == 0
    return mismatch_count, passed


def _check_no_forbidden_runtime_semantics(payload: Mapping[str, Any]) -> None:
    blob = _canonical_json(payload).lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in blob:
            raise ValueError("runtime execution semantics and authority/speedup claims are forbidden")


def build_dataframe_output_digest(backend_name: str, backend_manifest_hash: str, canonical_output_hash: str, row_count: int, column_count: int, schema_manifest_hash: str) -> DataframeOutputDigest:
    payload = {"backend_name": backend_name, "backend_manifest_hash": backend_manifest_hash, "canonical_output_hash": canonical_output_hash, "row_count": row_count, "column_count": column_count, "schema_manifest_hash": schema_manifest_hash}
    obj = DataframeOutputDigest(**payload, output_digest_hash=_hash_payload(payload))
    validate_dataframe_output_digest(obj)
    return obj


def validate_dataframe_output_digest(digest: DataframeOutputDigest) -> None:
    if len(digest.backend_name) == 0 or len(digest.backend_name) > _MAX_NAME_LENGTH:
        raise ValueError("backend_name must be non-empty and <= max length")
    if digest.row_count < 0 or digest.column_count < 0:
        raise ValueError("row_count/column_count must be non-negative")
    for field_name in ("backend_manifest_hash", "canonical_output_hash", "schema_manifest_hash", "output_digest_hash"):
        _validate_hash_format(getattr(digest, field_name), field_name)
    expected = _hash_payload(_base_payload(digest.__dict__, "output_digest_hash"))
    if expected != digest.output_digest_hash:
        raise ValueError("output digest hash mismatch")


def build_dataframe_equivalence_policy(equivalence_mode: str, dtype_policy: str, row_order_policy: str, declared_sort_keys: Sequence[str], null_policy: str, rounding_policy: str) -> DataframeEquivalencePolicy:
    payload = {"equivalence_mode": equivalence_mode, "dtype_policy": dtype_policy, "row_order_policy": row_order_policy, "declared_sort_keys": tuple(declared_sort_keys), "null_policy": null_policy, "rounding_policy": rounding_policy}
    obj = DataframeEquivalencePolicy(**payload, equivalence_policy_hash=_hash_payload(payload))
    validate_dataframe_equivalence_policy(obj)
    return obj


def validate_dataframe_equivalence_policy(policy: DataframeEquivalencePolicy) -> None:
    if policy.equivalence_mode not in _ALLOWED_EQUIVALENCE_MODES:
        raise ValueError("invalid equivalence mode")
    if policy.dtype_policy not in _ALLOWED_DTYPE_POLICIES:
        raise ValueError("invalid dtype policy")
    if policy.row_order_policy not in _ALLOWED_ROW_ORDER_POLICIES:
        raise ValueError("invalid row-order policy")
    if policy.null_policy not in _ALLOWED_NULL_POLICIES:
        raise ValueError("invalid null policy")
    if policy.rounding_policy not in _ALLOWED_ROUNDING_POLICIES:
        raise ValueError("invalid rounding policy")
    _validate_policy_consistency(policy)
    _validate_hash_format(policy.equivalence_policy_hash, "equivalence_policy_hash")
    expected = _hash_payload(_base_payload(policy.__dict__, "equivalence_policy_hash"))
    if expected != policy.equivalence_policy_hash:
        raise ValueError("equivalence policy hash mismatch")


def build_dataframe_mismatch_record(mismatch_index: int, mismatch_kind: str, left_value_hash: str, right_value_hash: str, reason: str) -> DataframeMismatchRecord:
    payload = {"mismatch_index": mismatch_index, "mismatch_kind": mismatch_kind, "left_value_hash": left_value_hash, "right_value_hash": right_value_hash, "reason": reason}
    obj = DataframeMismatchRecord(**payload, mismatch_record_hash=_hash_payload(payload))
    validate_dataframe_mismatch_record(obj)
    return obj


def validate_dataframe_mismatch_record(record: DataframeMismatchRecord) -> None:
    if record.mismatch_index < 0:
        raise ValueError("mismatch_index must be non-negative")
    if record.mismatch_kind not in _ALLOWED_MISMATCH_KINDS:
        raise ValueError("invalid mismatch kind")
    _validate_hash_format(record.left_value_hash, "left_value_hash")
    _validate_hash_format(record.right_value_hash, "right_value_hash")
    if len(record.reason) > 512:
        raise ValueError("reason too long")
    _check_no_forbidden_runtime_semantics({"reason": record.reason})
    _validate_hash_format(record.mismatch_record_hash, "mismatch_record_hash")
    expected = _hash_payload(_base_payload(record.__dict__, "mismatch_record_hash"))
    if expected != record.mismatch_record_hash:
        raise ValueError("mismatch record hash mismatch")


def build_dataframe_schema_comparison(left_schema_manifest_hash: str, right_schema_manifest_hash: str, schemas_match: bool) -> DataframeSchemaComparison:
    payload = {"left_schema_manifest_hash": left_schema_manifest_hash, "right_schema_manifest_hash": right_schema_manifest_hash, "schemas_match": schemas_match}
    obj = DataframeSchemaComparison(**payload, schema_comparison_hash=_hash_payload(payload))
    validate_dataframe_schema_comparison(obj)
    return obj


def validate_dataframe_schema_comparison(comp: DataframeSchemaComparison) -> None:
    _validate_hash_format(comp.left_schema_manifest_hash, "left_schema_manifest_hash")
    _validate_hash_format(comp.right_schema_manifest_hash, "right_schema_manifest_hash")
    if type(comp.schemas_match) is not bool:
        raise TypeError("schemas_match must be strict bool")
    if comp.schemas_match != (comp.left_schema_manifest_hash == comp.right_schema_manifest_hash):
        raise ValueError("schema comparison consistency mismatch")
    _validate_hash_format(comp.schema_comparison_hash, "schema_comparison_hash")
    expected = _hash_payload(_base_payload(comp.__dict__, "schema_comparison_hash"))
    if expected != comp.schema_comparison_hash:
        raise ValueError("schema comparison hash mismatch")


def build_polars_pandas_equivalence_receipt(left_backend_name: str, right_backend_name: str, left_output_digest: DataframeOutputDigest, right_output_digest: DataframeOutputDigest, equivalence_policy: DataframeEquivalencePolicy, schema_comparison: DataframeSchemaComparison, mismatches: Sequence[DataframeMismatchRecord] = (), lazy_plan_canonical_receipt_hash: str | None = None, lazy_plan_canonical_receipt: Any | None = None) -> PolarsPandasEquivalenceReceipt:
    mm = tuple(sorted(tuple(mismatches), key=lambda m: m.mismatch_index))
    mismatch_count = len(mm)
    receipt = PolarsPandasEquivalenceReceipt(_SCHEMA_VERSION, left_backend_name, right_backend_name, left_output_digest, right_output_digest, equivalence_policy, schema_comparison, mm, mismatch_count, lazy_plan_canonical_receipt_hash, False, True, "")
    _, passed = _validate_equivalence_semantics(receipt)
    receipt = PolarsPandasEquivalenceReceipt(**{**receipt.__dict__, "equivalence_passed": passed})
    payload = _base_payload(receipt.__dict__, "polars_pandas_equivalence_receipt_hash")
    obj = PolarsPandasEquivalenceReceipt(**payload, polars_pandas_equivalence_receipt_hash=_hash_payload(payload))
    validate_polars_pandas_equivalence_receipt(obj, lazy_plan_canonical_receipt=lazy_plan_canonical_receipt)
    return obj


def validate_polars_pandas_equivalence_receipt(receipt: PolarsPandasEquivalenceReceipt, *, lazy_plan_canonical_receipt: Any | None = None, left_backend_manifest: Any | None = None, right_backend_manifest: Any | None = None) -> None:
    validate_dataframe_output_digest(receipt.left_output_digest)
    validate_dataframe_output_digest(receipt.right_output_digest)
    validate_dataframe_equivalence_policy(receipt.equivalence_policy)
    validate_dataframe_schema_comparison(receipt.schema_comparison)
    if left_backend_manifest is not None:
        validate_dataframe_backend_manifest(left_backend_manifest)
    if right_backend_manifest is not None:
        validate_dataframe_backend_manifest(right_backend_manifest)
    if lazy_plan_canonical_receipt is not None:
        validate_lazy_plan_canonical_receipt(lazy_plan_canonical_receipt)
        if receipt.lazy_plan_canonical_receipt_hash != lazy_plan_canonical_receipt.lazy_plan_canonical_receipt_hash:
            raise ValueError("lazy plan canonical receipt hash mismatch")
    if receipt.lazy_plan_canonical_receipt_hash is not None:
        _validate_hash_format(receipt.lazy_plan_canonical_receipt_hash, "lazy_plan_canonical_receipt_hash")
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if type(receipt.adapter_only) is not bool or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be exactly True")
    if not isinstance(receipt.mismatches, tuple):
        raise TypeError("mismatches must be a tuple")
    if len(receipt.mismatches) > _MAX_MISMATCHES:
        raise ValueError("mismatch count exceeds max")
    for m in receipt.mismatches:
        validate_dataframe_mismatch_record(m)
    _validate_unique_mismatch_indices(receipt.mismatches)
    _validate_dense_mismatch_indices(receipt.mismatches)
    recomputed_count, recomputed_passed = _validate_equivalence_semantics(receipt)
    if receipt.mismatch_count != recomputed_count:
        raise ValueError("mismatch_count mismatch")
    if receipt.equivalence_passed != recomputed_passed:
        raise ValueError("equivalence_passed mismatch")
    if receipt.equivalence_passed and len(receipt.mismatches) != 0:
        raise ValueError("mismatch records forbidden when equivalence_passed is True")
    if (not receipt.equivalence_passed) and len(receipt.mismatches) == 0:
        raise ValueError("mismatch records required when equivalence_passed is False")
    _check_no_forbidden_runtime_semantics(_base_payload(receipt.__dict__, "polars_pandas_equivalence_receipt_hash"))
    _validate_hash_format(receipt.polars_pandas_equivalence_receipt_hash, "polars_pandas_equivalence_receipt_hash")
    expected = _hash_payload(_base_payload(receipt.__dict__, "polars_pandas_equivalence_receipt_hash"))
    if expected != receipt.polars_pandas_equivalence_receipt_hash:
        raise ValueError("receipt hash mismatch")
