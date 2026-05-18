from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from qec.analysis.canonical_hashing import canonicalize_json as _canonicalize_json
from qec.analysis.canonical_hashing import CanonicalHashingError

_SCHEMA_VERSION = "LAZY_PLAN_CANONICAL_RECEIPT_V1"
_MAX_OPERATIONS = 4096
_MAX_COLUMNS = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_OPERATION_TYPES = {
    "SELECT", "FILTER", "GROUP_BY", "AGGREGATE", "SORT", "JOIN", "LIMIT", "DISTINCT", "RENAME", "CAST", "UNION", "CONCAT", "SCHEMA_TRANSITION",
}
_ALLOWED_JOIN_TYPES = {"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "SEMI", "ANTI"}
_ALLOWED_SORT_DIRECTIONS = {"ASCENDING", "DESCENDING"}
_ALLOWED_EXECUTION_BOUNDARIES = {"FULLY_LAZY", "PARTIALLY_LAZY", "MATERIALIZATION_DECLARED", "EAGER_TRANSITION_DECLARED"}
_ALLOWED_SCHEMA_TRANSITIONS = {"NO_SCHEMA_CHANGE", "COLUMN_ADDED", "COLUMN_REMOVED", "COLUMN_RENAMED", "DTYPE_CHANGED", "NULLABILITY_CHANGED"}
_FORBIDDEN_RUNTIME_TOKENS = (
    "query executed", "backend executed", "runtime dataframe", "automatic optimization", "materialized dataframe result", "live execution",
)


@dataclass(frozen=True)
class LazyPlanOperation:
    operation_index: int
    operation_type: str
    source_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    operation_payload: Mapping[str, Any]
    operation_hash: str


@dataclass(frozen=True)
class LazyPlanProjection:
    projected_columns: tuple[str, ...]
    projection_hash: str


@dataclass(frozen=True)
class LazyPlanFilter:
    filter_expression: str
    referenced_columns: tuple[str, ...]
    filter_hash: str


@dataclass(frozen=True)
class LazyPlanAggregation:
    grouping_columns: tuple[str, ...]
    aggregation_functions: tuple[str, ...]
    aggregation_hash: str


@dataclass(frozen=True)
class LazyPlanJoin:
    join_type: str
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    join_hash: str


@dataclass(frozen=True)
class LazyPlanSort:
    sort_columns: tuple[str, ...]
    sort_directions: tuple[str, ...]
    sort_hash: str


@dataclass(frozen=True)
class LazyPlanSchemaTransition:
    transition_type: str
    input_schema_hash: str
    output_schema_hash: str
    schema_transition_hash: str


@dataclass(frozen=True)
class LazyExecutionBoundary:
    execution_boundary: str
    eager_materialization_allowed: bool
    lazy_reordering_allowed: bool
    execution_boundary_hash: str


@dataclass(frozen=True)
class LazyPlanCanonicalReceipt:
    schema_version: str
    backend_name: str
    backend_version: str
    adapter_only: bool
    operations: tuple[LazyPlanOperation, ...]
    projection: LazyPlanProjection
    filters: tuple[LazyPlanFilter, ...] = field(default_factory=tuple)
    aggregations: tuple[LazyPlanAggregation, ...] = field(default_factory=tuple)
    joins: tuple[LazyPlanJoin, ...] = field(default_factory=tuple)
    sorts: tuple[LazyPlanSort, ...] = field(default_factory=tuple)
    schema_transitions: tuple[LazyPlanSchemaTransition, ...] = field(default_factory=tuple)
    execution_boundary: LazyExecutionBoundary | None = None
    operation_count: int = 0
    lazy_plan_canonical_receipt_hash: str = ""


def _to_canonical_obj(value: Any) -> Any:
    """Convert value to canonical form, rejecting non-string mapping keys."""
    if isinstance(value, Mapping):
        for k in value.keys():
            if not isinstance(k, str):
                raise CanonicalHashingError("payload keys must be strings")
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, list):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize payload to canonical JSON, rejecting non-string keys."""
    _canonicalize_json(payload)  # Validate keys are strings (result not used, validation only)
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


def _validate_dense_operation_indices(operations: Sequence[LazyPlanOperation]) -> None:
    if sorted(op.operation_index for op in operations) != list(range(len(operations))):
        raise ValueError("operation indices must be dense and zero-indexed")


def _validate_unique_operation_indices(operations: Sequence[LazyPlanOperation]) -> None:
    idx = [op.operation_index for op in operations]
    if len(idx) != len(set(idx)):
        raise ValueError("duplicate operation indices are not allowed")


def _validate_sort_consistency(sort_obj: LazyPlanSort) -> None:
    if len(sort_obj.sort_columns) != len(sort_obj.sort_directions):
        raise ValueError("sort_columns and sort_directions must have the same length")
    if len(sort_obj.sort_columns) != len(set(sort_obj.sort_columns)):
        raise ValueError("duplicate sort keys are not allowed")


def _freeze_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def _check_no_forbidden_runtime_semantics(payload: Mapping[str, Any]) -> None:
    blob = _canonical_json(payload).lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in blob:
            raise ValueError("runtime execution semantics are forbidden in canonical receipts")


def build_lazy_plan_operation(operation_index: int, operation_type: str, source_columns: Sequence[str], target_columns: Sequence[str], operation_payload: Mapping[str, Any]) -> LazyPlanOperation:
    op = LazyPlanOperation(operation_index, operation_type, tuple(source_columns), tuple(target_columns), _freeze_payload(operation_payload), "")
    payload = _base_payload(op.__dict__, "operation_hash")
    obj = LazyPlanOperation(**payload, operation_hash=_hash_payload(payload))
    validate_lazy_plan_operation(obj)
    return obj


def validate_lazy_plan_operation(operation: LazyPlanOperation) -> None:
    if operation.operation_type not in _ALLOWED_OPERATION_TYPES:
        raise ValueError("invalid operation type")
    if operation.operation_index < 0:
        raise ValueError("operation_index must be non-negative")
    if not isinstance(operation.operation_payload, MappingProxyType):
        raise TypeError("operation_payload must be immutable")
    _check_no_forbidden_runtime_semantics(operation.operation_payload)
    _validate_hash_format(operation.operation_hash, "operation_hash")
    expected = _hash_payload(_base_payload(operation.__dict__, "operation_hash"))
    if expected != operation.operation_hash:
        raise ValueError("operation hash mismatch")


def build_lazy_plan_projection(projected_columns: Sequence[str]) -> LazyPlanProjection:
    payload = {"projected_columns": tuple(projected_columns)}
    obj = LazyPlanProjection(**payload, projection_hash=_hash_payload(payload))
    validate_lazy_plan_projection(obj)
    return obj


def validate_lazy_plan_projection(projection: LazyPlanProjection) -> None:
    if len(projection.projected_columns) > _MAX_COLUMNS:
        raise ValueError("projection exceeds max columns")
    _validate_hash_format(projection.projection_hash, "projection_hash")
    expected_payload = {"projected_columns": projection.projected_columns}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != projection.projection_hash:
        raise ValueError("projection hash mismatch")


def build_lazy_plan_filter(filter_expression: str, referenced_columns: Sequence[str]) -> LazyPlanFilter:
    payload = {"filter_expression": filter_expression, "referenced_columns": tuple(referenced_columns)}
    obj = LazyPlanFilter(**payload, filter_hash=_hash_payload(payload))
    validate_lazy_plan_filter(obj)
    return obj


def validate_lazy_plan_filter(filter_obj: LazyPlanFilter) -> None:
    _check_no_forbidden_runtime_semantics({"filter_expression": filter_obj.filter_expression})
    _validate_hash_format(filter_obj.filter_hash, "filter_hash")
    expected_payload = {"filter_expression": filter_obj.filter_expression, "referenced_columns": filter_obj.referenced_columns}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != filter_obj.filter_hash:
        raise ValueError("filter hash mismatch")


def build_lazy_plan_aggregation(grouping_columns: Sequence[str], aggregation_functions: Sequence[str]) -> LazyPlanAggregation:
    payload = {"grouping_columns": tuple(grouping_columns), "aggregation_functions": tuple(aggregation_functions)}
    obj = LazyPlanAggregation(**payload, aggregation_hash=_hash_payload(payload))
    validate_lazy_plan_aggregation(obj)
    return obj


def validate_lazy_plan_aggregation(aggregation: LazyPlanAggregation) -> None:
    _validate_hash_format(aggregation.aggregation_hash, "aggregation_hash")
    expected_payload = {"grouping_columns": aggregation.grouping_columns, "aggregation_functions": aggregation.aggregation_functions}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != aggregation.aggregation_hash:
        raise ValueError("aggregation hash mismatch")


def build_lazy_plan_join(join_type: str, left_keys: Sequence[str], right_keys: Sequence[str]) -> LazyPlanJoin:
    payload = {"join_type": join_type, "left_keys": tuple(left_keys), "right_keys": tuple(right_keys)}
    obj = LazyPlanJoin(**payload, join_hash=_hash_payload(payload))
    validate_lazy_plan_join(obj)
    return obj


def validate_lazy_plan_join(join_obj: LazyPlanJoin) -> None:
    if join_obj.join_type not in _ALLOWED_JOIN_TYPES:
        raise ValueError("invalid join type")
    _validate_hash_format(join_obj.join_hash, "join_hash")
    expected_payload = {"join_type": join_obj.join_type, "left_keys": join_obj.left_keys, "right_keys": join_obj.right_keys}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != join_obj.join_hash:
        raise ValueError("join hash mismatch")


def build_lazy_plan_sort(sort_columns: Sequence[str], sort_directions: Sequence[str]) -> LazyPlanSort:
    payload = {"sort_columns": tuple(sort_columns), "sort_directions": tuple(sort_directions)}
    obj = LazyPlanSort(**payload, sort_hash=_hash_payload(payload))
    validate_lazy_plan_sort(obj)
    return obj


def validate_lazy_plan_sort(sort_obj: LazyPlanSort) -> None:
    _validate_sort_consistency(sort_obj)
    for direction in sort_obj.sort_directions:
        if direction not in _ALLOWED_SORT_DIRECTIONS:
            raise ValueError("invalid sort direction")
    _validate_hash_format(sort_obj.sort_hash, "sort_hash")
    expected_payload = {"sort_columns": sort_obj.sort_columns, "sort_directions": sort_obj.sort_directions}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != sort_obj.sort_hash:
        raise ValueError("sort hash mismatch")


def build_lazy_plan_schema_transition(transition_type: str, input_schema_hash: str, output_schema_hash: str) -> LazyPlanSchemaTransition:
    payload = {"transition_type": transition_type, "input_schema_hash": input_schema_hash, "output_schema_hash": output_schema_hash}
    obj = LazyPlanSchemaTransition(**payload, schema_transition_hash=_hash_payload(payload))
    validate_lazy_plan_schema_transition(obj)
    return obj


def validate_lazy_plan_schema_transition(transition: LazyPlanSchemaTransition) -> None:
    if transition.transition_type not in _ALLOWED_SCHEMA_TRANSITIONS:
        raise ValueError("invalid schema transition")
    _validate_hash_format(transition.input_schema_hash, "input_schema_hash")
    _validate_hash_format(transition.output_schema_hash, "output_schema_hash")
    _validate_hash_format(transition.schema_transition_hash, "schema_transition_hash")
    expected_payload = {"transition_type": transition.transition_type, "input_schema_hash": transition.input_schema_hash, "output_schema_hash": transition.output_schema_hash}
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != transition.schema_transition_hash:
        raise ValueError("schema transition hash mismatch")


def build_lazy_execution_boundary(execution_boundary: str, eager_materialization_allowed: bool, lazy_reordering_allowed: bool) -> LazyExecutionBoundary:
    payload = {
        "execution_boundary": execution_boundary,
        "eager_materialization_allowed": eager_materialization_allowed,
        "lazy_reordering_allowed": lazy_reordering_allowed,
    }
    obj = LazyExecutionBoundary(**payload, execution_boundary_hash=_hash_payload(payload))
    validate_lazy_execution_boundary(obj)
    return obj


def validate_lazy_execution_boundary(boundary: LazyExecutionBoundary) -> None:
    if boundary.execution_boundary not in _ALLOWED_EXECUTION_BOUNDARIES:
        raise ValueError("invalid execution boundary")
    if type(boundary.eager_materialization_allowed) is not bool:
        raise TypeError("eager_materialization_allowed must be strict bool")
    if type(boundary.lazy_reordering_allowed) is not bool:
        raise TypeError("lazy_reordering_allowed must be strict bool")
    _validate_hash_format(boundary.execution_boundary_hash, "execution_boundary_hash")
    expected_payload = {
        "execution_boundary": boundary.execution_boundary,
        "eager_materialization_allowed": boundary.eager_materialization_allowed,
        "lazy_reordering_allowed": boundary.lazy_reordering_allowed,
    }
    expected_hash = _hash_payload(expected_payload)
    if expected_hash != boundary.execution_boundary_hash:
        raise ValueError("execution boundary hash mismatch")


def build_lazy_plan_canonical_receipt(backend_name: str, backend_version: str, adapter_only: bool, operations: Sequence[LazyPlanOperation], projection: LazyPlanProjection, filters: Sequence[LazyPlanFilter] = (), aggregations: Sequence[LazyPlanAggregation] = (), joins: Sequence[LazyPlanJoin] = (), sorts: Sequence[LazyPlanSort] = (), schema_transitions: Sequence[LazyPlanSchemaTransition] = (), execution_boundary: LazyExecutionBoundary | None = None) -> LazyPlanCanonicalReceipt:
    ops = tuple(sorted(tuple(operations), key=lambda op: op.operation_index))
    for op in ops:
        validate_lazy_plan_operation(op)
    _validate_unique_operation_indices(ops)
    _validate_dense_operation_indices(ops)
    for f in filters:
        validate_lazy_plan_filter(f)
    for a in aggregations:
        validate_lazy_plan_aggregation(a)
    for j in joins:
        validate_lazy_plan_join(j)
    for s in sorts:
        validate_lazy_plan_sort(s)
    for st in schema_transitions:
        validate_lazy_plan_schema_transition(st)
    if execution_boundary is not None:
        validate_lazy_execution_boundary(execution_boundary)
    validate_lazy_plan_projection(projection)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "adapter_only": adapter_only,
        "operations": [o.__dict__ for o in ops],
        "projection": projection.__dict__,
        "filters": [x.__dict__ for x in filters],
        "aggregations": [x.__dict__ for x in aggregations],
        "joins": [x.__dict__ for x in joins],
        "sorts": [x.__dict__ for x in sorts],
        "schema_transitions": [x.__dict__ for x in schema_transitions],
        "execution_boundary": execution_boundary.__dict__ if execution_boundary else None,
        "operation_count": len(ops),
    }
    obj = LazyPlanCanonicalReceipt(
        schema_version=_SCHEMA_VERSION,
        backend_name=backend_name,
        backend_version=backend_version,
        adapter_only=adapter_only,
        operations=ops,
        projection=projection,
        filters=tuple(filters),
        aggregations=tuple(aggregations),
        joins=tuple(joins),
        sorts=tuple(sorts),
        schema_transitions=tuple(schema_transitions),
        execution_boundary=execution_boundary,
        operation_count=len(ops),
        lazy_plan_canonical_receipt_hash=_hash_payload(payload),
    )
    validate_lazy_plan_canonical_receipt(obj)
    return obj


def validate_lazy_plan_canonical_receipt(receipt: LazyPlanCanonicalReceipt) -> None:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema version")
    if type(receipt.adapter_only) is not bool:
        raise TypeError("adapter_only must be strict bool")
    if len(receipt.operations) > _MAX_OPERATIONS:
        raise ValueError("too many operations")
    _validate_unique_operation_indices(receipt.operations)
    _validate_dense_operation_indices(receipt.operations)
    if tuple(sorted(receipt.operations, key=lambda op: op.operation_index)) != receipt.operations:
        raise ValueError("operations must be in deterministic index order")
    if receipt.operation_count != len(receipt.operations):
        raise ValueError("operation count mismatch")
    # Validate all nested components
    for op in receipt.operations:
        validate_lazy_plan_operation(op)
    validate_lazy_plan_projection(receipt.projection)
    for f in receipt.filters:
        validate_lazy_plan_filter(f)
    for a in receipt.aggregations:
        validate_lazy_plan_aggregation(a)
    for j in receipt.joins:
        validate_lazy_plan_join(j)
    for s in receipt.sorts:
        validate_lazy_plan_sort(s)
    for st in receipt.schema_transitions:
        validate_lazy_plan_schema_transition(st)
    if receipt.execution_boundary is not None:
        validate_lazy_execution_boundary(receipt.execution_boundary)
    _validate_hash_format(receipt.lazy_plan_canonical_receipt_hash, "lazy_plan_canonical_receipt_hash")
    payload = {
        "schema_version": receipt.schema_version,
        "backend_name": receipt.backend_name,
        "backend_version": receipt.backend_version,
        "adapter_only": receipt.adapter_only,
        "operations": [o.__dict__ for o in receipt.operations],
        "projection": receipt.projection.__dict__,
        "filters": [x.__dict__ for x in receipt.filters],
        "aggregations": [x.__dict__ for x in receipt.aggregations],
        "joins": [x.__dict__ for x in receipt.joins],
        "sorts": [x.__dict__ for x in receipt.sorts],
        "schema_transitions": [x.__dict__ for x in receipt.schema_transitions],
        "execution_boundary": receipt.execution_boundary.__dict__ if receipt.execution_boundary else None,
        "operation_count": len(receipt.operations),
    }
    expected = _hash_payload(payload)
    if expected != receipt.lazy_plan_canonical_receipt_hash:
        raise ValueError("lazy plan canonical receipt hash mismatch")
