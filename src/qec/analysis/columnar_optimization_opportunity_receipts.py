from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

from qec.analysis.lazy_plan_canonical_receipts import validate_lazy_plan_canonical_receipt
from qec.analysis.schema_equivalence_receipts import validate_schema_equivalence_receipt

_SCHEMA_VERSION = "COLUMNAR_OPTIMIZATION_OPPORTUNITY_RECEIPT_V1"
_MAX_PRECONDITIONS = 4096
_MAX_CONSTRAINTS = 4096
_MAX_RISKS = 4096
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_OPTIMIZATION_CLASSES = {
    "PROJECTION_PUSH_DOWN", "FILTER_PUSH_DOWN", "COLUMN_PRUNING", "PREDICATE_REORDERING", "SORT_ELISION",
    "JOIN_REORDERING", "AGGREGATION_FUSION", "SCHEMA_CANONICALIZATION", "ROW_GROUP_ELISION", "LAZY_STAGE_COLLAPSE", "DECLARED_EXTERNAL_OPTIMIZATION",
}
_ALLOWED_OPTIMIZATION_SCOPES = {"LOCAL_OPERATION", "MULTI_STAGE", "SCHEMA_BOUNDARY", "BACKEND_BOUNDARY", "DECLARED_PIPELINE_SEGMENT"}
_ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "REPLAY_CRITICAL"}
_ALLOWED_OPTIMIZATION_STATUSES = {"ELIGIBLE", "BLOCKED", "REQUIRES_EQUIVALENCE_PROOF", "REQUIRES_SCHEMA_PROOF", "DECLARED_UNSAFE"}
_ALLOWED_CONSTRAINT_TYPES = {
    "ROW_ORDER_MUST_PRESERVE", "NULLABILITY_MUST_PRESERVE", "DTYPE_MUST_PRESERVE", "SCHEMA_HASH_MUST_PRESERVE",
    "OUTPUT_DIGEST_MUST_PRESERVE", "EQUIVALENCE_RECEIPT_REQUIRED", "SCHEMA_EVOLUTION_FORBIDDEN",
}
_FORBIDDEN_RUNTIME_TOKENS = (
    "query executed", "optimization executed", "runtime dataframe", "automatic rewrite", "backend executed", "speedup proven", "optimization authority", "silent rewrite", "backend authority",
)


@dataclass(frozen=True)
class OptimizationPrecondition:
    precondition_index: int
    precondition_name: str
    precondition_satisfied: bool
    optimization_precondition_hash: str


@dataclass(frozen=True)
class OptimizationConstraint:
    constraint_index: int
    constraint_type: str
    constraint_reason: str
    optimization_constraint_hash: str


@dataclass(frozen=True)
class OptimizationRiskDeclaration:
    risk_index: int
    risk_level: str
    risk_reason: str
    optimization_risk_declaration_hash: str


@dataclass(frozen=True)
class ColumnarOptimizationCandidate:
    optimization_class: str
    optimization_status: str
    optimization_candidate_hash: str


@dataclass(frozen=True)
class ColumnarOptimizationScope:
    optimization_scope: str
    referenced_operation_indices: tuple[int, ...]
    optimization_scope_hash: str


@dataclass(frozen=True)
class ColumnarOptimizationOpportunityReceipt:
    schema_version: str
    lazy_plan_canonical_receipt_hash: str
    schema_equivalence_receipt_hash: str
    optimization_candidate: ColumnarOptimizationCandidate
    optimization_scope: ColumnarOptimizationScope
    preconditions: tuple[OptimizationPrecondition, ...] = field(default_factory=tuple)
    constraints: tuple[OptimizationConstraint, ...] = field(default_factory=tuple)
    risks: tuple[OptimizationRiskDeclaration, ...] = field(default_factory=tuple)
    precondition_count: int = 0
    constraint_count: int = 0
    risk_count: int = 0
    optimization_eligible: bool = False
    adapter_only: bool = True
    columnar_optimization_opportunity_receipt_hash: str = ""


def _to_canonical_obj(value: Any) -> Any:
    if isinstance(value, Mapping):
        for k in value.keys():
            if not isinstance(k, str):
                raise TypeError("payload keys must be strings")
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if is_dataclass(value):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, (tuple, list)):
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


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_dense_indices(indices: Sequence[int], label: str) -> None:
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"{label} indices must be dense and zero-indexed")


def _validate_unique_indices(indices: Sequence[int], label: str) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate {label} indices are not allowed")


def _validate_optimization_semantics(receipt: ColumnarOptimizationOpportunityReceipt) -> bool:
    status = receipt.optimization_candidate.optimization_status
    if status in {"REQUIRES_EQUIVALENCE_PROOF", "REQUIRES_SCHEMA_PROOF", "DECLARED_UNSAFE"}:
        return False
    if status != "ELIGIBLE":
        return False
    if not all(p.precondition_satisfied for p in receipt.preconditions):
        return False
    if any(r.risk_level == "REPLAY_CRITICAL" for r in receipt.risks):
        return False
    return True


def _validate_policy_consistency(receipt: ColumnarOptimizationOpportunityReceipt) -> None:
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")


def _check_no_forbidden_runtime_semantics(payload: Mapping[str, Any]) -> None:
    blob = _canonical_json(payload).lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in blob:
            raise ValueError("runtime execution and optimization/backend authority claims are forbidden")


def build_optimization_precondition(precondition_index: int, precondition_name: str, precondition_satisfied: bool) -> OptimizationPrecondition:
    payload = {"precondition_index": precondition_index, "precondition_name": precondition_name, "precondition_satisfied": precondition_satisfied}
    obj = OptimizationPrecondition(**payload, optimization_precondition_hash=_hash_payload(payload))
    validate_optimization_precondition(obj)
    return obj


def validate_optimization_precondition(item: OptimizationPrecondition) -> None:
    if not _is_non_bool_int(item.precondition_index) or item.precondition_index < 0:
        raise ValueError("precondition_index must be non-negative")
    if not isinstance(item.precondition_satisfied, bool):
        raise ValueError("precondition_satisfied must be bool")
    if not item.precondition_name or len(item.precondition_name) > _MAX_NAME_LENGTH:
        raise ValueError("invalid precondition_name")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_hash_format(item.optimization_precondition_hash, "optimization_precondition_hash")
    if _hash_payload(_base_payload(item.__dict__, "optimization_precondition_hash")) != item.optimization_precondition_hash:
        raise ValueError("optimization precondition hash mismatch")

def build_optimization_constraint(constraint_index: int, constraint_type: str, constraint_reason: str) -> OptimizationConstraint:
    payload = {"constraint_index": constraint_index, "constraint_type": constraint_type, "constraint_reason": constraint_reason}
    obj = OptimizationConstraint(**payload, optimization_constraint_hash=_hash_payload(payload))
    validate_optimization_constraint(obj)
    return obj


def validate_optimization_constraint(item: OptimizationConstraint) -> None:
    if not _is_non_bool_int(item.constraint_index) or item.constraint_index < 0:
        raise ValueError("constraint_index must be non-negative")
    if item.constraint_type not in _ALLOWED_CONSTRAINT_TYPES:
        raise ValueError("invalid constraint type")
    if not isinstance(item.constraint_reason, str) or len(item.constraint_reason) > _MAX_REASON_LENGTH:
        raise ValueError("invalid constraint_reason")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_hash_format(item.optimization_constraint_hash, "optimization_constraint_hash")
    if _hash_payload(_base_payload(item.__dict__, "optimization_constraint_hash")) != item.optimization_constraint_hash:
        raise ValueError("optimization constraint hash mismatch")


def build_optimization_risk_declaration(risk_index: int, risk_level: str, risk_reason: str) -> OptimizationRiskDeclaration:
    payload = {"risk_index": risk_index, "risk_level": risk_level, "risk_reason": risk_reason}
    obj = OptimizationRiskDeclaration(**payload, optimization_risk_declaration_hash=_hash_payload(payload))
    validate_optimization_risk_declaration(obj)
    return obj


def validate_optimization_risk_declaration(item: OptimizationRiskDeclaration) -> None:
    if not _is_non_bool_int(item.risk_index) or item.risk_index < 0:
        raise ValueError("risk_index must be non-negative")
    if item.risk_level not in _ALLOWED_RISK_LEVELS:
        raise ValueError("invalid risk level")
    if not isinstance(item.risk_reason, str) or len(item.risk_reason) > _MAX_REASON_LENGTH:
        raise ValueError("invalid risk_reason")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_hash_format(item.optimization_risk_declaration_hash, "optimization_risk_declaration_hash")
    if _hash_payload(_base_payload(item.__dict__, "optimization_risk_declaration_hash")) != item.optimization_risk_declaration_hash:
        raise ValueError("optimization risk declaration hash mismatch")


def build_columnar_optimization_candidate(optimization_class: str, optimization_status: str) -> ColumnarOptimizationCandidate:
    payload = {"optimization_class": optimization_class, "optimization_status": optimization_status}
    obj = ColumnarOptimizationCandidate(**payload, optimization_candidate_hash=_hash_payload(payload))
    validate_columnar_optimization_candidate(obj)
    return obj


def validate_columnar_optimization_candidate(item: ColumnarOptimizationCandidate) -> None:
    if item.optimization_class not in _ALLOWED_OPTIMIZATION_CLASSES:
        raise ValueError("invalid optimization class")
    if item.optimization_status not in _ALLOWED_OPTIMIZATION_STATUSES:
        raise ValueError("invalid optimization status")
    _validate_hash_format(item.optimization_candidate_hash, "optimization_candidate_hash")
    if _hash_payload(_base_payload(item.__dict__, "optimization_candidate_hash")) != item.optimization_candidate_hash:
        raise ValueError("optimization candidate hash mismatch")


def build_columnar_optimization_scope(optimization_scope: str, referenced_operation_indices: Sequence[int]) -> ColumnarOptimizationScope:
    payload = {"optimization_scope": optimization_scope, "referenced_operation_indices": tuple(referenced_operation_indices)}
    obj = ColumnarOptimizationScope(**payload, optimization_scope_hash=_hash_payload(payload))
    validate_columnar_optimization_scope(obj)
    return obj


def validate_columnar_optimization_scope(item: ColumnarOptimizationScope) -> None:
    if item.optimization_scope not in _ALLOWED_OPTIMIZATION_SCOPES:
        raise ValueError("invalid optimization scope")
    if any((not _is_non_bool_int(x) or x < 0) for x in item.referenced_operation_indices):
        raise ValueError("referenced_operation_indices must be non-negative ints")
    _validate_hash_format(item.optimization_scope_hash, "optimization_scope_hash")
    if _hash_payload(_base_payload(item.__dict__, "optimization_scope_hash")) != item.optimization_scope_hash:
        raise ValueError("optimization scope hash mismatch")


def build_columnar_optimization_opportunity_receipt(
    lazy_plan_canonical_receipt_hash: str,
    schema_equivalence_receipt_hash: str,
    optimization_candidate: ColumnarOptimizationCandidate,
    optimization_scope: ColumnarOptimizationScope,
    preconditions: Sequence[OptimizationPrecondition],
    constraints: Sequence[OptimizationConstraint],
    risks: Sequence[OptimizationRiskDeclaration],
    *,
    adapter_only: bool = True,
    lazy_plan_canonical_receipt: Any | None = None,
    schema_equivalence_receipt: Any | None = None,
) -> ColumnarOptimizationOpportunityReceipt:
    sorted_preconditions = tuple(sorted(tuple(preconditions), key=lambda x: x.precondition_index))
    sorted_constraints = tuple(sorted(tuple(constraints), key=lambda x: x.constraint_index))
    sorted_risks = tuple(sorted(tuple(risks), key=lambda x: x.risk_index))
    receipt = ColumnarOptimizationOpportunityReceipt(
        schema_version=_SCHEMA_VERSION,
        lazy_plan_canonical_receipt_hash=lazy_plan_canonical_receipt_hash,
        schema_equivalence_receipt_hash=schema_equivalence_receipt_hash,
        optimization_candidate=optimization_candidate,
        optimization_scope=optimization_scope,
        preconditions=sorted_preconditions,
        constraints=sorted_constraints,
        risks=sorted_risks,
        precondition_count=len(sorted_preconditions),
        constraint_count=len(sorted_constraints),
        risk_count=len(sorted_risks),
        optimization_eligible=False,
        adapter_only=adapter_only,
    )
    recomputed_eligible = _validate_optimization_semantics(receipt)
    payload = _base_payload({**receipt.__dict__, "optimization_eligible": recomputed_eligible}, "columnar_optimization_opportunity_receipt_hash")
    obj = ColumnarOptimizationOpportunityReceipt(**payload, columnar_optimization_opportunity_receipt_hash=_hash_payload(payload))
    validate_columnar_optimization_opportunity_receipt(obj, lazy_plan_canonical_receipt=lazy_plan_canonical_receipt, schema_equivalence_receipt=schema_equivalence_receipt)
    return obj


def validate_columnar_optimization_opportunity_receipt(receipt: ColumnarOptimizationOpportunityReceipt, *, lazy_plan_canonical_receipt: Any | None = None, schema_equivalence_receipt: Any | None = None) -> None:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if len(receipt.preconditions) > _MAX_PRECONDITIONS or len(receipt.constraints) > _MAX_CONSTRAINTS or len(receipt.risks) > _MAX_RISKS:
        raise ValueError("receipt component exceeds max size")
    if lazy_plan_canonical_receipt is not None:
        validate_lazy_plan_canonical_receipt(lazy_plan_canonical_receipt)
        if lazy_plan_canonical_receipt.lazy_plan_canonical_receipt_hash != receipt.lazy_plan_canonical_receipt_hash:
            raise ValueError("lazy plan lineage hash mismatch")
    if schema_equivalence_receipt is not None:
        validate_schema_equivalence_receipt(schema_equivalence_receipt)
        if schema_equivalence_receipt.schema_equivalence_receipt_hash != receipt.schema_equivalence_receipt_hash:
            raise ValueError("schema equivalence lineage hash mismatch")

    validate_columnar_optimization_candidate(receipt.optimization_candidate)
    validate_columnar_optimization_scope(receipt.optimization_scope)
    for x in receipt.preconditions:
        validate_optimization_precondition(x)
    for x in receipt.constraints:
        validate_optimization_constraint(x)
    for x in receipt.risks:
        validate_optimization_risk_declaration(x)

    _validate_dense_indices([x.precondition_index for x in receipt.preconditions], "precondition")
    _validate_unique_indices([x.precondition_index for x in receipt.preconditions], "precondition")
    if tuple(sorted(receipt.preconditions, key=lambda x: x.precondition_index)) != receipt.preconditions:
        raise ValueError("preconditions must be in deterministic index order")
    _validate_dense_indices([x.constraint_index for x in receipt.constraints], "constraint")
    _validate_unique_indices([x.constraint_index for x in receipt.constraints], "constraint")
    if tuple(sorted(receipt.constraints, key=lambda x: x.constraint_index)) != receipt.constraints:
        raise ValueError("constraints must be in deterministic index order")
    _validate_dense_indices([x.risk_index for x in receipt.risks], "risk")
    _validate_unique_indices([x.risk_index for x in receipt.risks], "risk")
    if tuple(sorted(receipt.risks, key=lambda x: x.risk_index)) != receipt.risks:
        raise ValueError("risks must be in deterministic index order")

    _validate_hash_format(receipt.lazy_plan_canonical_receipt_hash, "lazy_plan_canonical_receipt_hash")
    _validate_hash_format(receipt.schema_equivalence_receipt_hash, "schema_equivalence_receipt_hash")
    _validate_hash_format(receipt.columnar_optimization_opportunity_receipt_hash, "columnar_optimization_opportunity_receipt_hash")

    if receipt.precondition_count != len(receipt.preconditions) or receipt.constraint_count != len(receipt.constraints) or receipt.risk_count != len(receipt.risks):
        raise ValueError("count fields must match actual component counts")
    recomputed_eligible = _validate_optimization_semantics(receipt)
    if receipt.optimization_eligible is not recomputed_eligible:
        raise ValueError("optimization_eligible must be derived and deterministic")
    _validate_policy_consistency(receipt)
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    expected = _hash_payload(_base_payload(receipt.__dict__, "columnar_optimization_opportunity_receipt_hash"))
    if expected != receipt.columnar_optimization_opportunity_receipt_hash:
        raise ValueError("columnar optimization opportunity receipt hash mismatch")
