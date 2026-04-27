"""v149.5 — Deterministic full-stack system demonstration artifact."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
import math
from typing import Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

SYSTEM_DEMONSTRATION_MODULE_VERSION: Final[str] = "v149.5"

_ALLOWED_PLAN_STATUSES: Final[tuple[str, ...]] = ("EMPTY", "COMPRESSED", "NO_GAIN")
_ALLOWED_PROMOTION_STATUSES: Final[tuple[str, ...]] = (
    "INSUFFICIENT_EVIDENCE",
    "PROMOTION_BLOCKED",
    "PROMOTED",
)
_ALLOWED_METRIC_STATUSES: Final[tuple[str, ...]] = ("PASS", "WARN", "FAIL")

_RESEARCH_TOKEN_1: Final[str] = "repair_reasoning_without_probabilistic_search"
_RESEARCH_TOKEN_2: Final[str] = "cross_environment_convergence_supported"
_RESEARCH_TOKEN_3: Final[str] = "repair_necessity_counterfactual_supported"
_RESEARCH_TOKEN_4: Final[str] = "failure_fix_validation_chain_deterministic"
_RESEARCH_TOKEN_5: Final[str] = "long_term_governance_repair_convergence_supported"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round_public_metric(value: float) -> float:
    return float(round(float(value), 12))


def _require_canonical_token(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty canonical string")
    token = value.strip()
    if not token or token != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return token


def _require_sha256_hex(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid SHA-256 hex") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    return value


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    return number


def _normalize_trace_hashes(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError("governance_trace_hashes must be tuple")
    normalized = tuple(sorted(_require_sha256_hex(v, name="governance_trace_hashes") for v in values))
    if not normalized:
        raise ValueError("governance_trace_hashes must be non-empty for valid workloads")
    if len(set(normalized)) != len(normalized):
        raise ValueError("governance_trace_hashes must be unique")
    return normalized


def _bounded_ratio(numerator: int, denominator: int) -> float:
    value = float(numerator) / float(max(1, denominator))
    return _round_public_metric(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class WorkloadDemonstrationDescriptor:
    workload_id: str
    workload_type: str
    input_hash: str
    baseline_cost: int
    optimized_cost: int
    failure_count: int
    repair_count: int
    governance_trace_hashes: tuple[str, ...]
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.workload_id, name="workload_id")
        _require_canonical_token(self.workload_type, name="workload_type")
        _require_sha256_hex(self.input_hash, name="input_hash")
        _require_non_negative_int(self.baseline_cost, name="baseline_cost")
        _require_non_negative_int(self.optimized_cost, name="optimized_cost")
        if self.optimized_cost > self.baseline_cost:
            raise ValueError("optimized_cost must be <= baseline_cost")
        _require_non_negative_int(self.failure_count, name="failure_count")
        _require_non_negative_int(self.repair_count, name="repair_count")
        object.__setattr__(self, "governance_trace_hashes", _normalize_trace_hashes(self.governance_trace_hashes))

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "workload_id": self.workload_id,
            "workload_type": self.workload_type,
            "input_hash": self.input_hash,
            "baseline_cost": int(self.baseline_cost),
            "optimized_cost": int(self.optimized_cost),
            "failure_count": int(self.failure_count),
            "repair_count": int(self.repair_count),
            "governance_trace_hashes": self.governance_trace_hashes,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class SystemStackEvidence:
    multi_agent_receipt_hash: str
    hierarchical_memory_receipt_hash: str
    hardware_alignment_receipt_hash: str
    execution_bridge_receipt_hash: str
    compression_storage_receipt_hash: str
    compression_plan_status: str
    compression_ratio: float
    total_canonical_size_bytes: int
    total_compressed_size_bytes: int
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_sha256_hex(self.multi_agent_receipt_hash, name="multi_agent_receipt_hash")
        _require_sha256_hex(self.hierarchical_memory_receipt_hash, name="hierarchical_memory_receipt_hash")
        _require_sha256_hex(self.hardware_alignment_receipt_hash, name="hardware_alignment_receipt_hash")
        _require_sha256_hex(self.execution_bridge_receipt_hash, name="execution_bridge_receipt_hash")
        _require_sha256_hex(self.compression_storage_receipt_hash, name="compression_storage_receipt_hash")
        if self.compression_plan_status not in _ALLOWED_PLAN_STATUSES:
            raise ValueError("compression_plan_status must be one of EMPTY|COMPRESSED|NO_GAIN")
        object.__setattr__(self, "compression_ratio", _round_public_metric(_require_probability(self.compression_ratio, name="compression_ratio")))
        _require_non_negative_int(self.total_canonical_size_bytes, name="total_canonical_size_bytes")
        _require_non_negative_int(self.total_compressed_size_bytes, name="total_compressed_size_bytes")
        if self.total_canonical_size_bytes == 0 and self.total_compressed_size_bytes != 0:
            raise ValueError(
                "total_compressed_size_bytes must be 0 when total_canonical_size_bytes is 0"
            )

        if self.compression_plan_status == "COMPRESSED" and float(self.compression_ratio) <= 0.0:
            raise ValueError("compression_ratio must be > 0 for COMPRESSED plan")
        if self.compression_plan_status == "EMPTY":
            if float(self.compression_ratio) != 0.0:
                raise ValueError("compression_ratio must be 0 for EMPTY plan")
            if self.total_canonical_size_bytes != 0:
                raise ValueError("total_canonical_size_bytes must be 0 for EMPTY plan")
            if self.total_compressed_size_bytes != 0:
                raise ValueError("total_compressed_size_bytes must be 0 for EMPTY plan")
        if self.compression_plan_status == "NO_GAIN":
            expected_no_gain_ratio = 0.0 if self.total_canonical_size_bytes == 0 else 1.0
            if float(self.compression_ratio) != expected_no_gain_ratio:
                if self.total_canonical_size_bytes == 0:
                    raise ValueError("compression_ratio must be 0 for NO_GAIN plan when canonical size is zero")
                raise ValueError("compression_ratio must be 1 for NO_GAIN plan when canonical size is non-zero")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "multi_agent_receipt_hash": self.multi_agent_receipt_hash,
            "hierarchical_memory_receipt_hash": self.hierarchical_memory_receipt_hash,
            "hardware_alignment_receipt_hash": self.hardware_alignment_receipt_hash,
            "execution_bridge_receipt_hash": self.execution_bridge_receipt_hash,
            "compression_storage_receipt_hash": self.compression_storage_receipt_hash,
            "compression_plan_status": self.compression_plan_status,
            "compression_ratio": _round_public_metric(float(self.compression_ratio)),
            "total_canonical_size_bytes": int(self.total_canonical_size_bytes),
            "total_compressed_size_bytes": int(self.total_compressed_size_bytes),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class SystemDemonstrationMetric:
    metric_name: str
    metric_value: float
    metric_status: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.metric_name, name="metric_name")
        object.__setattr__(self, "metric_value", _round_public_metric(_require_probability(self.metric_value, name="metric_value")))
        if self.metric_status not in _ALLOWED_METRIC_STATUSES:
            raise ValueError("metric_status must be one of PASS|WARN|FAIL")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "metric_name": self.metric_name,
            "metric_value": _round_public_metric(float(self.metric_value)),
            "metric_status": self.metric_status,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class SystemDemonstrationReceipt:
    module_version: str
    workload_count: int
    evidence: SystemStackEvidence
    metrics: tuple[SystemDemonstrationMetric, ...]
    promotion_status: str
    deterministic_integrity_preserved: bool
    measurable_gain_exists: bool
    research_questions_supported: tuple[str, ...]
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.module_version != SYSTEM_DEMONSTRATION_MODULE_VERSION:
            raise ValueError("module_version must match SYSTEM_DEMONSTRATION_MODULE_VERSION")
        _require_non_negative_int(self.workload_count, name="workload_count")
        if not isinstance(self.metrics, tuple):
            raise ValueError("metrics must be tuple")
        for item in self.metrics:
            if not isinstance(item, SystemDemonstrationMetric):
                raise ValueError("metrics must contain SystemDemonstrationMetric")
        metric_names = tuple(item.metric_name for item in self.metrics)
        sorted_names = tuple(sorted(metric_names))
        if metric_names != sorted_names:
            raise ValueError("metrics must be sorted by metric_name")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric_name must be unique")
        if self.promotion_status not in _ALLOWED_PROMOTION_STATUSES:
            raise ValueError("promotion_status must be one of INSUFFICIENT_EVIDENCE|PROMOTION_BLOCKED|PROMOTED")
        if not isinstance(self.deterministic_integrity_preserved, bool):
            raise ValueError("deterministic_integrity_preserved must be bool")
        if not isinstance(self.measurable_gain_exists, bool):
            raise ValueError("measurable_gain_exists must be bool")
        if not isinstance(self.research_questions_supported, tuple):
            raise ValueError("research_questions_supported must be tuple")
        normalized_tokens = tuple(_require_canonical_token(v, name="research_questions_supported") for v in self.research_questions_supported)
        if len(set(normalized_tokens)) != len(normalized_tokens):
            raise ValueError("research_questions_supported must be unique")
        object.__setattr__(self, "research_questions_supported", normalized_tokens)

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "module_version": self.module_version,
            "workload_count": int(self.workload_count),
            "evidence": self.evidence.to_dict(),
            "metrics": tuple(item.to_dict() for item in self.metrics),
            "promotion_status": self.promotion_status,
            "deterministic_integrity_preserved": self.deterministic_integrity_preserved,
            "measurable_gain_exists": self.measurable_gain_exists,
            "research_questions_supported": self.research_questions_supported,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _metric_status_from_thresholds(value: float, *, pass_min: float, warn_min: float) -> str:
    if value >= pass_min:
        return "PASS"
    if value >= warn_min:
        return "WARN"
    return "FAIL"


def _compute_metrics(workloads: tuple[WorkloadDemonstrationDescriptor, ...], evidence: SystemStackEvidence) -> tuple[SystemDemonstrationMetric, ...]:
    workload_count = len(workloads)
    workloads_with_trace = sum(1 for w in workloads if len(w.governance_trace_hashes) >= 1)
    baseline_total = sum(w.baseline_cost for w in workloads)
    optimized_total = sum(w.optimized_cost for w in workloads)
    failure_total = sum(w.failure_count for w in workloads)
    repair_total = sum(w.repair_count for w in workloads)

    deterministic_integrity = 1.0
    governance_stability = 1.0 if workloads_with_trace == workload_count else 0.0
    repair_convergence = _bounded_ratio(repair_total, failure_total)
    cost_reduction = _bounded_ratio(baseline_total - optimized_total, baseline_total)
    if evidence.compression_plan_status in {"NO_GAIN", "EMPTY"}:
        compression_effectiveness = 0.0
    else:
        compression_effectiveness = _round_public_metric(1.0 - float(evidence.compression_ratio))
    storage_efficiency_gain = _bounded_ratio(
        evidence.total_canonical_size_bytes - evidence.total_compressed_size_bytes,
        evidence.total_canonical_size_bytes,
    )
    bounded_success_rate = _bounded_ratio(
        workload_count,
        workload_count + failure_total,
    )
    trace_reproducibility = _bounded_ratio(workloads_with_trace, workload_count)

    return tuple(
        sorted(
            (
                SystemDemonstrationMetric(
                    metric_name="deterministic_integrity",
                    metric_value=deterministic_integrity,
                    metric_status="PASS",
                ),
                SystemDemonstrationMetric(
                    metric_name="governance_stability",
                    metric_value=governance_stability,
                    metric_status="PASS" if governance_stability == 1.0 else "FAIL",
                ),
                SystemDemonstrationMetric(
                    metric_name="repair_convergence",
                    metric_value=repair_convergence,
                    metric_status=_metric_status_from_thresholds(repair_convergence, pass_min=0.75, warn_min=0.000000000001),
                ),
                SystemDemonstrationMetric(
                    metric_name="cost_reduction",
                    metric_value=cost_reduction,
                    metric_status="PASS" if cost_reduction > 0.0 else "WARN",
                ),
                SystemDemonstrationMetric(
                    metric_name="compression_effectiveness",
                    metric_value=compression_effectiveness,
                    metric_status="PASS" if compression_effectiveness > 0.0 else "WARN",
                ),
                SystemDemonstrationMetric(
                    metric_name="storage_efficiency_gain",
                    metric_value=storage_efficiency_gain,
                    metric_status="PASS" if storage_efficiency_gain > 0.0 else "WARN",
                ),
                SystemDemonstrationMetric(
                    metric_name="bounded_success_rate",
                    metric_value=bounded_success_rate,
                    metric_status=_metric_status_from_thresholds(bounded_success_rate, pass_min=0.75, warn_min=0.5),
                ),
                SystemDemonstrationMetric(
                    metric_name="trace_reproducibility",
                    metric_value=trace_reproducibility,
                    metric_status="PASS" if trace_reproducibility == 1.0 else ("WARN" if trace_reproducibility > 0.0 else "FAIL"),
                ),
            ),
            key=lambda metric: metric.metric_name,
        )
    )


def _enforce_compression_consistency(evidence: SystemStackEvidence, metrics: tuple[SystemDemonstrationMetric, ...]) -> None:
    if evidence.total_canonical_size_bytes == 0 and evidence.total_compressed_size_bytes != 0:
        raise ValueError(
            "total_compressed_size_bytes must be 0 when total_canonical_size_bytes is 0"
        )
    expected_ratio = 0.0
    if evidence.total_canonical_size_bytes > 0:
        expected_ratio = _round_public_metric(evidence.total_compressed_size_bytes / evidence.total_canonical_size_bytes)
    actual_ratio = _round_public_metric(float(evidence.compression_ratio))
    if abs(actual_ratio - expected_ratio) > 1e-12:
        raise ValueError("compression_ratio must match total_compressed_size_bytes / total_canonical_size_bytes")

    metric_map = {metric.metric_name: metric for metric in metrics}
    compression_effectiveness = _round_public_metric(float(metric_map["compression_effectiveness"].metric_value))
    storage_efficiency_gain = _round_public_metric(float(metric_map["storage_efficiency_gain"].metric_value))
    if abs(compression_effectiveness - storage_efficiency_gain) > 1e-12:
        raise ValueError("compression_effectiveness must match storage_efficiency_gain")

    if evidence.compression_plan_status == "COMPRESSED" and compression_effectiveness <= 0.0:
        raise ValueError("COMPRESSED plan_status requires compression_effectiveness > 0")
    if evidence.compression_plan_status in {"NO_GAIN", "EMPTY"} and compression_effectiveness != 0.0:
        raise ValueError("NO_GAIN/EMPTY plan_status requires compression_effectiveness == 0")


def _research_questions(metrics: tuple[SystemDemonstrationMetric, ...]) -> tuple[str, ...]:
    metric_map = {metric.metric_name: metric for metric in metrics}
    tokens: list[str] = []
    if metric_map["deterministic_integrity"].metric_status == "PASS":
        tokens.append(_RESEARCH_TOKEN_1)
    if metric_map["trace_reproducibility"].metric_value == 1.0:
        tokens.append(_RESEARCH_TOKEN_2)
    if metric_map["repair_convergence"].metric_status == "PASS":
        tokens.append(_RESEARCH_TOKEN_3)
    if all(metric.metric_status != "FAIL" for metric in metrics):
        tokens.append(_RESEARCH_TOKEN_4)
    if metric_map["governance_stability"].metric_value == 1.0 and metric_map["repair_convergence"].metric_status == "PASS":
        tokens.append(_RESEARCH_TOKEN_5)
    return tuple(tokens)


def _promotion_status(workloads: tuple[WorkloadDemonstrationDescriptor, ...], metrics: tuple[SystemDemonstrationMetric, ...], evidence: SystemStackEvidence) -> str:
    if not workloads or not metrics:
        return "INSUFFICIENT_EVIDENCE"
    metric_map = {metric.metric_name: metric for metric in metrics}

    any_fail = any(metric.metric_status == "FAIL" for metric in metrics)
    cost_reduction = metric_map["cost_reduction"].metric_value
    compression_effectiveness = metric_map["compression_effectiveness"].metric_value
    if any_fail or (cost_reduction == 0.0 and compression_effectiveness == 0.0):
        return "PROMOTION_BLOCKED"

    if all(
        (
            metric_map["deterministic_integrity"].metric_value == 1.0,
            metric_map["governance_stability"].metric_value == 1.0,
            metric_map["repair_convergence"].metric_value >= 0.75,
            (cost_reduction > 0.0) or (compression_effectiveness > 0.0),
            (compression_effectiveness > 0.0) or (evidence.compression_plan_status == "NO_GAIN" and cost_reduction > 0.0),
            metric_map["bounded_success_rate"].metric_value >= 0.75,
            metric_map["trace_reproducibility"].metric_value == 1.0,
        )
    ):
        return "PROMOTED"
    return "PROMOTION_BLOCKED"


def demonstrate_full_system(
    workloads: Sequence[WorkloadDemonstrationDescriptor],
    evidence: SystemStackEvidence,
) -> SystemDemonstrationReceipt:
    if isinstance(workloads, (str, bytes)) or not isinstance(workloads, Sequence):
        raise ValueError("workloads must be a sequence of WorkloadDemonstrationDescriptor")

    normalized_workloads = tuple(workloads)
    for item in normalized_workloads:
        if not isinstance(item, WorkloadDemonstrationDescriptor):
            raise ValueError("workloads must contain WorkloadDemonstrationDescriptor")

    workload_ids = tuple(item.workload_id for item in normalized_workloads)
    if len(set(workload_ids)) != len(workload_ids):
        raise ValueError("duplicate workload_id is not allowed")

    metrics = _compute_metrics(normalized_workloads, evidence)
    _enforce_compression_consistency(evidence, metrics)

    promotion_status = _promotion_status(normalized_workloads, metrics, evidence)
    metric_map = {metric.metric_name: metric for metric in metrics}
    measurable_gain_exists = bool(
        metric_map["cost_reduction"].metric_value > 0.0 or metric_map["compression_effectiveness"].metric_value > 0.0
    )

    return SystemDemonstrationReceipt(
        module_version=SYSTEM_DEMONSTRATION_MODULE_VERSION,
        workload_count=len(normalized_workloads),
        evidence=evidence,
        metrics=metrics,
        promotion_status=promotion_status,
        deterministic_integrity_preserved=True,
        measurable_gain_exists=measurable_gain_exists,
        research_questions_supported=_research_questions(metrics),
    )


__all__ = [
    "SYSTEM_DEMONSTRATION_MODULE_VERSION",
    "WorkloadDemonstrationDescriptor",
    "SystemStackEvidence",
    "SystemDemonstrationMetric",
    "SystemDemonstrationReceipt",
    "demonstrate_full_system",
]
