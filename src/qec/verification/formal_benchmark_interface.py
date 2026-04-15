"""v137.21.3 — Formal Benchmark Interface.

Deterministic interface layer that converts benchmark evidence into a
machine-checkable report and merge/CI gate receipt.

This module is additive and decoder-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Tuple

INTERFACE_VERSION = "v137.21.3"

CATEGORY_ORDER: Tuple[str, ...] = (
    "logical_correctness",
    "physical_timing",
    "replay_integrity",
    "proof_contracts",
    "suppression_integrity",
    "equivalence",
    "benchmark_acceptance",
)

DECISION_PASS = "pass"
DECISION_WARN = "warn"
DECISION_FAIL = "fail"

MERGE_READY = "merge_ready"
MERGE_BLOCKED = "merge_blocked"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _normalize_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool")
    return value


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _normalize_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    normalized: Dict[str, Any] = {}
    for key in sorted(value.keys(), key=lambda x: str(x)):
        skey = str(key)
        if skey in normalized:
            raise ValueError(f"{field} contains duplicate canonical key {skey!r}")
        normalized[skey] = value[key]
    return normalized


def _normalize_category(value: Any) -> str:
    text = _normalize_text(value, field="category").lower().strip()
    if text not in CATEGORY_ORDER:
        raise ValueError(f"category must be one of {CATEGORY_ORDER}")
    return text


def _compare(observed: float, threshold: float, comparator: str) -> bool:
    if comparator == ">=":
        return observed >= threshold
    if comparator == "<=":
        return observed <= threshold
    if comparator == "==":
        return observed == threshold
    raise ValueError("comparator must be one of: >=, <=, ==")


@dataclass(frozen=True)
class FormalBenchmarkCheck:
    name: str
    category: str
    required: bool
    passed: bool
    severity: str
    observed_value: float
    threshold_value: float
    comparator: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "required": self.required,
            "passed": self.passed,
            "severity": self.severity,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "comparator": self.comparator,
            "explanation": self.explanation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FormalBenchmarkThresholdSet:
    replay_integrity_min: float
    proof_contract_completeness_min: float
    latency_compliance_min: float
    throughput_compliance_min: float
    equivalence_required: bool
    suppression_receipt_completeness_min: float
    benchmark_acceptance_floor: float
    benchmark_acceptance_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_integrity_min": self.replay_integrity_min,
            "proof_contract_completeness_min": self.proof_contract_completeness_min,
            "latency_compliance_min": self.latency_compliance_min,
            "throughput_compliance_min": self.throughput_compliance_min,
            "equivalence_required": self.equivalence_required,
            "suppression_receipt_completeness_min": self.suppression_receipt_completeness_min,
            "benchmark_acceptance_floor": self.benchmark_acceptance_floor,
            "benchmark_acceptance_required": self.benchmark_acceptance_required,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FormalBenchmarkInterfaceReport:
    checks: Tuple[FormalBenchmarkCheck, ...]
    category_summaries: Dict[str, Dict[str, int]]
    counts_by_status: Dict[str, int]
    failing_required_checks: Tuple[str, ...]
    advisory_warnings: Tuple[str, ...]
    logical_gate_passed: bool
    physical_gate_passed: bool
    replay_gate_passed: bool
    proof_gate_passed: bool
    overall_decision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "category_summaries": {name: dict(self.category_summaries[name]) for name in CATEGORY_ORDER},
            "counts_by_status": dict(self.counts_by_status),
            "failing_required_checks": list(self.failing_required_checks),
            "advisory_warnings": list(self.advisory_warnings),
            "logical_gate_passed": self.logical_gate_passed,
            "physical_gate_passed": self.physical_gate_passed,
            "replay_gate_passed": self.replay_gate_passed,
            "proof_gate_passed": self.proof_gate_passed,
            "overall_decision": self.overall_decision,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FormalBenchmarkGateReceipt:
    version: str
    gate_decision: str
    report_hash: str
    input_digests: Dict[str, str]
    merge_readiness: str
    merge_ready: bool
    ci_gate_status: str
    rationale: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "gate_decision": self.gate_decision,
            "report_hash": self.report_hash,
            "input_digests": dict(self.input_digests),
            "merge_readiness": self.merge_readiness,
            "merge_ready": self.merge_ready,
            "ci_gate_status": self.ci_gate_status,
            "rationale": list(self.rationale),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FormalBenchmarkInterface:
    thresholds: FormalBenchmarkThresholdSet

    def to_dict(self) -> Dict[str, Any]:
        return {"thresholds": self.thresholds.to_dict()}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())

    @staticmethod
    def build_threshold_set(raw: Any) -> FormalBenchmarkThresholdSet:
        mapping = _normalize_mapping(raw, field="thresholds")
        required_keys = (
            "replay_integrity_min",
            "proof_contract_completeness_min",
            "latency_compliance_min",
            "throughput_compliance_min",
            "equivalence_required",
            "suppression_receipt_completeness_min",
            "benchmark_acceptance_floor",
        )
        for key in required_keys:
            if key not in mapping:
                raise ValueError(f"thresholds missing required field: {key}")
        return FormalBenchmarkThresholdSet(
            replay_integrity_min=_normalize_float(mapping["replay_integrity_min"], field="replay_integrity_min"),
            proof_contract_completeness_min=_normalize_float(
                mapping["proof_contract_completeness_min"], field="proof_contract_completeness_min"
            ),
            latency_compliance_min=_normalize_float(mapping["latency_compliance_min"], field="latency_compliance_min"),
            throughput_compliance_min=_normalize_float(
                mapping["throughput_compliance_min"], field="throughput_compliance_min"
            ),
            equivalence_required=_normalize_bool(mapping["equivalence_required"], field="equivalence_required"),
            suppression_receipt_completeness_min=_normalize_float(
                mapping["suppression_receipt_completeness_min"], field="suppression_receipt_completeness_min"
            ),
            benchmark_acceptance_floor=_normalize_float(
                mapping["benchmark_acceptance_floor"], field="benchmark_acceptance_floor"
            ),
            benchmark_acceptance_required=_normalize_bool(
                mapping.get("benchmark_acceptance_required", False),
                field="benchmark_acceptance_required",
            ),
        )

    @staticmethod
    def normalize_inputs(
        benchmark_summary: Any,
        proof_contract_receipts: Any,
        suppression_receipts: Any,
        latency_throughput_receipts: Any,
        equivalence_checks: Any,
    ) -> Dict[str, Dict[str, Any]]:
        bench = _normalize_mapping(benchmark_summary, field="benchmark_summary")
        proof = _normalize_mapping(proof_contract_receipts, field="proof_contract_receipts")
        suppression = _normalize_mapping(suppression_receipts, field="suppression_receipts")
        timing = _normalize_mapping(latency_throughput_receipts, field="latency_throughput_receipts")
        equivalence = _normalize_mapping(equivalence_checks, field="equivalence_checks")

        required_fields = {
            "benchmark_summary": ("logical_pass_ratio", "benchmark_pass_ratio"),
            "proof_contract_receipts": ("completeness_ratio",),
            "suppression_receipts": ("completeness_ratio",),
            "latency_throughput_receipts": ("latency_compliance_ratio", "throughput_compliance_ratio"),
            "equivalence_checks": ("replay_integrity_ratio", "offline_realtime_equivalent"),
        }

        groups = {
            "benchmark_summary": bench,
            "proof_contract_receipts": proof,
            "suppression_receipts": suppression,
            "latency_throughput_receipts": timing,
            "equivalence_checks": equivalence,
        }
        for group_name, fields in required_fields.items():
            mapping = groups[group_name]
            for field in fields:
                if field not in mapping:
                    raise ValueError(f"{group_name} missing required field: {field}")

        normalized = {
            "benchmark_summary": {
                "logical_pass_ratio": _normalize_float(bench["logical_pass_ratio"], field="logical_pass_ratio"),
                "benchmark_pass_ratio": _normalize_float(bench["benchmark_pass_ratio"], field="benchmark_pass_ratio"),
            },
            "proof_contract_receipts": {
                "completeness_ratio": _normalize_float(proof["completeness_ratio"], field="proof completeness_ratio"),
            },
            "suppression_receipts": {
                "completeness_ratio": _normalize_float(
                    suppression["completeness_ratio"], field="suppression completeness_ratio"
                ),
            },
            "latency_throughput_receipts": {
                "latency_compliance_ratio": _normalize_float(
                    timing["latency_compliance_ratio"], field="latency_compliance_ratio"
                ),
                "throughput_compliance_ratio": _normalize_float(
                    timing["throughput_compliance_ratio"], field="throughput_compliance_ratio"
                ),
            },
            "equivalence_checks": {
                "replay_integrity_ratio": _normalize_float(
                    equivalence["replay_integrity_ratio"], field="replay_integrity_ratio"
                ),
                "offline_realtime_equivalent": _normalize_bool(
                    equivalence["offline_realtime_equivalent"], field="offline_realtime_equivalent"
                ),
            },
        }
        return normalized

    @staticmethod
    def _build_check(
        *,
        name: str,
        category: str,
        required: bool,
        observed_value: float,
        threshold_value: float,
        comparator: str,
        explanation: str,
    ) -> FormalBenchmarkCheck:
        normalized_category = _normalize_category(category)
        passed = _compare(observed_value, threshold_value, comparator)
        severity = "required" if required else "advisory"
        return FormalBenchmarkCheck(
            name=_normalize_text(name, field="name"),
            category=normalized_category,
            required=required,
            passed=passed,
            severity=severity,
            observed_value=observed_value,
            threshold_value=threshold_value,
            comparator=comparator,
            explanation=_normalize_text(explanation, field="explanation"),
        )

    def evaluate(
        self,
        benchmark_summary: Any,
        proof_contract_receipts: Any,
        suppression_receipts: Any,
        latency_throughput_receipts: Any,
        equivalence_checks: Any,
    ) -> Tuple[FormalBenchmarkInterfaceReport, FormalBenchmarkGateReceipt]:
        normalized = self.normalize_inputs(
            benchmark_summary=benchmark_summary,
            proof_contract_receipts=proof_contract_receipts,
            suppression_receipts=suppression_receipts,
            latency_throughput_receipts=latency_throughput_receipts,
            equivalence_checks=equivalence_checks,
        )

        checks: Tuple[FormalBenchmarkCheck, ...] = (
            self._build_check(
                name="logical_pass_ratio",
                category="logical_correctness",
                required=True,
                observed_value=normalized["benchmark_summary"]["logical_pass_ratio"],
                threshold_value=1.0,
                comparator=">=",
                explanation="Logical correctness requires full pass ratio.",
            ),
            self._build_check(
                name="latency_compliance_ratio",
                category="physical_timing",
                required=True,
                observed_value=normalized["latency_throughput_receipts"]["latency_compliance_ratio"],
                threshold_value=self.thresholds.latency_compliance_min,
                comparator=">=",
                explanation="Physical timing latency budget compliance.",
            ),
            self._build_check(
                name="throughput_compliance_ratio",
                category="physical_timing",
                required=True,
                observed_value=normalized["latency_throughput_receipts"]["throughput_compliance_ratio"],
                threshold_value=self.thresholds.throughput_compliance_min,
                comparator=">=",
                explanation="Physical timing throughput budget compliance.",
            ),
            self._build_check(
                name="replay_integrity_ratio",
                category="replay_integrity",
                required=True,
                observed_value=normalized["equivalence_checks"]["replay_integrity_ratio"],
                threshold_value=self.thresholds.replay_integrity_min,
                comparator=">=",
                explanation="Replay integrity must satisfy deterministic floor.",
            ),
            self._build_check(
                name="proof_contract_completeness",
                category="proof_contracts",
                required=True,
                observed_value=normalized["proof_contract_receipts"]["completeness_ratio"],
                threshold_value=self.thresholds.proof_contract_completeness_min,
                comparator=">=",
                explanation="Proof and contract receipts must be complete.",
            ),
            self._build_check(
                name="suppression_receipt_completeness",
                category="suppression_integrity",
                required=True,
                observed_value=normalized["suppression_receipts"]["completeness_ratio"],
                threshold_value=self.thresholds.suppression_receipt_completeness_min,
                comparator=">=",
                explanation="Suppression-before-correction receipts must be complete.",
            ),
            self._build_check(
                name="offline_realtime_equivalence",
                category="equivalence",
                required=True,
                observed_value=1.0 if normalized["equivalence_checks"]["offline_realtime_equivalent"] else 0.0,
                threshold_value=1.0 if self.thresholds.equivalence_required else 0.0,
                comparator=">=",
                explanation="Offline and realtime outputs must remain equivalent when required.",
            ),
            self._build_check(
                name="benchmark_pass_ratio",
                category="benchmark_acceptance",
                required=self.thresholds.benchmark_acceptance_required,
                observed_value=normalized["benchmark_summary"]["benchmark_pass_ratio"],
                threshold_value=self.thresholds.benchmark_acceptance_floor,
                comparator=">=",
                explanation="Benchmark acceptance floor check for governance tracking.",
            ),
        )

        deterministic_checks = tuple(sorted(checks, key=lambda c: (CATEGORY_ORDER.index(c.category), c.name)))

        category_summaries: Dict[str, Dict[str, int]] = {
            category: {"total": 0, "passed": 0, "failed": 0, "required_failed": 0, "advisory_failed": 0}
            for category in CATEGORY_ORDER
        }
        counts_by_status = {"passed": 0, "failed": 0, "required_failed": 0, "advisory_failed": 0}
        failing_required: list[str] = []
        advisory_warnings: list[str] = []

        for check in deterministic_checks:
            summary = category_summaries[check.category]
            summary["total"] += 1
            if check.passed:
                summary["passed"] += 1
                counts_by_status["passed"] += 1
            else:
                summary["failed"] += 1
                counts_by_status["failed"] += 1
                if check.required:
                    summary["required_failed"] += 1
                    counts_by_status["required_failed"] += 1
                    failing_required.append(check.name)
                else:
                    summary["advisory_failed"] += 1
                    counts_by_status["advisory_failed"] += 1
                    advisory_warnings.append(check.name)

        logical_gate_passed = all(
            check.passed
            for check in deterministic_checks
            if check.category in ("logical_correctness", "equivalence", "benchmark_acceptance") and check.required
        )
        physical_gate_passed = all(
            check.passed for check in deterministic_checks if check.category == "physical_timing" and check.required
        )
        replay_gate_passed = all(
            check.passed for check in deterministic_checks if check.category == "replay_integrity" and check.required
        )
        proof_gate_passed = all(
            check.passed
            for check in deterministic_checks
            if check.category in ("proof_contracts", "suppression_integrity") and check.required
        )

        if failing_required:
            overall = DECISION_FAIL
        elif advisory_warnings:
            overall = DECISION_WARN
        else:
            overall = DECISION_PASS

        report = FormalBenchmarkInterfaceReport(
            checks=deterministic_checks,
            category_summaries=category_summaries,
            counts_by_status=counts_by_status,
            failing_required_checks=tuple(sorted(failing_required)),
            advisory_warnings=tuple(sorted(advisory_warnings)),
            logical_gate_passed=logical_gate_passed,
            physical_gate_passed=physical_gate_passed,
            replay_gate_passed=replay_gate_passed,
            proof_gate_passed=proof_gate_passed,
            overall_decision=overall,
        )

        input_digests = {
            "benchmark_summary": _stable_hash(normalized["benchmark_summary"]),
            "proof_contract_receipts": _stable_hash(normalized["proof_contract_receipts"]),
            "suppression_receipts": _stable_hash(normalized["suppression_receipts"]),
            "latency_throughput_receipts": _stable_hash(normalized["latency_throughput_receipts"]),
            "equivalence_checks": _stable_hash(normalized["equivalence_checks"]),
            "thresholds": self.thresholds.stable_hash(),
        }
        input_digests["aggregate"] = _stable_hash(input_digests)

        merge_ready = overall != DECISION_FAIL
        receipt = FormalBenchmarkGateReceipt(
            version=INTERFACE_VERSION,
            gate_decision=overall,
            report_hash=report.stable_hash(),
            input_digests=input_digests,
            merge_readiness=MERGE_READY if merge_ready else MERGE_BLOCKED,
            merge_ready=merge_ready,
            ci_gate_status=DECISION_PASS if merge_ready else DECISION_FAIL,
            rationale=tuple(
                sorted(
                    (
                        [f"required_failure:{name}" for name in report.failing_required_checks]
                        + [f"advisory_warning:{name}" for name in report.advisory_warnings]
                        + [f"overall_decision:{report.overall_decision}"]
                    )
                )
            ),
        )
        return report, receipt


def run_formal_benchmark_interface(
    *,
    benchmark_summary: Any,
    proof_contract_receipts: Any,
    suppression_receipts: Any,
    latency_throughput_receipts: Any,
    equivalence_checks: Any,
    thresholds: Any,
) -> Tuple[FormalBenchmarkInterfaceReport, FormalBenchmarkGateReceipt]:
    threshold_set = (
        thresholds
        if isinstance(thresholds, FormalBenchmarkThresholdSet)
        else FormalBenchmarkInterface.build_threshold_set(thresholds)
    )
    interface = FormalBenchmarkInterface(thresholds=threshold_set)
    return interface.evaluate(
        benchmark_summary=benchmark_summary,
        proof_contract_receipts=proof_contract_receipts,
        suppression_receipts=suppression_receipts,
        latency_throughput_receipts=latency_throughput_receipts,
        equivalence_checks=equivalence_checks,
    )


def summarize_formal_benchmark_report(report: FormalBenchmarkInterfaceReport) -> Dict[str, Any]:
    return {
        "overall_decision": report.overall_decision,
        "required_failures": list(report.failing_required_checks),
        "advisory_warnings": list(report.advisory_warnings),
        "logical_gate_passed": report.logical_gate_passed,
        "physical_gate_passed": report.physical_gate_passed,
        "replay_gate_passed": report.replay_gate_passed,
        "proof_gate_passed": report.proof_gate_passed,
    }


def validate_formal_benchmark_report(report: Any) -> Dict[str, Any]:
    violations: list[str] = []
    if not isinstance(report, FormalBenchmarkInterfaceReport):
        return {"valid": False, "violations": ("wrong_type",)}
    if report.overall_decision not in (DECISION_PASS, DECISION_WARN, DECISION_FAIL):
        violations.append("invalid_overall_decision")
    if any(check.category not in CATEGORY_ORDER for check in report.checks):
        violations.append("invalid_check_category")
    if report.failing_required_checks != tuple(sorted(report.failing_required_checks)):
        violations.append("required_failures_not_sorted")
    return {"valid": not violations, "violations": tuple(violations)}
