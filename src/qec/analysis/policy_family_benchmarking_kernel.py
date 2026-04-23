"""v145.3 — Policy Family Benchmarking Kernel (PFBK)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import (
    SimulationConfig,
    ensure_stable_hash,
    round12,
    validate_sha256_hex,
    validate_unit_interval,
)
from qec.analysis.governed_closed_loop_simulation import GovernedClosedLoopReceipt, run_governed_closed_loop
from qec.analysis.governed_orchestration_layer import GovernancePolicy

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

MAX_BENCHMARK_FAMILY_SIZE = 64
FAMILY_VARIATION_MODERATE_THRESHOLD = 0.25
FAMILY_VARIATION_HIGH_THRESHOLD = 0.50

BENCHMARK_RELATION_EQUIVALENT = "equivalent"
BENCHMARK_RELATION_MORE_PERMISSIVE = "more_permissive"
BENCHMARK_RELATION_MORE_RESTRICTIVE = "more_restrictive"
BENCHMARK_RELATION_MIXED = "mixed"

FAMILY_CLASS_STABLE = "stable_family"
FAMILY_CLASS_MIXED = "mixed_family"
FAMILY_CLASS_HIGH_VARIATION = "high_variation_family"

_ALLOWED_SWEEP_PARAMETERS = frozenset(
    {
        "min_required_score",
        "min_required_confidence",
        "min_required_margin",
        "min_required_convergence",
        "allow_tie_break",
        "allow_no_improvement",
        "require_stable_transition",
    }
)
_BOOLEAN_SWEEP_PARAMETERS = frozenset(
    {
        "allow_tie_break",
        "allow_no_improvement",
        "require_stable_transition",
    }
)
_ALLOWED_BENCHMARK_RELATIONS = frozenset(
    {
        BENCHMARK_RELATION_EQUIVALENT,
        BENCHMARK_RELATION_MORE_PERMISSIVE,
        BENCHMARK_RELATION_MORE_RESTRICTIVE,
        BENCHMARK_RELATION_MIXED,
    }
)
_ALLOWED_FAMILY_CLASSIFICATIONS = frozenset(
    {FAMILY_CLASS_STABLE, FAMILY_CLASS_MIXED, FAMILY_CLASS_HIGH_VARIATION}
)


def _is_strict_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _relation_for_deltas(*, admissible_delta: int, reject_delta: int, convergence_delta: float) -> str:
    if admissible_delta == 0 and reject_delta == 0 and convergence_delta == 0.0:
        return BENCHMARK_RELATION_EQUIVALENT
    if admissible_delta > 0 and reject_delta <= 0:
        return BENCHMARK_RELATION_MORE_PERMISSIVE
    if admissible_delta < 0 and reject_delta >= 0:
        return BENCHMARK_RELATION_MORE_RESTRICTIVE
    return BENCHMARK_RELATION_MIXED


def _choose_by_extrema(
    records: tuple["GeneratedPolicyRecord", ...],
    *,
    metric_name: str,
    invert: bool,
) -> str:
    if invert:
        selected = min(records, key=lambda rec: (-getattr(rec, metric_name), rec.policy_hash))
        return selected.policy_hash
    selected = min(records, key=lambda rec: (getattr(rec, metric_name), rec.policy_hash))
    return selected.policy_hash


def _variation_score(
    records: tuple["GeneratedPolicyRecord", ...],
    comparisons: tuple["PolicyBenchmarkComparison", ...],
) -> float:
    if not comparisons:
        return 0.0
    cycle_count = max(
        (record.allow_count + record.hold_count + record.reject_count for record in records),
        default=1,
    )
    if cycle_count < 1:
        cycle_count = 1
    max_delta = 0.0
    for comparison in comparisons:
        normalized_admissible = abs(comparison.admissible_delta) / float(cycle_count)
        normalized_reject = abs(comparison.reject_delta) / float(cycle_count)
        normalized_convergence = abs(comparison.convergence_delta)
        max_delta = max(max_delta, normalized_admissible, normalized_reject, normalized_convergence)
    return round12(max_delta)


@dataclass(frozen=True)
class PolicySweepAxis:
    parameter_name: str
    values: tuple[float | bool, ...]
    stable_hash: str

    def __post_init__(self) -> None:
        if self.parameter_name not in _ALLOWED_SWEEP_PARAMETERS:
            raise ValueError("parameter_name is invalid")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("values must be a non-empty tuple")

        normalized_values: list[float | bool] = []
        seen: set[float | bool] = set()
        if self.parameter_name in _BOOLEAN_SWEEP_PARAMETERS:
            for value in self.values:
                if not isinstance(value, bool):
                    raise ValueError("boolean sweep axis values must be strict bool")
                if value in seen:
                    raise ValueError("values must not contain duplicates")
                seen.add(value)
                normalized_values.append(value)
        else:
            for value in self.values:
                if not _is_strict_numeric(value):
                    raise ValueError("numeric sweep axis values must be strict numeric")
                numeric = round12(validate_unit_interval(float(value), "axis_value"))
                if numeric in seen:
                    raise ValueError("values must not contain duplicates")
                seen.add(numeric)
                normalized_values.append(numeric)

        object.__setattr__(self, "values", tuple(normalized_values))
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        values: tuple[_JSONValue, ...] = tuple(
            round12(float(value)) if _is_strict_numeric(value) else bool(value) for value in self.values
        )
        return {
            "parameter_name": self.parameter_name,
            "values": values,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class PolicyFamilySpec:
    axes: tuple[PolicySweepAxis, ...]
    max_family_size: int
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.axes, tuple) or not self.axes:
            raise ValueError("axes must be a non-empty tuple")
        if any(not isinstance(axis, PolicySweepAxis) for axis in self.axes):
            raise ValueError("axes must contain PolicySweepAxis entries")
        for axis in self.axes:
            ensure_stable_hash(axis, "axis")
        sorted_axes = tuple(sorted(self.axes, key=lambda axis: axis.parameter_name))
        object.__setattr__(self, "axes", sorted_axes)
        axis_names = tuple(axis.parameter_name for axis in self.axes)
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("duplicate parameter_name across axes")
        if not isinstance(self.max_family_size, int) or isinstance(self.max_family_size, bool) or self.max_family_size < 1:
            raise ValueError("max_family_size must be int >= 1")
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "axes": tuple(axis.to_dict() for axis in self.axes),
            "max_family_size": self.max_family_size,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GeneratedPolicyRecord:
    policy_hash: str
    parameter_overrides: tuple[tuple[str, float | bool], ...]
    governed_receipt_hash: str
    allow_count: int
    hold_count: int
    reject_count: int
    admissible_count: int
    non_admissible_count: int
    mean_convergence_metric: float
    stable_hash: str

    def __post_init__(self) -> None:
        validate_sha256_hex(self.policy_hash, "policy_hash")
        validate_sha256_hex(self.governed_receipt_hash, "governed_receipt_hash")
        if not isinstance(self.parameter_overrides, tuple):
            raise ValueError("parameter_overrides must be tuple")
        for item in self.parameter_overrides:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("parameter_overrides must be tuple[(str, float|bool), ...]")
            parameter_name, value = item
            if parameter_name not in _ALLOWED_SWEEP_PARAMETERS:
                raise ValueError("parameter_overrides contains invalid parameter")
            if parameter_name in _BOOLEAN_SWEEP_PARAMETERS:
                if not isinstance(value, bool):
                    raise ValueError("boolean parameter override must be bool")
            else:
                if not _is_strict_numeric(value):
                    raise ValueError("numeric parameter override must be numeric")
                validate_unit_interval(float(value), "parameter_override")
        sorted_overrides = tuple(sorted(self.parameter_overrides, key=lambda item: item[0]))
        if self.parameter_overrides != sorted_overrides:
            raise ValueError("parameter_overrides must be sorted by parameter_name")
        for field_name in (
            "allow_count",
            "hold_count",
            "reject_count",
            "admissible_count",
            "non_admissible_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be non-negative int")
        cycle_count = self.allow_count + self.hold_count + self.reject_count
        if self.admissible_count + self.non_admissible_count != cycle_count:
            raise ValueError("admissible/non_admissible count inconsistency")
        object.__setattr__(
            self,
            "mean_convergence_metric",
            validate_unit_interval(self.mean_convergence_metric, "mean_convergence_metric"),
        )
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy_hash": self.policy_hash,
            "parameter_overrides": tuple(
                (name, round12(float(value)) if _is_strict_numeric(value) else bool(value))
                for name, value in self.parameter_overrides
            ),
            "governed_receipt_hash": self.governed_receipt_hash,
            "allow_count": self.allow_count,
            "hold_count": self.hold_count,
            "reject_count": self.reject_count,
            "admissible_count": self.admissible_count,
            "non_admissible_count": self.non_admissible_count,
            "mean_convergence_metric": round12(self.mean_convergence_metric),
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class PolicyBenchmarkComparison:
    left_policy_hash: str
    right_policy_hash: str
    admissible_delta: int
    reject_delta: int
    convergence_delta: float
    benchmark_relation: str
    stable_hash: str

    def __post_init__(self) -> None:
        validate_sha256_hex(self.left_policy_hash, "left_policy_hash")
        validate_sha256_hex(self.right_policy_hash, "right_policy_hash")
        if self.left_policy_hash >= self.right_policy_hash:
            raise ValueError("comparison policy hashes must be canonical ascending order")
        for field_name in ("admissible_delta", "reject_delta"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be int")
        if self.benchmark_relation not in _ALLOWED_BENCHMARK_RELATIONS:
            raise ValueError("benchmark_relation is invalid")
        if isinstance(self.convergence_delta, bool) or not isinstance(self.convergence_delta, (int, float)):
            raise ValueError("convergence_delta must be numeric")
        convergence_delta = float(self.convergence_delta)
        if not math.isfinite(convergence_delta) or not (-1.0 <= convergence_delta <= 1.0):
            raise ValueError("convergence_delta must be finite and in [-1,1]")
        object.__setattr__(self, "convergence_delta", round12(convergence_delta))
        expected_relation = _relation_for_deltas(
            admissible_delta=self.admissible_delta,
            reject_delta=self.reject_delta,
            convergence_delta=self.convergence_delta,
        )
        if self.benchmark_relation != expected_relation:
            raise ValueError("benchmark_relation mismatch")
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "left_policy_hash": self.left_policy_hash,
            "right_policy_hash": self.right_policy_hash,
            "admissible_delta": self.admissible_delta,
            "reject_delta": self.reject_delta,
            "convergence_delta": round12(self.convergence_delta),
            "benchmark_relation": self.benchmark_relation,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class PolicyFamilyBenchmarkSummary:
    family_size: int
    comparison_count: int
    most_permissive_policy_hash: str
    most_restrictive_policy_hash: str
    highest_convergence_policy_hash: str
    lowest_convergence_policy_hash: str
    family_behavior_classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.family_size, int) or isinstance(self.family_size, bool) or self.family_size < 1:
            raise ValueError("family_size must be int >= 1")
        if not isinstance(self.comparison_count, int) or isinstance(self.comparison_count, bool) or self.comparison_count < 0:
            raise ValueError("comparison_count must be int >= 0")
        expected_comparisons = self.family_size * (self.family_size - 1) // 2
        if self.comparison_count != expected_comparisons:
            raise ValueError("comparison_count mismatch")
        validate_sha256_hex(self.most_permissive_policy_hash, "most_permissive_policy_hash")
        validate_sha256_hex(self.most_restrictive_policy_hash, "most_restrictive_policy_hash")
        validate_sha256_hex(self.highest_convergence_policy_hash, "highest_convergence_policy_hash")
        validate_sha256_hex(self.lowest_convergence_policy_hash, "lowest_convergence_policy_hash")
        if self.family_behavior_classification not in _ALLOWED_FAMILY_CLASSIFICATIONS:
            raise ValueError("family_behavior_classification is invalid")
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "family_size": self.family_size,
            "comparison_count": self.comparison_count,
            "most_permissive_policy_hash": self.most_permissive_policy_hash,
            "most_restrictive_policy_hash": self.most_restrictive_policy_hash,
            "highest_convergence_policy_hash": self.highest_convergence_policy_hash,
            "lowest_convergence_policy_hash": self.lowest_convergence_policy_hash,
            "family_behavior_classification": self.family_behavior_classification,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class PolicyFamilyBenchmarkReceipt:
    config: SimulationConfig
    baseline_policy_hash: str
    family_spec: PolicyFamilySpec
    generated_policy_records: tuple[GeneratedPolicyRecord, ...]
    comparison_records: tuple[PolicyBenchmarkComparison, ...]
    summary: PolicyFamilyBenchmarkSummary
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("config must be SimulationConfig")
        ensure_stable_hash(self.config, "config")
        validate_sha256_hex(self.baseline_policy_hash, "baseline_policy_hash")
        if not isinstance(self.family_spec, PolicyFamilySpec):
            raise ValueError("family_spec must be PolicyFamilySpec")
        ensure_stable_hash(self.family_spec, "family_spec")
        if not isinstance(self.generated_policy_records, tuple) or any(
            not isinstance(item, GeneratedPolicyRecord) for item in self.generated_policy_records
        ):
            raise ValueError("generated_policy_records must be tuple[GeneratedPolicyRecord, ...]")
        if not self.generated_policy_records:
            raise ValueError("generated_policy_records must not be empty")
        sorted_records = tuple(sorted(self.generated_policy_records, key=lambda item: item.policy_hash))
        if self.generated_policy_records != sorted_records:
            raise ValueError("generated_policy_records must be sorted by policy_hash")
        policy_hashes = tuple(record.policy_hash for record in self.generated_policy_records)
        if len(set(policy_hashes)) != len(policy_hashes):
            raise ValueError("generated_policy_records contains duplicate policy_hash")

        if not isinstance(self.comparison_records, tuple) or any(
            not isinstance(item, PolicyBenchmarkComparison) for item in self.comparison_records
        ):
            raise ValueError("comparison_records must be tuple[PolicyBenchmarkComparison, ...]")
        expected_pair_count = len(self.generated_policy_records) * (len(self.generated_policy_records) - 1) // 2
        if len(self.comparison_records) != expected_pair_count:
            raise ValueError("comparison_records length mismatch")
        sorted_pairs = tuple(sorted(self.comparison_records, key=lambda item: (item.left_policy_hash, item.right_policy_hash)))
        if self.comparison_records != sorted_pairs:
            raise ValueError("comparison_records must be sorted by policy-hash pair")

        expected_pairs = {
            (policy_hashes[i], policy_hashes[j])
            for i in range(len(policy_hashes))
            for j in range(i + 1, len(policy_hashes))
        }
        actual_pairs = {(item.left_policy_hash, item.right_policy_hash) for item in self.comparison_records}
        if actual_pairs != expected_pairs:
            raise ValueError("comparison_records must cover all unordered policy pairs exactly once")

        if not isinstance(self.summary, PolicyFamilyBenchmarkSummary):
            raise ValueError("summary must be PolicyFamilyBenchmarkSummary")
        ensure_stable_hash(self.summary, "summary")
        if self.summary.family_size != len(self.generated_policy_records):
            raise ValueError("summary family_size mismatch")
        if self.summary.comparison_count != len(self.comparison_records):
            raise ValueError("summary comparison_count mismatch")

        most_permissive = _choose_by_extrema(self.generated_policy_records, metric_name="admissible_count", invert=True)
        most_restrictive = _choose_by_extrema(self.generated_policy_records, metric_name="admissible_count", invert=False)
        highest_convergence = _choose_by_extrema(
            self.generated_policy_records,
            metric_name="mean_convergence_metric",
            invert=True,
        )
        lowest_convergence = _choose_by_extrema(
            self.generated_policy_records,
            metric_name="mean_convergence_metric",
            invert=False,
        )

        score = _variation_score(self.generated_policy_records, self.comparison_records)
        if score >= FAMILY_VARIATION_HIGH_THRESHOLD:
            family_class = FAMILY_CLASS_HIGH_VARIATION
        elif score >= FAMILY_VARIATION_MODERATE_THRESHOLD:
            family_class = FAMILY_CLASS_MIXED
        else:
            family_class = FAMILY_CLASS_STABLE

        if self.summary.most_permissive_policy_hash != most_permissive:
            raise ValueError("most_permissive_policy_hash mismatch")
        if self.summary.most_restrictive_policy_hash != most_restrictive:
            raise ValueError("most_restrictive_policy_hash mismatch")
        if self.summary.highest_convergence_policy_hash != highest_convergence:
            raise ValueError("highest_convergence_policy_hash mismatch")
        if self.summary.lowest_convergence_policy_hash != lowest_convergence:
            raise ValueError("lowest_convergence_policy_hash mismatch")
        if self.summary.family_behavior_classification != family_class:
            raise ValueError("family_behavior_classification mismatch")

        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "baseline_policy_hash": self.baseline_policy_hash,
            "family_spec": self.family_spec.to_dict(),
            "generated_policy_records": tuple(item.to_dict() for item in self.generated_policy_records),
            "comparison_records": tuple(item.to_dict() for item in self.comparison_records),
            "summary": self.summary.to_dict(),
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _generated_family_capacity(family_spec: PolicyFamilySpec) -> int:
    size = 1
    for axis in family_spec.axes:
        size *= len(axis.values)
    return size


def _build_policy_from_overrides(
    baseline_policy: GovernancePolicy,
    parameter_overrides: tuple[tuple[str, float | bool], ...],
) -> GovernancePolicy:
    policy_fields = {
        "min_required_score": baseline_policy.min_required_score,
        "min_required_confidence": baseline_policy.min_required_confidence,
        "min_required_margin": baseline_policy.min_required_margin,
        "min_required_convergence": baseline_policy.min_required_convergence,
        "allow_tie_break": baseline_policy.allow_tie_break,
        "allow_no_improvement": baseline_policy.allow_no_improvement,
        "require_stable_transition": baseline_policy.require_stable_transition,
    }
    for parameter_name, value in parameter_overrides:
        policy_fields[parameter_name] = value
    payload = {
        "min_required_score": round12(float(policy_fields["min_required_score"])),
        "min_required_confidence": round12(float(policy_fields["min_required_confidence"])),
        "min_required_margin": round12(float(policy_fields["min_required_margin"])),
        "min_required_convergence": round12(float(policy_fields["min_required_convergence"])),
        "allow_tie_break": bool(policy_fields["allow_tie_break"]),
        "allow_no_improvement": bool(policy_fields["allow_no_improvement"]),
        "require_stable_transition": bool(policy_fields["require_stable_transition"]),
    }
    return GovernancePolicy(
        min_required_score=float(policy_fields["min_required_score"]),
        min_required_confidence=float(policy_fields["min_required_confidence"]),
        min_required_margin=float(policy_fields["min_required_margin"]),
        min_required_convergence=float(policy_fields["min_required_convergence"]),
        allow_tie_break=bool(policy_fields["allow_tie_break"]),
        allow_no_improvement=bool(policy_fields["allow_no_improvement"]),
        require_stable_transition=bool(policy_fields["require_stable_transition"]),
        stable_hash=sha256_hex(payload),
    )


def _generate_policy_family(
    baseline_policy: GovernancePolicy,
    family_spec: PolicyFamilySpec,
) -> tuple[tuple[GovernancePolicy, tuple[tuple[str, float | bool], ...]], ...]:
    max_allowed_size = min(family_spec.max_family_size, MAX_BENCHMARK_FAMILY_SIZE)
    generated_count = _generated_family_capacity(family_spec)
    if generated_count > max_allowed_size:
        raise ValueError("generated family size exceeds configured maximum")

    generated: list[tuple[GovernancePolicy, tuple[tuple[str, float | bool], ...]]] = []
    seen_policy_hashes: set[str] = set()

    axis_names = tuple(axis.parameter_name for axis in family_spec.axes)
    axis_values = tuple(axis.values for axis in family_spec.axes)
    for product_values in product(*axis_values):
        overrides = tuple(sorted(zip(axis_names, product_values), key=lambda item: item[0]))
        policy = _build_policy_from_overrides(baseline_policy, overrides)
        ensure_stable_hash(policy, "generated_policy")
        if policy.stable_hash in seen_policy_hashes:
            raise ValueError("duplicate generated policy hashes are not allowed")
        seen_policy_hashes.add(policy.stable_hash)
        generated.append((policy, overrides))

    return tuple(sorted(generated, key=lambda item: item[0].stable_hash))


def _build_generated_policy_record(
    policy_hash: str,
    parameter_overrides: tuple[tuple[str, float | bool], ...],
    governed_receipt: GovernedClosedLoopReceipt,
) -> GeneratedPolicyRecord:
    summary = governed_receipt.summary
    payload = {
        "policy_hash": policy_hash,
        "parameter_overrides": tuple(
            (name, round12(float(value)) if _is_strict_numeric(value) else bool(value))
            for name, value in parameter_overrides
        ),
        "governed_receipt_hash": governed_receipt.stable_hash,
        "allow_count": summary.allow_count,
        "hold_count": summary.hold_count,
        "reject_count": summary.reject_count,
        "admissible_count": summary.admissible_count,
        "non_admissible_count": summary.non_admissible_count,
        "mean_convergence_metric": round12(summary.mean_convergence_metric),
    }
    return GeneratedPolicyRecord(
        policy_hash=policy_hash,
        parameter_overrides=parameter_overrides,
        governed_receipt_hash=governed_receipt.stable_hash,
        allow_count=summary.allow_count,
        hold_count=summary.hold_count,
        reject_count=summary.reject_count,
        admissible_count=summary.admissible_count,
        non_admissible_count=summary.non_admissible_count,
        mean_convergence_metric=summary.mean_convergence_metric,
        stable_hash=sha256_hex(payload),
    )


def _build_comparison_records(
    records: tuple[GeneratedPolicyRecord, ...],
) -> tuple[PolicyBenchmarkComparison, ...]:
    comparisons: list[PolicyBenchmarkComparison] = []
    for left, right in combinations(records, 2):
        admissible_delta = right.admissible_count - left.admissible_count
        reject_delta = right.reject_count - left.reject_count
        convergence_delta = round12(right.mean_convergence_metric - left.mean_convergence_metric)
        relation = _relation_for_deltas(
            admissible_delta=admissible_delta,
            reject_delta=reject_delta,
            convergence_delta=convergence_delta,
        )
        payload = {
            "left_policy_hash": left.policy_hash,
            "right_policy_hash": right.policy_hash,
            "admissible_delta": admissible_delta,
            "reject_delta": reject_delta,
            "convergence_delta": convergence_delta,
            "benchmark_relation": relation,
        }
        comparisons.append(
            PolicyBenchmarkComparison(
                left_policy_hash=left.policy_hash,
                right_policy_hash=right.policy_hash,
                admissible_delta=admissible_delta,
                reject_delta=reject_delta,
                convergence_delta=convergence_delta,
                benchmark_relation=relation,
                stable_hash=sha256_hex(payload),
            )
        )
    return tuple(sorted(comparisons, key=lambda item: (item.left_policy_hash, item.right_policy_hash)))


def _build_summary(
    records: tuple[GeneratedPolicyRecord, ...],
    comparisons: tuple[PolicyBenchmarkComparison, ...],
) -> PolicyFamilyBenchmarkSummary:
    most_permissive = _choose_by_extrema(records, metric_name="admissible_count", invert=True)
    most_restrictive = _choose_by_extrema(records, metric_name="admissible_count", invert=False)
    highest_convergence = _choose_by_extrema(records, metric_name="mean_convergence_metric", invert=True)
    lowest_convergence = _choose_by_extrema(records, metric_name="mean_convergence_metric", invert=False)

    score = _variation_score(records, comparisons)
    if score >= FAMILY_VARIATION_HIGH_THRESHOLD:
        family_class = FAMILY_CLASS_HIGH_VARIATION
    elif score >= FAMILY_VARIATION_MODERATE_THRESHOLD:
        family_class = FAMILY_CLASS_MIXED
    else:
        family_class = FAMILY_CLASS_STABLE

    payload = {
        "family_size": len(records),
        "comparison_count": len(comparisons),
        "most_permissive_policy_hash": most_permissive,
        "most_restrictive_policy_hash": most_restrictive,
        "highest_convergence_policy_hash": highest_convergence,
        "lowest_convergence_policy_hash": lowest_convergence,
        "family_behavior_classification": family_class,
    }
    return PolicyFamilyBenchmarkSummary(
        family_size=len(records),
        comparison_count=len(comparisons),
        most_permissive_policy_hash=most_permissive,
        most_restrictive_policy_hash=most_restrictive,
        highest_convergence_policy_hash=highest_convergence,
        lowest_convergence_policy_hash=lowest_convergence,
        family_behavior_classification=family_class,
        stable_hash=sha256_hex(payload),
    )


def benchmark_policy_family(
    config: SimulationConfig,
    baseline_policy: GovernancePolicy,
    family_spec: PolicyFamilySpec,
) -> PolicyFamilyBenchmarkReceipt:
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be SimulationConfig")
    ensure_stable_hash(config, "config")
    if not isinstance(baseline_policy, GovernancePolicy):
        raise ValueError("baseline_policy must be GovernancePolicy")
    ensure_stable_hash(baseline_policy, "baseline_policy")
    if not isinstance(family_spec, PolicyFamilySpec):
        raise ValueError("family_spec must be PolicyFamilySpec")
    ensure_stable_hash(family_spec, "family_spec")

    generated_family = _generate_policy_family(baseline_policy, family_spec)

    generated_records: list[GeneratedPolicyRecord] = []
    for policy, overrides in generated_family:
        governed_receipt = run_governed_closed_loop(config, policy)
        if not isinstance(governed_receipt, GovernedClosedLoopReceipt):
            raise ValueError("run_governed_closed_loop must return GovernedClosedLoopReceipt")
        ensure_stable_hash(governed_receipt, "governed_receipt")
        if governed_receipt.config.stable_hash != config.stable_hash:
            raise ValueError("governed receipt config mismatch")
        if governed_receipt.policy.stable_hash != policy.stable_hash:
            raise ValueError("governed receipt policy mismatch")
        generated_records.append(_build_generated_policy_record(policy.stable_hash, overrides, governed_receipt))

    sorted_records = tuple(sorted(generated_records, key=lambda item: item.policy_hash))
    comparisons = _build_comparison_records(sorted_records)
    summary = _build_summary(sorted_records, comparisons)

    payload = {
        "config": config.to_dict(),
        "baseline_policy_hash": baseline_policy.stable_hash,
        "family_spec": family_spec.to_dict(),
        "generated_policy_records": tuple(record.to_dict() for record in sorted_records),
        "comparison_records": tuple(record.to_dict() for record in comparisons),
        "summary": summary.to_dict(),
    }
    return PolicyFamilyBenchmarkReceipt(
        config=config,
        baseline_policy_hash=baseline_policy.stable_hash,
        family_spec=family_spec,
        generated_policy_records=sorted_records,
        comparison_records=comparisons,
        summary=summary,
        stable_hash=sha256_hex(payload),
    )


__all__ = [
    "MAX_BENCHMARK_FAMILY_SIZE",
    "FAMILY_VARIATION_MODERATE_THRESHOLD",
    "FAMILY_VARIATION_HIGH_THRESHOLD",
    "PolicySweepAxis",
    "PolicyFamilySpec",
    "GeneratedPolicyRecord",
    "PolicyBenchmarkComparison",
    "PolicyFamilyBenchmarkSummary",
    "PolicyFamilyBenchmarkReceipt",
    "benchmark_policy_family",
]
