"""v145.2 — Policy Sensitivity Analysis Kernel (PSAK)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

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

MIN_POLICY_COUNT = 2
SENSITIVITY_MODERATE_THRESHOLD = 0.25
SENSITIVITY_HIGH_THRESHOLD = 0.50

_ALLOWED_COMPARISON_CLASSES = frozenset({"equivalent", "moderate_shift", "high_shift"})
_ALLOWED_GLOBAL_CLASSES = frozenset({"low", "moderate", "high"})


def _effect_score(*, left: "PolicyRunRecord", right: "PolicyRunRecord") -> float:
    cycle_count = max(left.allow_count + left.hold_count + left.reject_count, 1)
    normalized_allow = abs(right.allow_count - left.allow_count) / float(cycle_count)
    normalized_reject = abs(right.reject_count - left.reject_count) / float(cycle_count)
    normalized_admissible = abs(right.admissible_count - left.admissible_count) / float(cycle_count)
    convergence = abs(round12(right.mean_convergence_metric - left.mean_convergence_metric))
    return round12(max(normalized_allow, normalized_reject, normalized_admissible, convergence))


def _classify_effect(effect_score: float) -> str:
    if effect_score >= SENSITIVITY_HIGH_THRESHOLD:
        return "high_shift"
    if effect_score >= SENSITIVITY_MODERATE_THRESHOLD:
        return "moderate_shift"
    return "equivalent"


@dataclass(frozen=True)
class PolicyRunRecord:
    policy_hash: str
    governed_receipt_hash: str
    allow_count: int
    hold_count: int
    reject_count: int
    admissible_count: int
    non_admissible_count: int
    mean_convergence_metric: float
    stable_transition_count: int
    uncertain_transition_count: int
    stable_hash: str

    def __post_init__(self) -> None:
        validate_sha256_hex(self.policy_hash, "policy_hash")
        validate_sha256_hex(self.governed_receipt_hash, "governed_receipt_hash")
        for field_name in (
            "allow_count",
            "hold_count",
            "reject_count",
            "admissible_count",
            "non_admissible_count",
            "stable_transition_count",
            "uncertain_transition_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be non-negative int")
        cycle_count = self.allow_count + self.hold_count + self.reject_count
        if self.admissible_count + self.non_admissible_count != cycle_count:
            raise ValueError("admissible/non_admissible count inconsistency")
        if self.stable_transition_count + self.uncertain_transition_count != cycle_count:
            raise ValueError("transition count inconsistency")
        object.__setattr__(self, "mean_convergence_metric", validate_unit_interval(self.mean_convergence_metric, "mean_convergence_metric"))
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy_hash": self.policy_hash,
            "governed_receipt_hash": self.governed_receipt_hash,
            "allow_count": self.allow_count,
            "hold_count": self.hold_count,
            "reject_count": self.reject_count,
            "admissible_count": self.admissible_count,
            "non_admissible_count": self.non_admissible_count,
            "mean_convergence_metric": round12(self.mean_convergence_metric),
            "stable_transition_count": self.stable_transition_count,
            "uncertain_transition_count": self.uncertain_transition_count,
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
class PolicyComparisonRecord:
    left_policy_hash: str
    right_policy_hash: str
    allow_delta: int
    hold_delta: int
    reject_delta: int
    admissible_delta: int
    convergence_delta: float
    sensitivity_classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        validate_sha256_hex(self.left_policy_hash, "left_policy_hash")
        validate_sha256_hex(self.right_policy_hash, "right_policy_hash")
        if self.left_policy_hash >= self.right_policy_hash:
            raise ValueError("comparison policy hashes must be canonical ascending order")
        for field_name in ("allow_delta", "hold_delta", "reject_delta", "admissible_delta"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be int")
        if self.sensitivity_classification not in _ALLOWED_COMPARISON_CLASSES:
            raise ValueError("sensitivity_classification is invalid")
        if isinstance(self.convergence_delta, bool) or not isinstance(self.convergence_delta, (int, float)):
            raise ValueError("convergence_delta must be numeric")
        convergence_delta = float(self.convergence_delta)
        if not (-1.0 <= convergence_delta <= 1.0):
            raise ValueError("convergence_delta must be in [-1,1]")
        object.__setattr__(self, "convergence_delta", round12(convergence_delta))
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "left_policy_hash": self.left_policy_hash,
            "right_policy_hash": self.right_policy_hash,
            "allow_delta": self.allow_delta,
            "hold_delta": self.hold_delta,
            "reject_delta": self.reject_delta,
            "admissible_delta": self.admissible_delta,
            "convergence_delta": round12(self.convergence_delta),
            "sensitivity_classification": self.sensitivity_classification,
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
class PolicySensitivitySummary:
    policy_count: int
    comparison_count: int
    most_permissive_policy_hash: str
    most_restrictive_policy_hash: str
    highest_convergence_policy_hash: str
    lowest_convergence_policy_hash: str
    global_sensitivity_classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy_count, int) or isinstance(self.policy_count, bool) or self.policy_count < MIN_POLICY_COUNT:
            raise ValueError("policy_count must be int >= MIN_POLICY_COUNT")
        if not isinstance(self.comparison_count, int) or isinstance(self.comparison_count, bool) or self.comparison_count < 1:
            raise ValueError("comparison_count must be int >= 1")
        expected_comparisons = self.policy_count * (self.policy_count - 1) // 2
        if self.comparison_count != expected_comparisons:
            raise ValueError("comparison_count mismatch")
        validate_sha256_hex(self.most_permissive_policy_hash, "most_permissive_policy_hash")
        validate_sha256_hex(self.most_restrictive_policy_hash, "most_restrictive_policy_hash")
        validate_sha256_hex(self.highest_convergence_policy_hash, "highest_convergence_policy_hash")
        validate_sha256_hex(self.lowest_convergence_policy_hash, "lowest_convergence_policy_hash")
        if self.global_sensitivity_classification not in _ALLOWED_GLOBAL_CLASSES:
            raise ValueError("global_sensitivity_classification is invalid")
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy_count": self.policy_count,
            "comparison_count": self.comparison_count,
            "most_permissive_policy_hash": self.most_permissive_policy_hash,
            "most_restrictive_policy_hash": self.most_restrictive_policy_hash,
            "highest_convergence_policy_hash": self.highest_convergence_policy_hash,
            "lowest_convergence_policy_hash": self.lowest_convergence_policy_hash,
            "global_sensitivity_classification": self.global_sensitivity_classification,
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
class PolicySensitivityReceipt:
    config: SimulationConfig
    policy_run_records: tuple[PolicyRunRecord, ...]
    comparison_records: tuple[PolicyComparisonRecord, ...]
    summary: PolicySensitivitySummary
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("config must be SimulationConfig")
        ensure_stable_hash(self.config, "config")
        if not isinstance(self.policy_run_records, tuple) or any(not isinstance(item, PolicyRunRecord) for item in self.policy_run_records):
            raise ValueError("policy_run_records must be tuple[PolicyRunRecord, ...]")
        if len(self.policy_run_records) < MIN_POLICY_COUNT:
            raise ValueError("policy_run_records must contain at least MIN_POLICY_COUNT entries")
        sorted_runs = tuple(sorted(self.policy_run_records, key=lambda item: item.policy_hash))
        if self.policy_run_records != sorted_runs:
            raise ValueError("policy_run_records must be sorted by policy_hash")
        policy_hashes = tuple(record.policy_hash for record in self.policy_run_records)
        if len(set(policy_hashes)) != len(policy_hashes):
            raise ValueError("policy_run_records contains duplicate policy_hash")
        if not isinstance(self.comparison_records, tuple) or any(not isinstance(item, PolicyComparisonRecord) for item in self.comparison_records):
            raise ValueError("comparison_records must be tuple[PolicyComparisonRecord, ...]")
        expected_pairs = len(self.policy_run_records) * (len(self.policy_run_records) - 1) // 2
        if len(self.comparison_records) != expected_pairs:
            raise ValueError("comparison_records length mismatch")
        sorted_pairs = tuple(
            sorted(self.comparison_records, key=lambda item: (item.left_policy_hash, item.right_policy_hash))
        )
        if self.comparison_records != sorted_pairs:
            raise ValueError("comparison_records must be sorted by policy-hash pair")
        if not isinstance(self.summary, PolicySensitivitySummary):
            raise ValueError("summary must be PolicySensitivitySummary")
        ensure_stable_hash(self.summary, "summary")
        if self.summary.policy_count != len(self.policy_run_records):
            raise ValueError("summary policy_count mismatch")
        if self.summary.comparison_count != len(self.comparison_records):
            raise ValueError("summary comparison_count mismatch")
        for field_name, policy_hash in (
            ("most_permissive_policy_hash", self.summary.most_permissive_policy_hash),
            ("most_restrictive_policy_hash", self.summary.most_restrictive_policy_hash),
            ("highest_convergence_policy_hash", self.summary.highest_convergence_policy_hash),
            ("lowest_convergence_policy_hash", self.summary.lowest_convergence_policy_hash),
        ):
            if policy_hash not in policy_hashes:
                raise ValueError(f"summary {field_name} must reference a policy_run_records policy_hash")
        validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "policy_run_records": tuple(item.to_dict() for item in self.policy_run_records),
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


def _make_policy_run_record(policy_hash: str, governed_receipt: GovernedClosedLoopReceipt) -> PolicyRunRecord:
    summary = governed_receipt.summary
    payload = {
        "policy_hash": policy_hash,
        "governed_receipt_hash": governed_receipt.stable_hash,
        "allow_count": summary.allow_count,
        "hold_count": summary.hold_count,
        "reject_count": summary.reject_count,
        "admissible_count": summary.admissible_count,
        "non_admissible_count": summary.non_admissible_count,
        "mean_convergence_metric": round12(summary.mean_convergence_metric),
        "stable_transition_count": summary.stable_transition_count,
        "uncertain_transition_count": summary.uncertain_transition_count,
    }
    return PolicyRunRecord(
        policy_hash=policy_hash,
        governed_receipt_hash=governed_receipt.stable_hash,
        allow_count=summary.allow_count,
        hold_count=summary.hold_count,
        reject_count=summary.reject_count,
        admissible_count=summary.admissible_count,
        non_admissible_count=summary.non_admissible_count,
        mean_convergence_metric=summary.mean_convergence_metric,
        stable_transition_count=summary.stable_transition_count,
        uncertain_transition_count=summary.uncertain_transition_count,
        stable_hash=sha256_hex(payload),
    )


def _build_comparisons(policy_runs: tuple[PolicyRunRecord, ...]) -> tuple[PolicyComparisonRecord, ...]:
    records: list[PolicyComparisonRecord] = []
    for left, right in combinations(policy_runs, 2):
        convergence_delta = round12(right.mean_convergence_metric - left.mean_convergence_metric)
        effect_score = _effect_score(left=left, right=right)
        sensitivity_classification = _classify_effect(effect_score)
        payload = {
            "left_policy_hash": left.policy_hash,
            "right_policy_hash": right.policy_hash,
            "allow_delta": right.allow_count - left.allow_count,
            "hold_delta": right.hold_count - left.hold_count,
            "reject_delta": right.reject_count - left.reject_count,
            "admissible_delta": right.admissible_count - left.admissible_count,
            "convergence_delta": convergence_delta,
            "sensitivity_classification": sensitivity_classification,
        }
        records.append(
            PolicyComparisonRecord(
                left_policy_hash=left.policy_hash,
                right_policy_hash=right.policy_hash,
                allow_delta=right.allow_count - left.allow_count,
                hold_delta=right.hold_count - left.hold_count,
                reject_delta=right.reject_count - left.reject_count,
                admissible_delta=right.admissible_count - left.admissible_count,
                convergence_delta=convergence_delta,
                sensitivity_classification=sensitivity_classification,
                stable_hash=sha256_hex(payload),
            )
        )
    return tuple(sorted(records, key=lambda item: (item.left_policy_hash, item.right_policy_hash)))


def _choose_by_extrema(
    policy_runs: tuple[PolicyRunRecord, ...],
    *,
    metric_name: str,
    invert: bool,
) -> str:
    if invert:
        selected = min(policy_runs, key=lambda rec: (-getattr(rec, metric_name), rec.policy_hash))
        return selected.policy_hash
    selected = min(policy_runs, key=lambda rec: (getattr(rec, metric_name), rec.policy_hash))
    return selected.policy_hash


def _build_summary(
    policy_runs: tuple[PolicyRunRecord, ...],
    comparison_records: tuple[PolicyComparisonRecord, ...],
) -> PolicySensitivitySummary:
    if any(record.sensitivity_classification == "high_shift" for record in comparison_records):
        global_class = "high"
    elif any(record.sensitivity_classification == "moderate_shift" for record in comparison_records):
        global_class = "moderate"
    else:
        global_class = "low"

    most_permissive = _choose_by_extrema(policy_runs, metric_name="admissible_count", invert=True)
    most_restrictive = _choose_by_extrema(policy_runs, metric_name="admissible_count", invert=False)
    highest_convergence = _choose_by_extrema(policy_runs, metric_name="mean_convergence_metric", invert=True)
    lowest_convergence = _choose_by_extrema(policy_runs, metric_name="mean_convergence_metric", invert=False)

    payload = {
        "policy_count": len(policy_runs),
        "comparison_count": len(comparison_records),
        "most_permissive_policy_hash": most_permissive,
        "most_restrictive_policy_hash": most_restrictive,
        "highest_convergence_policy_hash": highest_convergence,
        "lowest_convergence_policy_hash": lowest_convergence,
        "global_sensitivity_classification": global_class,
    }
    return PolicySensitivitySummary(
        policy_count=len(policy_runs),
        comparison_count=len(comparison_records),
        most_permissive_policy_hash=most_permissive,
        most_restrictive_policy_hash=most_restrictive,
        highest_convergence_policy_hash=highest_convergence,
        lowest_convergence_policy_hash=lowest_convergence,
        global_sensitivity_classification=global_class,
        stable_hash=sha256_hex(payload),
    )


def analyze_policy_sensitivity(
    config: SimulationConfig,
    policies: tuple[GovernancePolicy, ...],
) -> PolicySensitivityReceipt:
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be SimulationConfig")
    ensure_stable_hash(config, "config")
    if not isinstance(policies, tuple):
        raise ValueError("policies must be tuple[GovernancePolicy, ...]")
    if len(policies) < MIN_POLICY_COUNT:
        raise ValueError("policies must contain at least MIN_POLICY_COUNT entries")
    if any(not isinstance(policy, GovernancePolicy) for policy in policies):
        raise ValueError("policies must be tuple[GovernancePolicy, ...]")

    policy_hashes: list[str] = []
    for index, policy in enumerate(policies):
        ensure_stable_hash(policy, f"policies[{index}]")
        policy_hashes.append(policy.stable_hash)
    if len(set(policy_hashes)) != len(policy_hashes):
        raise ValueError("duplicate policy stable_hash values are not allowed")

    run_records: list[PolicyRunRecord] = []
    for policy in policies:
        governed_receipt = run_governed_closed_loop(config, policy)
        if not isinstance(governed_receipt, GovernedClosedLoopReceipt):
            raise ValueError("run_governed_closed_loop must return GovernedClosedLoopReceipt")
        ensure_stable_hash(governed_receipt, "governed_receipt")
        if governed_receipt.config.stable_hash != config.stable_hash:
            raise ValueError("governed receipt config mismatch")
        if governed_receipt.policy.stable_hash != policy.stable_hash:
            raise ValueError("governed receipt policy mismatch")
        run_records.append(_make_policy_run_record(policy.stable_hash, governed_receipt))

    sorted_runs = tuple(sorted(run_records, key=lambda item: item.policy_hash))
    comparison_records = _build_comparisons(sorted_runs)
    summary = _build_summary(sorted_runs, comparison_records)
    payload = {
        "config": config.to_dict(),
        "policy_run_records": tuple(item.to_dict() for item in sorted_runs),
        "comparison_records": tuple(item.to_dict() for item in comparison_records),
        "summary": summary.to_dict(),
    }
    return PolicySensitivityReceipt(
        config=config,
        policy_run_records=sorted_runs,
        comparison_records=comparison_records,
        summary=summary,
        stable_hash=sha256_hex(payload),
    )


__all__ = [
    "MIN_POLICY_COUNT",
    "SENSITIVITY_MODERATE_THRESHOLD",
    "SENSITIVITY_HIGH_THRESHOLD",
    "PolicyRunRecord",
    "PolicyComparisonRecord",
    "PolicySensitivitySummary",
    "PolicySensitivityReceipt",
    "analyze_policy_sensitivity",
]
