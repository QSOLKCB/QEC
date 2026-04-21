from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

DYNAMIC_CODE_POLICY_KERNEL_VERSION: str = "v138.9.3"

ALLOWED_CODE_FAMILIES: tuple[str, ...] = (
    "surface",
    "color",
    "ldpc",
    "concatenated",
    "topological",
    "subsystem",
    "bosonic",
)
ALLOWED_ACTIONS: tuple[str, ...] = ("stay", "switch", "migrate", "orchestrate", "defer", "reject")
ALLOWED_ESCALATION_LEVELS: tuple[str, ...] = ("none", "observe", "review", "high")
_ACTION_PRIORITY: tuple[str, ...] = ("orchestrate", "migrate", "switch", "stay", "defer", "reject")
_PRECISION: int = 12


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _clamp01(value: float) -> float:
    return _round64(min(1.0, max(0.0, float(value))))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_mapping(payload: Mapping[str, Any]) -> str:
    return _sha256_hex(_canonical_bytes(payload))


def _require_non_empty_token(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    token = value.strip()
    if not token:
        raise ValueError(f"{field} must be a non-empty string")
    return token


def _require_family(value: str, *, field: str) -> str:
    family = _require_non_empty_token(value, field=field)
    if family not in ALLOWED_CODE_FAMILIES:
        raise ValueError(f"{field} must be one of {ALLOWED_CODE_FAMILIES}")
    return family


def _require_bool(value: bool, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _require_unit_interval_float(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric and bool is not allowed")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    return _round64(numeric)


def _require_non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return int(value)


def _bounded_gain(candidate_value: float, current_value: float) -> float:
    return _clamp01((float(candidate_value) - float(current_value) + 1.0) / 2.0)


def _average(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    return _clamp01(sum(float(v) for v in values) / float(len(values)))


@dataclass(frozen=True)
class RuntimeCodeState:
    current_code_id: str
    current_code_family: str
    current_logical_stability: float
    current_projected_loss: float
    current_hardware_alignment: float
    current_execution_efficiency: float
    current_migration_overhead: float
    current_orchestration_depth: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_code_id", _require_non_empty_token(self.current_code_id, field="current_code_id"))
        object.__setattr__(self, "current_code_family", _require_family(self.current_code_family, field="current_code_family"))
        object.__setattr__(self, "current_logical_stability", _require_unit_interval_float(self.current_logical_stability, field="current_logical_stability"))
        object.__setattr__(self, "current_projected_loss", _require_unit_interval_float(self.current_projected_loss, field="current_projected_loss"))
        object.__setattr__(self, "current_hardware_alignment", _require_unit_interval_float(self.current_hardware_alignment, field="current_hardware_alignment"))
        object.__setattr__(self, "current_execution_efficiency", _require_unit_interval_float(self.current_execution_efficiency, field="current_execution_efficiency"))
        object.__setattr__(self, "current_migration_overhead", _require_unit_interval_float(self.current_migration_overhead, field="current_migration_overhead"))
        object.__setattr__(self, "current_orchestration_depth", _require_non_negative_int(self.current_orchestration_depth, field="current_orchestration_depth"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_code_id": self.current_code_id,
            "current_code_family": self.current_code_family,
            "current_logical_stability": self.current_logical_stability,
            "current_projected_loss": self.current_projected_loss,
            "current_hardware_alignment": self.current_hardware_alignment,
            "current_execution_efficiency": self.current_execution_efficiency,
            "current_migration_overhead": self.current_migration_overhead,
            "current_orchestration_depth": self.current_orchestration_depth,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class CandidatePolicyInput:
    candidate_code_id: str
    candidate_code_family: str
    selection_confidence: float
    candidate_logical_stability: float
    candidate_projected_loss: float
    candidate_hardware_alignment: float
    candidate_execution_efficiency: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_code_id", _require_non_empty_token(self.candidate_code_id, field="candidate_code_id"))
        object.__setattr__(self, "candidate_code_family", _require_family(self.candidate_code_family, field="candidate_code_family"))
        object.__setattr__(self, "selection_confidence", _require_unit_interval_float(self.selection_confidence, field="selection_confidence"))
        object.__setattr__(self, "candidate_logical_stability", _require_unit_interval_float(self.candidate_logical_stability, field="candidate_logical_stability"))
        object.__setattr__(self, "candidate_projected_loss", _require_unit_interval_float(self.candidate_projected_loss, field="candidate_projected_loss"))
        object.__setattr__(self, "candidate_hardware_alignment", _require_unit_interval_float(self.candidate_hardware_alignment, field="candidate_hardware_alignment"))
        object.__setattr__(self, "candidate_execution_efficiency", _require_unit_interval_float(self.candidate_execution_efficiency, field="candidate_execution_efficiency"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_code_id": self.candidate_code_id,
            "candidate_code_family": self.candidate_code_family,
            "selection_confidence": self.selection_confidence,
            "candidate_logical_stability": self.candidate_logical_stability,
            "candidate_projected_loss": self.candidate_projected_loss,
            "candidate_hardware_alignment": self.candidate_hardware_alignment,
            "candidate_execution_efficiency": self.candidate_execution_efficiency,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class MigrationPolicyInput:
    migration_target_family: str
    migration_compatibility: float
    migration_projected_loss: float
    migration_distance_retention: float
    migration_observable_overlap: float
    migration_hardware_fit: float
    migration_confidence: float
    migration_admissible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "migration_target_family", _require_family(self.migration_target_family, field="migration_target_family"))
        object.__setattr__(self, "migration_compatibility", _require_unit_interval_float(self.migration_compatibility, field="migration_compatibility"))
        object.__setattr__(self, "migration_projected_loss", _require_unit_interval_float(self.migration_projected_loss, field="migration_projected_loss"))
        object.__setattr__(self, "migration_distance_retention", _require_unit_interval_float(self.migration_distance_retention, field="migration_distance_retention"))
        object.__setattr__(self, "migration_observable_overlap", _require_unit_interval_float(self.migration_observable_overlap, field="migration_observable_overlap"))
        object.__setattr__(self, "migration_hardware_fit", _require_unit_interval_float(self.migration_hardware_fit, field="migration_hardware_fit"))
        object.__setattr__(self, "migration_confidence", _require_unit_interval_float(self.migration_confidence, field="migration_confidence"))
        object.__setattr__(self, "migration_admissible", _require_bool(self.migration_admissible, field="migration_admissible"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_target_family": self.migration_target_family,
            "migration_compatibility": self.migration_compatibility,
            "migration_projected_loss": self.migration_projected_loss,
            "migration_distance_retention": self.migration_distance_retention,
            "migration_observable_overlap": self.migration_observable_overlap,
            "migration_hardware_fit": self.migration_hardware_fit,
            "migration_confidence": self.migration_confidence,
            "migration_admissible": self.migration_admissible,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class OrchestrationPolicyInput:
    benchmark_best_candidate_id: str
    benchmark_best_family: str
    benchmark_best_utility: float
    benchmark_baseline_utility: float
    benchmark_improvement_margin: float
    cross_family_winner: bool
    benchmark_admissible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_best_candidate_id", _require_non_empty_token(self.benchmark_best_candidate_id, field="benchmark_best_candidate_id"))
        object.__setattr__(self, "benchmark_best_family", _require_family(self.benchmark_best_family, field="benchmark_best_family"))
        object.__setattr__(self, "benchmark_best_utility", _require_unit_interval_float(self.benchmark_best_utility, field="benchmark_best_utility"))
        object.__setattr__(self, "benchmark_baseline_utility", _require_unit_interval_float(self.benchmark_baseline_utility, field="benchmark_baseline_utility"))
        object.__setattr__(self, "benchmark_improvement_margin", _require_unit_interval_float(self.benchmark_improvement_margin, field="benchmark_improvement_margin"))
        object.__setattr__(self, "cross_family_winner", _require_bool(self.cross_family_winner, field="cross_family_winner"))
        object.__setattr__(self, "benchmark_admissible", _require_bool(self.benchmark_admissible, field="benchmark_admissible"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_best_candidate_id": self.benchmark_best_candidate_id,
            "benchmark_best_family": self.benchmark_best_family,
            "benchmark_best_utility": self.benchmark_best_utility,
            "benchmark_baseline_utility": self.benchmark_baseline_utility,
            "benchmark_improvement_margin": self.benchmark_improvement_margin,
            "cross_family_winner": self.cross_family_winner,
            "benchmark_admissible": self.benchmark_admissible,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class DynamicCodePolicy:
    minimum_selection_confidence: float
    minimum_migration_confidence: float
    minimum_benchmark_utility: float
    minimum_improvement_margin: float
    maximum_projected_loss: float
    maximum_migration_overhead: float
    require_cross_family_benefit: bool
    require_migration_admissibility: bool
    prefer_stability_gain: bool
    prefer_hardware_alignment: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_selection_confidence", _require_unit_interval_float(self.minimum_selection_confidence, field="minimum_selection_confidence"))
        object.__setattr__(self, "minimum_migration_confidence", _require_unit_interval_float(self.minimum_migration_confidence, field="minimum_migration_confidence"))
        object.__setattr__(self, "minimum_benchmark_utility", _require_unit_interval_float(self.minimum_benchmark_utility, field="minimum_benchmark_utility"))
        object.__setattr__(self, "minimum_improvement_margin", _require_unit_interval_float(self.minimum_improvement_margin, field="minimum_improvement_margin"))
        object.__setattr__(self, "maximum_projected_loss", _require_unit_interval_float(self.maximum_projected_loss, field="maximum_projected_loss"))
        object.__setattr__(self, "maximum_migration_overhead", _require_unit_interval_float(self.maximum_migration_overhead, field="maximum_migration_overhead"))
        object.__setattr__(self, "require_cross_family_benefit", _require_bool(self.require_cross_family_benefit, field="require_cross_family_benefit"))
        object.__setattr__(self, "require_migration_admissibility", _require_bool(self.require_migration_admissibility, field="require_migration_admissibility"))
        object.__setattr__(self, "prefer_stability_gain", _require_bool(self.prefer_stability_gain, field="prefer_stability_gain"))
        object.__setattr__(self, "prefer_hardware_alignment", _require_bool(self.prefer_hardware_alignment, field="prefer_hardware_alignment"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_selection_confidence": self.minimum_selection_confidence,
            "minimum_migration_confidence": self.minimum_migration_confidence,
            "minimum_benchmark_utility": self.minimum_benchmark_utility,
            "minimum_improvement_margin": self.minimum_improvement_margin,
            "maximum_projected_loss": self.maximum_projected_loss,
            "maximum_migration_overhead": self.maximum_migration_overhead,
            "require_cross_family_benefit": self.require_cross_family_benefit,
            "require_migration_admissibility": self.require_migration_admissibility,
            "prefer_stability_gain": self.prefer_stability_gain,
            "prefer_hardware_alignment": self.prefer_hardware_alignment,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class PolicyDecision:
    selected_action: str
    target_code_id: str
    target_code_family: str
    stay_on_current_code: bool
    approve_migration: bool
    recommend_orchestration: bool
    policy_confidence: float
    improvement_score: float
    risk_score: float
    rationale: tuple[str, ...]
    escalation_level: str

    def __post_init__(self) -> None:
        action = _require_non_empty_token(self.selected_action, field="selected_action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"selected_action must be one of {ALLOWED_ACTIONS}")
        object.__setattr__(self, "selected_action", action)
        object.__setattr__(self, "target_code_id", _require_non_empty_token(self.target_code_id, field="target_code_id"))
        object.__setattr__(self, "target_code_family", _require_family(self.target_code_family, field="target_code_family"))
        object.__setattr__(self, "stay_on_current_code", _require_bool(self.stay_on_current_code, field="stay_on_current_code"))
        object.__setattr__(self, "approve_migration", _require_bool(self.approve_migration, field="approve_migration"))
        object.__setattr__(self, "recommend_orchestration", _require_bool(self.recommend_orchestration, field="recommend_orchestration"))
        object.__setattr__(self, "policy_confidence", _require_unit_interval_float(self.policy_confidence, field="policy_confidence"))
        object.__setattr__(self, "improvement_score", _require_unit_interval_float(self.improvement_score, field="improvement_score"))
        object.__setattr__(self, "risk_score", _require_unit_interval_float(self.risk_score, field="risk_score"))
        if not isinstance(self.rationale, tuple) or any(not isinstance(item, str) or not item for item in self.rationale):
            raise ValueError("rationale must be a tuple of non-empty strings")
        escalation = _require_non_empty_token(self.escalation_level, field="escalation_level")
        if escalation not in ALLOWED_ESCALATION_LEVELS:
            raise ValueError(f"escalation_level must be one of {ALLOWED_ESCALATION_LEVELS}")
        object.__setattr__(self, "escalation_level", escalation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "target_code_id": self.target_code_id,
            "target_code_family": self.target_code_family,
            "stay_on_current_code": self.stay_on_current_code,
            "approve_migration": self.approve_migration,
            "recommend_orchestration": self.recommend_orchestration,
            "policy_confidence": self.policy_confidence,
            "improvement_score": self.improvement_score,
            "risk_score": self.risk_score,
            "rationale": list(self.rationale),
            "escalation_level": self.escalation_level,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _hash_mapping(self.to_dict())


@dataclass(frozen=True)
class PolicyDecisionReceipt:
    runtime_state: RuntimeCodeState
    candidate_input: CandidatePolicyInput
    migration_input: MigrationPolicyInput
    orchestration_input: OrchestrationPolicyInput
    policy_snapshot: DynamicCodePolicy
    decision: PolicyDecision
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_non_empty_token(self.schema_version, field="schema_version"))
        if not isinstance(self.replay_identity, str) or len(self.replay_identity) != 64:
            raise ValueError("replay_identity must be a 64-character SHA-256 hex string")
        if not isinstance(self.stable_hash, str) or len(self.stable_hash) != 64:
            raise ValueError("stable_hash must be a 64-character SHA-256 hex string")
        expected_replay_identity = _compute_replay_identity(
            self.runtime_state,
            self.candidate_input,
            self.migration_input,
            self.orchestration_input,
            self.policy_snapshot,
            self.schema_version,
        )
        if self.replay_identity != expected_replay_identity:
            raise ValueError("replay_identity does not match canonical replay payload")
        expected_stable_hash = _compute_receipt_hash_payload(
            self.runtime_state,
            self.candidate_input,
            self.migration_input,
            self.orchestration_input,
            self.policy_snapshot,
            self.decision,
            self.schema_version,
            self.replay_identity,
        )
        if self.stable_hash != expected_stable_hash:
            raise ValueError("stable_hash does not match canonical receipt payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_state": self.runtime_state.to_dict(),
            "candidate_input": self.candidate_input.to_dict(),
            "migration_input": self.migration_input.to_dict(),
            "orchestration_input": self.orchestration_input.to_dict(),
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "decision": self.decision.to_dict(),
            "schema_version": self.schema_version,
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _build_replay_identity_payload(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy_snapshot: DynamicCodePolicy,
    schema_version: str,
) -> dict[str, Any]:
    return {
        "runtime_state": runtime_state.to_dict(),
        "candidate_input": candidate_input.to_dict(),
        "migration_input": migration_input.to_dict(),
        "orchestration_input": orchestration_input.to_dict(),
        "policy_snapshot": policy_snapshot.to_dict(),
        "schema_version": schema_version,
    }


def _compute_replay_identity(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy_snapshot: DynamicCodePolicy,
    schema_version: str,
) -> str:
    payload = _build_replay_identity_payload(
        runtime_state, candidate_input, migration_input, orchestration_input, policy_snapshot, schema_version
    )
    return _hash_mapping(payload)


def _build_receipt_hash_payload(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy_snapshot: DynamicCodePolicy,
    decision: PolicyDecision,
    schema_version: str,
    replay_identity: str,
) -> dict[str, Any]:
    return {
        "runtime_state": runtime_state.to_dict(),
        "candidate_input": candidate_input.to_dict(),
        "migration_input": migration_input.to_dict(),
        "orchestration_input": orchestration_input.to_dict(),
        "policy_snapshot": policy_snapshot.to_dict(),
        "decision": decision.to_dict(),
        "schema_version": schema_version,
        "replay_identity": replay_identity,
    }


def _compute_receipt_hash_payload(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy_snapshot: DynamicCodePolicy,
    decision: PolicyDecision,
    schema_version: str,
    replay_identity: str,
) -> str:
    return _hash_mapping(
        _build_receipt_hash_payload(
            runtime_state,
            candidate_input,
            migration_input,
            orchestration_input,
            policy_snapshot,
            decision,
            schema_version,
            replay_identity,
        )
    )


def _compute_signals(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy: DynamicCodePolicy,
) -> tuple[float, float, float]:
    stability_gain = _bounded_gain(candidate_input.candidate_logical_stability, runtime_state.current_logical_stability)
    loss_advantage = _bounded_gain(runtime_state.current_projected_loss, candidate_input.candidate_projected_loss)
    efficiency_gain = _bounded_gain(candidate_input.candidate_execution_efficiency, runtime_state.current_execution_efficiency)
    hardware_support = candidate_input.candidate_hardware_alignment if policy.prefer_hardware_alignment else 0.5
    stability_support = stability_gain if policy.prefer_stability_gain else 0.5
    utility_support = orchestration_input.benchmark_best_utility

    improvement_score = _average((stability_support, loss_advantage, efficiency_gain, hardware_support, utility_support))

    risk_score = _average(
        (
            candidate_input.candidate_projected_loss,
            migration_input.migration_projected_loss,
            runtime_state.current_migration_overhead,
            1.0 - migration_input.migration_compatibility,
            1.0 - migration_input.migration_distance_retention,
        )
    )

    policy_confidence = _average(
        (
            candidate_input.selection_confidence,
            migration_input.migration_confidence,
            orchestration_input.benchmark_best_utility,
            orchestration_input.benchmark_improvement_margin,
            improvement_score,
            1.0 - risk_score,
        )
    )
    return improvement_score, risk_score, policy_confidence


def _select_escalation(action: str, risk_score: float) -> str:
    if action == "reject":
        return "high" if risk_score >= 0.7 else "review"
    if action == "defer":
        return "observe"
    if action == "orchestrate":
        return "review" if risk_score >= 0.45 else "none"
    if action == "migrate":
        return "review" if risk_score >= 0.4 else "none"
    if action == "switch":
        return "observe" if risk_score >= 0.35 else "none"
    return "none"


def decide_dynamic_code_policy(
    runtime_state: RuntimeCodeState,
    candidate_input: CandidatePolicyInput,
    migration_input: MigrationPolicyInput,
    orchestration_input: OrchestrationPolicyInput,
    policy: DynamicCodePolicy,
) -> PolicyDecisionReceipt:
    improvement_score, risk_score, policy_confidence = _compute_signals(
        runtime_state, candidate_input, migration_input, orchestration_input, policy
    )

    candidate_changes_runtime = (
        candidate_input.candidate_code_id != runtime_state.current_code_id
        or candidate_input.candidate_code_family != runtime_state.current_code_family
    )
    cross_family_action = candidate_input.candidate_code_family != runtime_state.current_code_family

    selection_ok = candidate_input.selection_confidence >= policy.minimum_selection_confidence
    improvement_margin_ok = orchestration_input.benchmark_improvement_margin >= policy.minimum_improvement_margin
    benchmark_utility_ok = orchestration_input.benchmark_best_utility >= policy.minimum_benchmark_utility
    candidate_loss_ok = candidate_input.candidate_projected_loss <= policy.maximum_projected_loss
    overhead_ok = runtime_state.current_migration_overhead <= policy.maximum_migration_overhead
    migration_conf_ok = migration_input.migration_confidence >= policy.minimum_migration_confidence
    migration_family_ok = migration_input.migration_target_family == candidate_input.candidate_code_family
    migration_admissibility_ok = (not policy.require_migration_admissibility) or migration_input.migration_admissible
    cross_family_ok = (not policy.require_cross_family_benefit) or (not cross_family_action) or orchestration_input.cross_family_winner
    benchmark_candidate_ok = (
        orchestration_input.benchmark_best_candidate_id == candidate_input.candidate_code_id
        and orchestration_input.benchmark_best_family == candidate_input.candidate_code_family
    )

    hard_reject = (not candidate_loss_ok) or (not overhead_ok) or risk_score >= 0.8

    orchestrate_eligible = (
        candidate_changes_runtime
        and selection_ok
        and benchmark_utility_ok
        and improvement_margin_ok
        and benchmark_candidate_ok
        and orchestration_input.benchmark_admissible
        and cross_family_ok
        and not hard_reject
        and improvement_score >= 0.5
    )

    migrate_eligible = (
        candidate_changes_runtime
        and selection_ok
        and migration_conf_ok
        and migration_family_ok
        and migration_admissibility_ok
        and cross_family_ok
        and candidate_loss_ok
        and overhead_ok
        and not hard_reject
        and improvement_score >= 0.5
    )

    switch_eligible = (
        candidate_changes_runtime
        and selection_ok
        and improvement_margin_ok
        and cross_family_ok
        and candidate_loss_ok
        and overhead_ok
        and not hard_reject
        and improvement_score >= 0.5
    )

    stay_eligible = (not candidate_changes_runtime or improvement_score < 0.5 or not selection_ok) and not hard_reject
    defer_eligible = candidate_changes_runtime and not hard_reject and not (orchestrate_eligible or migrate_eligible or switch_eligible or stay_eligible)
    reject_eligible = hard_reject or not any((orchestrate_eligible, migrate_eligible, switch_eligible, stay_eligible, defer_eligible))

    action_eligibility = {
        "orchestrate": orchestrate_eligible,
        "migrate": migrate_eligible,
        "switch": switch_eligible,
        "stay": stay_eligible,
        "defer": defer_eligible,
        "reject": reject_eligible,
    }

    selected_action = "reject"
    for candidate_action in _ACTION_PRIORITY:
        if action_eligibility[candidate_action]:
            selected_action = candidate_action
            break

    rationale: list[str] = []
    rationale.append("selection confidence threshold satisfied" if selection_ok else "selection confidence below policy minimum")
    rationale.append("improvement margin threshold satisfied" if improvement_margin_ok else "benchmark improvement margin insufficient")
    rationale.append("benchmark utility threshold satisfied" if benchmark_utility_ok else "benchmark utility below policy minimum")
    rationale.append("candidate projected loss within policy maximum" if candidate_loss_ok else "projected loss exceeds policy maximum")
    rationale.append("migration overhead within policy maximum" if overhead_ok else "migration overhead exceeds policy maximum")
    rationale.append("migration confidence threshold satisfied" if migration_conf_ok else "migration confidence below policy minimum")
    rationale.append("migration target family matches candidate" if migration_family_ok else "migration target family mismatch")
    rationale.append("migration admissibility satisfied" if migration_admissibility_ok else "migration admissibility required but not satisfied")
    rationale.append("cross-family benefit satisfied" if cross_family_ok else "cross-family benefit required but not demonstrated")
    rationale.append("benchmark candidate alignment satisfied" if benchmark_candidate_ok else "benchmark winner does not match candidate")
    rationale.append("risk within bounded policy tolerance" if not hard_reject else "risk exceeds bounded policy tolerance")
    rationale.append(f"selected action: {selected_action}")

    target_code_id = runtime_state.current_code_id if selected_action == "stay" else candidate_input.candidate_code_id
    target_code_family = runtime_state.current_code_family if selected_action == "stay" else candidate_input.candidate_code_family

    decision = PolicyDecision(
        selected_action=selected_action,
        target_code_id=target_code_id,
        target_code_family=target_code_family,
        stay_on_current_code=selected_action == "stay",
        approve_migration=selected_action == "migrate",
        recommend_orchestration=selected_action == "orchestrate",
        policy_confidence=policy_confidence,
        improvement_score=improvement_score,
        risk_score=risk_score,
        rationale=tuple(rationale),
        escalation_level=_select_escalation(selected_action, risk_score),
    )

    replay_identity = _compute_replay_identity(
        runtime_state,
        candidate_input,
        migration_input,
        orchestration_input,
        policy,
        DYNAMIC_CODE_POLICY_KERNEL_VERSION,
    )

    stable_hash = _compute_receipt_hash_payload(
        runtime_state,
        candidate_input,
        migration_input,
        orchestration_input,
        policy,
        decision,
        DYNAMIC_CODE_POLICY_KERNEL_VERSION,
        replay_identity,
    )

    return PolicyDecisionReceipt(
        runtime_state=runtime_state,
        candidate_input=candidate_input,
        migration_input=migration_input,
        orchestration_input=orchestration_input,
        policy_snapshot=policy,
        decision=decision,
        schema_version=DYNAMIC_CODE_POLICY_KERNEL_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )
