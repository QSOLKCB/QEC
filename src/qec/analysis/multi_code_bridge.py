# SPDX-License-Identifier: MIT
"""v138.9.4 — deterministic additive analysis-layer multi-code bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .canonical_hashing import canonical_json, sha256_hex

ALLOWED_CODE_FAMILIES: tuple[str, ...] = ("surface", "qldpc", "ternary", "color")
ALLOWED_SELECTED_ACTIONS: tuple[str, ...] = (
    "stay",
    "switch",
    "migrate",
    "orchestrate",
    "defer",
    "reject",
)
ALLOWED_ESCALATION_LEVELS: tuple[str, ...] = ("none", "observe", "review", "high")
BRIDGE_STEP_NAMES: tuple[str, ...] = (
    "validate_selection_artifact",
    "validate_migration_artifact",
    "validate_benchmark_artifact",
    "validate_policy_artifact",
    "reconcile_target_alignment",
    "assess_bridge_readiness",
    "emit_execution_handoff",
)

SCHEMA_VERSION = "v138.9.4"
_SHA256_HEX_LEN = 64


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _validate_unit_interval_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric in [0,1]")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be a finite numeric in [0,1]")
    return numeric


def _validate_sha256_hex(name: str, value: Any) -> str:
    text = _validate_non_empty_str(name, value)
    if len(text) != _SHA256_HEX_LEN or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex string")
    return text


def _validate_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


@dataclass(frozen=True)
class BridgeSelectionInput:
    selected_code_id: str
    selected_code_family: str
    selection_confidence: float
    ranking_order: tuple[str, ...]
    selection_stable_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str("selected_code_id", self.selected_code_id)
        _validate_non_empty_str("selected_code_family", self.selected_code_family)
        _validate_unit_interval_float("selection_confidence", self.selection_confidence)
        if not isinstance(self.ranking_order, tuple):
            raise ValueError("ranking_order must be a non-empty tuple[str, ...]")
        if not self.ranking_order:
            raise ValueError("ranking_order must be non-empty")
        seen: set[str] = set()
        for entry in self.ranking_order:
            _validate_non_empty_str("ranking_order entry", entry)
            if entry in seen:
                raise ValueError("ranking_order entries must be unique")
            seen.add(entry)
        _validate_sha256_hex("selection_stable_hash", self.selection_stable_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_code_id": self.selected_code_id,
            "selected_code_family": self.selected_code_family,
            "selection_confidence": float(self.selection_confidence),
            "ranking_order": self.ranking_order,
            "selection_stable_hash": self.selection_stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BridgeMigrationInput:
    source_code_id: str
    source_code_family: str
    target_code_family: str
    migration_admissible: bool
    migration_confidence: float
    migration_projected_loss: float
    migration_distance_retention: float
    migration_stable_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str("source_code_id", self.source_code_id)
        _validate_non_empty_str("source_code_family", self.source_code_family)
        _validate_non_empty_str("target_code_family", self.target_code_family)
        _validate_bool("migration_admissible", self.migration_admissible)
        _validate_unit_interval_float("migration_confidence", self.migration_confidence)
        _validate_unit_interval_float("migration_projected_loss", self.migration_projected_loss)
        _validate_unit_interval_float("migration_distance_retention", self.migration_distance_retention)
        _validate_sha256_hex("migration_stable_hash", self.migration_stable_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_code_id": self.source_code_id,
            "source_code_family": self.source_code_family,
            "target_code_family": self.target_code_family,
            "migration_admissible": self.migration_admissible,
            "migration_confidence": float(self.migration_confidence),
            "migration_projected_loss": float(self.migration_projected_loss),
            "migration_distance_retention": float(self.migration_distance_retention),
            "migration_stable_hash": self.migration_stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BridgeBenchmarkInput:
    best_candidate_id: str
    best_candidate_family: str
    best_utility: float
    baseline_utility: float
    improvement_margin: float
    cross_family_winner: bool
    benchmark_admissible: bool
    benchmark_stable_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str("best_candidate_id", self.best_candidate_id)
        _validate_non_empty_str("best_candidate_family", self.best_candidate_family)
        _validate_unit_interval_float("best_utility", self.best_utility)
        _validate_unit_interval_float("baseline_utility", self.baseline_utility)
        _validate_unit_interval_float("improvement_margin", self.improvement_margin)
        _validate_bool("cross_family_winner", self.cross_family_winner)
        _validate_bool("benchmark_admissible", self.benchmark_admissible)
        _validate_sha256_hex("benchmark_stable_hash", self.benchmark_stable_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "best_candidate_family": self.best_candidate_family,
            "best_utility": float(self.best_utility),
            "baseline_utility": float(self.baseline_utility),
            "improvement_margin": float(self.improvement_margin),
            "cross_family_winner": self.cross_family_winner,
            "benchmark_admissible": self.benchmark_admissible,
            "benchmark_stable_hash": self.benchmark_stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BridgePolicyInput:
    selected_action: str
    target_code_id: str
    target_code_family: str
    stay_on_current_code: bool
    approve_migration: bool
    recommend_orchestration: bool
    policy_confidence: float
    improvement_score: float
    risk_score: float
    escalation_level: str
    policy_stable_hash: str

    def __post_init__(self) -> None:
        if self.selected_action not in ALLOWED_SELECTED_ACTIONS:
            raise ValueError("selected_action must be an allowed action")
        _validate_non_empty_str("target_code_id", self.target_code_id)
        _validate_non_empty_str("target_code_family", self.target_code_family)
        _validate_bool("stay_on_current_code", self.stay_on_current_code)
        _validate_bool("approve_migration", self.approve_migration)
        _validate_bool("recommend_orchestration", self.recommend_orchestration)
        _validate_unit_interval_float("policy_confidence", self.policy_confidence)
        _validate_unit_interval_float("improvement_score", self.improvement_score)
        _validate_unit_interval_float("risk_score", self.risk_score)
        if self.escalation_level not in ALLOWED_ESCALATION_LEVELS:
            raise ValueError("escalation_level must be an allowed escalation level")
        _validate_sha256_hex("policy_stable_hash", self.policy_stable_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "target_code_id": self.target_code_id,
            "target_code_family": self.target_code_family,
            "stay_on_current_code": self.stay_on_current_code,
            "approve_migration": self.approve_migration,
            "recommend_orchestration": self.recommend_orchestration,
            "policy_confidence": float(self.policy_confidence),
            "improvement_score": float(self.improvement_score),
            "risk_score": float(self.risk_score),
            "escalation_level": self.escalation_level,
            "policy_stable_hash": self.policy_stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BridgeStep:
    step_index: int
    step_name: str
    source_stage: str
    target_stage: str
    blocking: bool
    ready: bool
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be int >= 0")
        _validate_non_empty_str("step_name", self.step_name)
        _validate_non_empty_str("source_stage", self.source_stage)
        _validate_non_empty_str("target_stage", self.target_stage)
        _validate_bool("blocking", self.blocking)
        _validate_bool("ready", self.ready)
        _validate_non_empty_str("detail", self.detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "blocking": self.blocking,
            "ready": self.ready,
            "detail": self.detail,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BridgeReadiness:
    structurally_consistent: bool
    selection_policy_aligned: bool
    migration_policy_aligned: bool
    benchmark_policy_aligned: bool
    bridge_ready: bool
    bridge_confidence: float
    bridge_risk: float
    rationale: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_bool("structurally_consistent", self.structurally_consistent)
        _validate_bool("selection_policy_aligned", self.selection_policy_aligned)
        _validate_bool("migration_policy_aligned", self.migration_policy_aligned)
        _validate_bool("benchmark_policy_aligned", self.benchmark_policy_aligned)
        _validate_bool("bridge_ready", self.bridge_ready)
        _validate_unit_interval_float("bridge_confidence", self.bridge_confidence)
        _validate_unit_interval_float("bridge_risk", self.bridge_risk)
        if not isinstance(self.rationale, tuple):
            raise ValueError("rationale must be tuple[str, ...]")
        for item in self.rationale:
            _validate_non_empty_str("rationale entry", item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structurally_consistent": self.structurally_consistent,
            "selection_policy_aligned": self.selection_policy_aligned,
            "migration_policy_aligned": self.migration_policy_aligned,
            "benchmark_policy_aligned": self.benchmark_policy_aligned,
            "bridge_ready": self.bridge_ready,
            "bridge_confidence": float(self.bridge_confidence),
            "bridge_risk": float(self.bridge_risk),
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MultiCodeBridgeReceipt:
    selection_input: BridgeSelectionInput
    migration_input: BridgeMigrationInput
    benchmark_input: BridgeBenchmarkInput
    policy_input: BridgePolicyInput
    bridge_steps: tuple[BridgeStep, ...]
    readiness: BridgeReadiness
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.bridge_steps, tuple) or not self.bridge_steps:
            raise ValueError("bridge_steps must be a non-empty tuple[BridgeStep, ...]")
        for step in self.bridge_steps:
            if not isinstance(step, BridgeStep):
                raise ValueError("bridge_steps entries must be BridgeStep")
        _validate_non_empty_str("schema_version", self.schema_version)
        _validate_sha256_hex("replay_identity", self.replay_identity)
        _validate_sha256_hex("stable_hash", self.stable_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_input": self.selection_input.to_dict(),
            "migration_input": self.migration_input.to_dict(),
            "benchmark_input": self.benchmark_input.to_dict(),
            "policy_input": self.policy_input.to_dict(),
            "bridge_steps": tuple(step.to_dict() for step in self.bridge_steps),
            "readiness": self.readiness.to_dict(),
            "schema_version": self.schema_version,
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return sha256_hex(_receipt_hash_payload(self))


def _consistency_snapshot(
    selection_input: BridgeSelectionInput,
    migration_input: BridgeMigrationInput,
    benchmark_input: BridgeBenchmarkInput,
    policy_input: BridgePolicyInput,
) -> tuple[bool, bool, bool, bool, tuple[str, ...]]:
    action = policy_input.selected_action

    selection_policy_aligned = (
        selection_input.selected_code_family == policy_input.target_code_family
        and selection_input.selected_code_id == policy_input.target_code_id
    )

    migration_policy_aligned = (
        migration_input.target_code_family == policy_input.target_code_family
        and selection_input.selected_code_family == policy_input.target_code_family
    )

    benchmark_policy_aligned = (
        benchmark_input.best_candidate_family == policy_input.target_code_family
        and benchmark_input.best_candidate_id == policy_input.target_code_id
        and benchmark_input.benchmark_admissible
    )

    structural_ok = True
    rationale: list[str] = []

    if action in ("stay", "defer", "reject"):
        if selection_policy_aligned and policy_input.stay_on_current_code:
            rationale.append("policy action keeps current runtime target")
        else:
            structural_ok = False
            rationale.append("policy action current-target semantics not satisfied")

    if action == "stay":
        if policy_input.approve_migration or policy_input.recommend_orchestration:
            structural_ok = False
            rationale.append("stay action forbids migration approval and orchestration recommendation")

    if action == "switch":
        if policy_input.stay_on_current_code or policy_input.approve_migration:
            structural_ok = False
            rationale.append("switch action requires candidate-target handoff without migration approval")

    if action == "migrate":
        if policy_input.stay_on_current_code:
            structural_ok = False
            rationale.append("migrate action cannot stay on current runtime target")
        if not policy_input.approve_migration:
            structural_ok = False
            rationale.append("migration approval required but not satisfied")

    if action == "orchestrate":
        if policy_input.stay_on_current_code:
            structural_ok = False
            rationale.append("orchestrate action cannot stay on current runtime target")
        if not policy_input.recommend_orchestration:
            structural_ok = False
            rationale.append("orchestration recommendation required but not satisfied")

    if selection_policy_aligned:
        rationale.append("selection and policy targets align")
    else:
        rationale.append("selection and policy targets mismatch")

    if policy_input.approve_migration:
        if migration_policy_aligned:
            rationale.append("migration target aligns with policy")
        else:
            rationale.append("migration target mismatch with policy")

    if policy_input.recommend_orchestration or action == "orchestrate":
        if benchmark_policy_aligned:
            rationale.append("benchmark winner supports policy target")
        else:
            rationale.append("benchmark winner does not support policy target")

    if not structural_ok:
        rationale.append("bridge blocked by structural inconsistency")

    return structural_ok, selection_policy_aligned, migration_policy_aligned, benchmark_policy_aligned, tuple(rationale)


def _compute_readiness(
    selection_input: BridgeSelectionInput,
    migration_input: BridgeMigrationInput,
    benchmark_input: BridgeBenchmarkInput,
    policy_input: BridgePolicyInput,
) -> BridgeReadiness:
    (
        structurally_consistent,
        selection_policy_aligned,
        migration_policy_aligned,
        benchmark_policy_aligned,
        rationale_seed,
    ) = _consistency_snapshot(selection_input, migration_input, benchmark_input, policy_input)

    action = policy_input.selected_action
    migration_required = action in ("migrate", "orchestrate")

    alignment_score = (
        (1.0 if selection_policy_aligned else 0.0)
        + (1.0 if migration_policy_aligned else 0.0)
        + (1.0 if benchmark_policy_aligned else 0.0)
    ) / 3.0

    bridge_confidence = _clamp01(
        0.20 * selection_input.selection_confidence
        + 0.20 * migration_input.migration_confidence
        + 0.15 * benchmark_input.improvement_margin
        + 0.15 * policy_input.policy_confidence
        + 0.10 * policy_input.improvement_score
        + 0.20 * alignment_score
    )
    if not structurally_consistent:
        bridge_confidence = _clamp01(bridge_confidence * 0.5)
    if action in ("defer", "reject"):
        bridge_confidence = _clamp01(bridge_confidence * 0.5)

    mismatch_penalty = 0.0 if selection_policy_aligned else 1.0
    migration_penalty = 1.0 if (migration_required and not migration_input.migration_admissible) else 0.0
    weak_benchmark_penalty = 1.0 - benchmark_input.improvement_margin
    bridge_risk = _clamp01(
        0.50 * policy_input.risk_score
        + 0.20 * (1.0 - policy_input.improvement_score)
        + 0.10 * weak_benchmark_penalty
        + 0.10 * migration_penalty
        + 0.10 * mismatch_penalty
    )

    if action == "stay":
        bridge_ready = structurally_consistent and selection_policy_aligned
    elif action == "switch":
        bridge_ready = structurally_consistent and selection_policy_aligned
    elif action == "migrate":
        bridge_ready = (
            structurally_consistent
            and selection_policy_aligned
            and migration_policy_aligned
            and migration_input.migration_admissible
        )
    elif action == "orchestrate":
        bridge_ready = (
            structurally_consistent
            and selection_policy_aligned
            and migration_policy_aligned
            and benchmark_policy_aligned
            and migration_input.migration_admissible
            and policy_input.recommend_orchestration
        )
    else:
        bridge_ready = False

    rationale = list(rationale_seed)
    if bridge_ready:
        rationale.append("bridge readiness satisfied")
    elif action == "defer":
        rationale.append("defer action emits hold-state handoff")
    elif action == "reject":
        rationale.append("reject action emits blocked handoff")
    else:
        rationale.append("bridge readiness not satisfied")

    return BridgeReadiness(
        structurally_consistent=structurally_consistent,
        selection_policy_aligned=selection_policy_aligned,
        migration_policy_aligned=migration_policy_aligned,
        benchmark_policy_aligned=benchmark_policy_aligned,
        bridge_ready=bridge_ready,
        bridge_confidence=bridge_confidence,
        bridge_risk=bridge_risk,
        rationale=tuple(rationale),
    )


def _build_steps(readiness: BridgeReadiness) -> tuple[BridgeStep, ...]:
    ready_map = {
        "validate_selection_artifact": True,
        "validate_migration_artifact": True,
        "validate_benchmark_artifact": True,
        "validate_policy_artifact": True,
        "reconcile_target_alignment": readiness.selection_policy_aligned,
        "assess_bridge_readiness": readiness.structurally_consistent,
        "emit_execution_handoff": readiness.bridge_ready,
    }

    return tuple(
        BridgeStep(
            step_index=index,
            step_name=step_name,
            source_stage="analysis",
            target_stage="runtime_handoff",
            blocking=True,
            ready=ready_map[step_name],
            detail=(
                "ready" if ready_map[step_name] else "not_ready"
            ),
        )
        for index, step_name in enumerate(BRIDGE_STEP_NAMES)
    )


def _replay_payload(
    selection_input: BridgeSelectionInput,
    migration_input: BridgeMigrationInput,
    benchmark_input: BridgeBenchmarkInput,
    policy_input: BridgePolicyInput,
    readiness: BridgeReadiness,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_stable_hash": selection_input.selection_stable_hash,
        "migration_stable_hash": migration_input.migration_stable_hash,
        "benchmark_stable_hash": benchmark_input.benchmark_stable_hash,
        "policy_stable_hash": policy_input.policy_stable_hash,
        "selected_action": policy_input.selected_action,
        "bridge_ready": readiness.bridge_ready,
    }


def _receipt_hash_payload(receipt: MultiCodeBridgeReceipt) -> dict[str, Any]:
    return {
        "selection_input": receipt.selection_input.to_dict(),
        "migration_input": receipt.migration_input.to_dict(),
        "benchmark_input": receipt.benchmark_input.to_dict(),
        "policy_input": receipt.policy_input.to_dict(),
        "bridge_steps": tuple(step.to_dict() for step in receipt.bridge_steps),
        "readiness": receipt.readiness.to_dict(),
        "schema_version": receipt.schema_version,
        "replay_identity": receipt.replay_identity,
    }


def build_multi_code_bridge(
    selection_input: BridgeSelectionInput,
    migration_input: BridgeMigrationInput,
    benchmark_input: BridgeBenchmarkInput,
    policy_input: BridgePolicyInput,
) -> MultiCodeBridgeReceipt:
    if not isinstance(selection_input, BridgeSelectionInput):
        raise ValueError("selection_input must be BridgeSelectionInput")
    if not isinstance(migration_input, BridgeMigrationInput):
        raise ValueError("migration_input must be BridgeMigrationInput")
    if not isinstance(benchmark_input, BridgeBenchmarkInput):
        raise ValueError("benchmark_input must be BridgeBenchmarkInput")
    if not isinstance(policy_input, BridgePolicyInput):
        raise ValueError("policy_input must be BridgePolicyInput")

    readiness = _compute_readiness(selection_input, migration_input, benchmark_input, policy_input)
    bridge_steps = _build_steps(readiness)
    replay_identity = sha256_hex(
        _replay_payload(selection_input, migration_input, benchmark_input, policy_input, readiness)
    )

    receipt_no_hash = MultiCodeBridgeReceipt(
        selection_input=selection_input,
        migration_input=migration_input,
        benchmark_input=benchmark_input,
        policy_input=policy_input,
        bridge_steps=bridge_steps,
        readiness=readiness,
        schema_version=SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash="0" * 64,
    )
    stable_hash = sha256_hex(_receipt_hash_payload(receipt_no_hash))

    return MultiCodeBridgeReceipt(
        selection_input=selection_input,
        migration_input=migration_input,
        benchmark_input=benchmark_input,
        policy_input=policy_input,
        bridge_steps=bridge_steps,
        readiness=readiness,
        schema_version=SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


__all__ = [
    "ALLOWED_CODE_FAMILIES",
    "ALLOWED_SELECTED_ACTIONS",
    "ALLOWED_ESCALATION_LEVELS",
    "BRIDGE_STEP_NAMES",
    "SCHEMA_VERSION",
    "BridgeSelectionInput",
    "BridgeMigrationInput",
    "BridgeBenchmarkInput",
    "BridgePolicyInput",
    "BridgeStep",
    "BridgeReadiness",
    "MultiCodeBridgeReceipt",
    "build_multi_code_bridge",
]
