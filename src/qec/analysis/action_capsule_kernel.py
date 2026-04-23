"""v146.1 — Execution Bridge (Proof-Carrying Actions)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import string
from typing import Any

from qec.analysis.bounded_refinement_kernel import RefinementReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.deterministic_transition_policy import TransitionPolicyReceipt
from qec.analysis.governed_orchestration_layer import GovernedOrchestrationReceipt

_ALLOWED_TARGET_SCOPES = frozenset({"transition", "refinement", "orchestration"})
_ALLOWED_ACTION_TYPE = "governed_transition_commitment"


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _validate_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _validate_sha256_hex(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be 64-char hex string")
    if any(ch not in string.hexdigits for ch in value) or value.lower() != value:
        raise ValueError(f"{field_name} must be 64-char hex string")
    return value


def _validate_reasons_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be tuple[str, ...]")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name} must contain non-empty strings")
    return value


def _normalize_json_payload(value: Any, field_name: str) -> _JSONValue:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains non-finite float")
        return float(value)
    if isinstance(value, tuple):
        return tuple(_normalize_json_payload(item, field_name) for item in value)
    if isinstance(value, list):
        return tuple(_normalize_json_payload(item, field_name) for item in value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field_name} contains non-string key")
        return {key: _normalize_json_payload(value[key], field_name) for key in sorted(value)}
    raise ValueError(f"{field_name} contains non-canonicalizable content")


def _forbidden_execution_payload(value: Any) -> bool:
    forbidden_key_tokens = (
        "command",
        "callback",
        "hook",
        "socket",
        "subprocess",
        "exec",
        "runtime",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = key.lower()
            if any(token in key_lower for token in forbidden_key_tokens):
                return True
            if _forbidden_execution_payload(item):
                return True
        return False
    if isinstance(value, (tuple, list)):
        return any(_forbidden_execution_payload(item) for item in value)
    if callable(value):
        return True
    return False


def _extract_mapping(receipt: Any, field_name: str) -> dict[str, Any]:
    if not hasattr(receipt, "to_dict"):
        raise ValueError(f"{field_name} must support to_dict()")
    payload = receipt.to_dict()
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name}.to_dict() must return dict")
    return payload


def _get_hash(receipt_dict: dict[str, Any], name: str, aliases: tuple[str, ...], *, required: bool) -> str | None:
    for key in aliases:
        value = receipt_dict.get(key)
        if value is not None:
            return _validate_sha256_hex(value, name)
    if required:
        raise ValueError(f"missing required lineage field: {name}")
    return None


def _require_self_hash(receipt: Any, field_name: str) -> str:
    stable_hash = getattr(receipt, "stable_hash", None)
    _validate_sha256_hex(stable_hash, f"{field_name}.stable_hash")
    if not hasattr(receipt, "computed_stable_hash"):
        raise ValueError(f"{field_name} must support computed_stable_hash()")
    computed = receipt.computed_stable_hash()
    if stable_hash != computed:
        raise ValueError(f"{field_name} stable_hash is invalid")
    return stable_hash


def _compute_replay_identity(
    transition_receipt_hash: str,
    refinement_receipt_hash: str,
    governance_receipt_hash: str,
) -> str:
    return sha256_hex(
        {
            "transition_receipt_hash": transition_receipt_hash,
            "refinement_receipt_hash": refinement_receipt_hash,
            "governance_receipt_hash": governance_receipt_hash,
        }
    )


def _assert_canonical_round_trip(value: Any, field_name: str) -> None:
    canonical = canonical_json(value)
    reparsed = json.loads(canonical)
    recanonical = canonical_json(reparsed)
    if canonical != recanonical:
        raise ValueError(f"{field_name} canonical serialization round-trip is unstable")


def _require_nested_hash_recomputation(value: Any, field_name: str) -> str:
    stable_hash = getattr(value, "stable_hash", None)
    _validate_sha256_hex(stable_hash, f"{field_name}.stable_hash")
    if not hasattr(value, "computed_stable_hash"):
        raise ValueError(f"{field_name} must support computed_stable_hash()")
    computed = value.computed_stable_hash()
    _validate_sha256_hex(computed, f"{field_name}.computed_stable_hash")
    if computed != stable_hash:
        raise ValueError(f"{field_name} stable_hash is invalid")
    if not hasattr(value, "to_dict"):
        raise ValueError(f"{field_name} must support to_dict()")
    mapping = value.to_dict()
    if not isinstance(mapping, dict):
        raise ValueError(f"{field_name}.to_dict() must return dict")
    recomputed = sha256_hex({k: mapping[k] for k in mapping if k != "stable_hash"})
    if recomputed != stable_hash:
        raise ValueError(f"{field_name} canonical field recomputation hash is invalid")
    return stable_hash


@dataclass(frozen=True)
class ActionDescriptor:
    action_type: str
    target_scope: str
    action_payload: dict[str, Any]
    bound_constraints: dict[str, Any]
    representation_only: bool
    payload_schema_version: str

    def __post_init__(self) -> None:
        if self.action_type != _ALLOWED_ACTION_TYPE:
            raise ValueError("action_type is invalid")
        if self.target_scope not in _ALLOWED_TARGET_SCOPES:
            raise ValueError("target_scope is invalid")
        if not isinstance(self.action_payload, dict):
            raise ValueError("action_payload must be dict")
        if not isinstance(self.bound_constraints, dict):
            raise ValueError("bound_constraints must be dict")
        if self.representation_only is not True:
            raise ValueError("representation_only must be True")
        _validate_non_empty_string(self.payload_schema_version, "payload_schema_version")

        expected_payload_keys = (
            "governance_linkage",
            "refined_outcome",
            "selected_transition",
        )
        if tuple(sorted(self.action_payload.keys())) != expected_payload_keys:
            raise ValueError("action_payload must use minimal governed commitment schema")
        if _forbidden_execution_payload(self.action_payload) or _forbidden_execution_payload(self.bound_constraints):
            raise ValueError("payload must not encode execution semantics")

        normalized_payload = _normalize_json_payload(self.action_payload, "action_payload")
        normalized_constraints = _normalize_json_payload(self.bound_constraints, "bound_constraints")
        object.__setattr__(self, "action_payload", normalized_payload)
        object.__setattr__(self, "bound_constraints", normalized_constraints)

        selected_transition = self.action_payload.get("selected_transition")
        if not isinstance(selected_transition, dict) or tuple(sorted(selected_transition.keys())) != (
            "ordering_signature",
            "transition_hash",
        ):
            raise ValueError("selected_transition must have exactly keys: ordering_signature, transition_hash")
        _validate_non_empty_string(selected_transition["ordering_signature"], "selected_transition.ordering_signature")
        _validate_sha256_hex(selected_transition["transition_hash"], "selected_transition.transition_hash")

        refined_outcome = self.action_payload.get("refined_outcome")
        if not isinstance(refined_outcome, dict) or tuple(sorted(refined_outcome.keys())) != (
            "classification",
            "convergence_metric",
            "refinement_hash",
        ):
            raise ValueError("refined_outcome must have exactly keys: classification, convergence_metric, refinement_hash")
        _validate_non_empty_string(refined_outcome["classification"], "refined_outcome.classification")
        _validate_sha256_hex(refined_outcome["refinement_hash"], "refined_outcome.refinement_hash")

        governance_linkage = self.action_payload.get("governance_linkage")
        if not isinstance(governance_linkage, dict) or tuple(sorted(governance_linkage.keys())) != (
            "governance_hash",
            "verdict",
            "verdict_hash",
        ):
            raise ValueError("governance_linkage must have exactly keys: governance_hash, verdict, verdict_hash")
        if governance_linkage["verdict"] != "allow":
            raise ValueError("governance_linkage.verdict must be allow")
        _validate_sha256_hex(governance_linkage["verdict_hash"], "governance_linkage.verdict_hash")
        _validate_sha256_hex(governance_linkage["governance_hash"], "governance_linkage.governance_hash")
        convergence_metric = refined_outcome["convergence_metric"]
        if isinstance(convergence_metric, bool) or not isinstance(convergence_metric, (int, float)):
            raise ValueError("refined_outcome.convergence_metric must be numeric")
        normalized_metric = float(convergence_metric)
        if not math.isfinite(normalized_metric):
            raise ValueError("refined_outcome.convergence_metric must be finite")
        if normalized_metric != round(normalized_metric, 12):
            raise ValueError("refined_outcome.convergence_metric must be pre-rounded to 12 decimals")
        if not isinstance(convergence_metric, float) or refined_outcome["convergence_metric"] != normalized_metric:
            normalized_refined_outcome = {
                "classification": refined_outcome["classification"],
                "convergence_metric": normalized_metric,
                "refinement_hash": refined_outcome["refinement_hash"],
            }
            normalized_full_payload = {
                "governance_linkage": self.action_payload["governance_linkage"],
                "refined_outcome": normalized_refined_outcome,
                "selected_transition": self.action_payload["selected_transition"],
            }
            object.__setattr__(self, "action_payload", normalized_full_payload)
        _assert_canonical_round_trip(self.to_dict(), "ActionDescriptor")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "action_type": self.action_type,
            "target_scope": self.target_scope,
            "action_payload": self.action_payload,
            "bound_constraints": self.bound_constraints,
            "representation_only": self.representation_only,
            "payload_schema_version": self.payload_schema_version,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return self._payload_without_hash()

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class ProofCarryingActionCapsule:
    capsule_version: str
    capsule_kind: str
    action_descriptor: ActionDescriptor
    transition_receipt_hash: str
    refinement_receipt_hash: str
    governance_receipt_hash: str
    stress_hash: str
    state_hash: str
    mesh_hash: str
    transition_hash: str
    refinement_hash: str
    governance_hash: str
    governance_verdict: str
    admissibility_status: str
    certification_status: str
    boundedness_status: str
    replay_safety_status: str
    analysis_only: bool
    non_executing: bool
    side_effect_free: bool
    admissibility_reasons: tuple[str, ...]
    certification_reasons: tuple[str, ...]
    validation_notes: tuple[str, ...]
    capsule_hash: str
    replay_identity: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.capsule_version, "capsule_version")
        if self.capsule_kind != "proof_carrying_action_capsule":
            raise ValueError("capsule_kind is invalid")
        if not isinstance(self.action_descriptor, ActionDescriptor):
            raise ValueError("action_descriptor must be ActionDescriptor")
        for field_name in (
            "transition_receipt_hash",
            "refinement_receipt_hash",
            "governance_receipt_hash",
            "stress_hash",
            "state_hash",
            "mesh_hash",
            "transition_hash",
            "refinement_hash",
            "governance_hash",
            "capsule_hash",
            "replay_identity",
        ):
            _validate_sha256_hex(getattr(self, field_name), field_name)
        if self.governance_verdict != "allow":
            raise ValueError("governance_verdict must be allow")
        if self.admissibility_status != "admissible":
            raise ValueError("admissibility_status must be admissible")
        if self.certification_status != "certified":
            raise ValueError("certification_status must be certified")
        if self.boundedness_status != "bounded":
            raise ValueError("boundedness_status must be bounded")
        if self.replay_safety_status != "replay_safe":
            raise ValueError("replay_safety_status must be replay_safe")
        if self.analysis_only is not True:
            raise ValueError("analysis_only must be True")
        if self.non_executing is not True:
            raise ValueError("non_executing must be True")
        if self.side_effect_free is not True:
            raise ValueError("side_effect_free must be True")

        _validate_reasons_tuple(self.admissibility_reasons, "admissibility_reasons")
        _validate_reasons_tuple(self.certification_reasons, "certification_reasons")
        _validate_reasons_tuple(self.validation_notes, "validation_notes")

        if self.transition_hash != self.transition_receipt_hash:
            raise ValueError("transition_hash must equal transition_receipt_hash")
        if self.refinement_hash != self.refinement_receipt_hash:
            raise ValueError("refinement_hash must equal refinement_receipt_hash")
        if self.governance_hash != self.governance_receipt_hash:
            raise ValueError("governance_hash must equal governance_receipt_hash")

        expected_replay_identity = _compute_replay_identity(
            self.transition_receipt_hash,
            self.refinement_receipt_hash,
            self.governance_receipt_hash,
        )
        if self.replay_identity != expected_replay_identity:
            raise ValueError("replay_identity mismatch")

        _ap = self.action_descriptor.action_payload
        if _ap["selected_transition"]["transition_hash"] != self.transition_receipt_hash:
            raise ValueError("action_descriptor transition_hash mismatch with transition_receipt_hash")
        if _ap["refined_outcome"]["refinement_hash"] != self.refinement_receipt_hash:
            raise ValueError("action_descriptor refinement_hash mismatch with refinement_receipt_hash")
        if _ap["governance_linkage"]["governance_hash"] != self.governance_receipt_hash:
            raise ValueError("action_descriptor governance_hash mismatch with governance_receipt_hash")
        if _ap["governance_linkage"]["verdict"] != self.governance_verdict:
            raise ValueError("action_descriptor verdict mismatch with governance_verdict")

        if self.capsule_hash != self.stable_hash():
            raise ValueError("capsule_hash must match canonical capsule payload")
        _assert_canonical_round_trip(self.to_dict(), "ProofCarryingActionCapsule")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "capsule_version": self.capsule_version,
            "capsule_kind": self.capsule_kind,
            "action_descriptor": self.action_descriptor.to_dict(),
            "transition_receipt_hash": self.transition_receipt_hash,
            "refinement_receipt_hash": self.refinement_receipt_hash,
            "governance_receipt_hash": self.governance_receipt_hash,
            "stress_hash": self.stress_hash,
            "state_hash": self.state_hash,
            "mesh_hash": self.mesh_hash,
            "transition_hash": self.transition_hash,
            "refinement_hash": self.refinement_hash,
            "governance_hash": self.governance_hash,
            "governance_verdict": self.governance_verdict,
            "admissibility_status": self.admissibility_status,
            "certification_status": self.certification_status,
            "boundedness_status": self.boundedness_status,
            "replay_safety_status": self.replay_safety_status,
            "analysis_only": self.analysis_only,
            "non_executing": self.non_executing,
            "side_effect_free": self.side_effect_free,
            "admissibility_reasons": self.admissibility_reasons,
            "certification_reasons": self.certification_reasons,
            "validation_notes": self.validation_notes,
            "replay_identity": self.replay_identity,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "capsule_hash": self.capsule_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class ActionCapsuleProofReceipt:
    receipt_version: str
    capsule_hash: str
    replay_identity: str
    transition_receipt_hash: str
    refinement_receipt_hash: str
    governance_receipt_hash: str
    verification_passed: bool
    verification_checks: tuple[str, ...]
    proof_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.receipt_version, "receipt_version")
        for field_name in (
            "capsule_hash",
            "replay_identity",
            "transition_receipt_hash",
            "refinement_receipt_hash",
            "governance_receipt_hash",
            "proof_receipt_hash",
        ):
            _validate_sha256_hex(getattr(self, field_name), field_name)
        if self.verification_passed is not True:
            raise ValueError("verification_passed must be True")
        _validate_reasons_tuple(self.verification_checks, "verification_checks")
        if self.proof_receipt_hash != self.stable_hash():
            raise ValueError("proof_receipt_hash must match canonical proof receipt payload")
        _assert_canonical_round_trip(self.to_dict(), "ActionCapsuleProofReceipt")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "receipt_version": self.receipt_version,
            "capsule_hash": self.capsule_hash,
            "replay_identity": self.replay_identity,
            "transition_receipt_hash": self.transition_receipt_hash,
            "refinement_receipt_hash": self.refinement_receipt_hash,
            "governance_receipt_hash": self.governance_receipt_hash,
            "verification_passed": self.verification_passed,
            "verification_checks": self.verification_checks,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "proof_receipt_hash": self.proof_receipt_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


def build_action_capsule(
    transition_receipt: TransitionPolicyReceipt,
    refinement_receipt: RefinementReceipt,
    governance_receipt: GovernedOrchestrationReceipt,
) -> tuple[ProofCarryingActionCapsule, ActionCapsuleProofReceipt]:
    if not isinstance(transition_receipt, TransitionPolicyReceipt):
        raise ValueError("transition_receipt must be TransitionPolicyReceipt")
    if not isinstance(refinement_receipt, RefinementReceipt):
        raise ValueError("refinement_receipt must be RefinementReceipt")
    if not isinstance(governance_receipt, GovernedOrchestrationReceipt):
        raise ValueError("governance_receipt must be GovernedOrchestrationReceipt")

    transition_receipt_hash = _require_self_hash(transition_receipt, "transition_receipt")
    refinement_receipt_hash = _require_self_hash(refinement_receipt, "refinement_receipt")
    governance_receipt_hash = _require_self_hash(governance_receipt, "governance_receipt")
    transition_snapshot = canonical_json(transition_receipt.to_dict())
    refinement_snapshot = canonical_json(refinement_receipt.to_dict())
    governance_snapshot = canonical_json(governance_receipt.to_dict())

    _require_nested_hash_recomputation(
        transition_receipt.selected_decision,
        "transition_receipt.selected_decision",
    )
    for _i, _step in enumerate(refinement_receipt.steps):
        _require_nested_hash_recomputation(_step, f"refinement_receipt.steps[{_i}]")
    _require_nested_hash_recomputation(governance_receipt.policy, "governance_receipt.policy")
    for _i, _check in enumerate(governance_receipt.checks):
        _require_nested_hash_recomputation(_check, f"governance_receipt.checks[{_i}]")
    _require_nested_hash_recomputation(governance_receipt.verdict, "governance_receipt.verdict")

    transition_payload = _extract_mapping(transition_receipt, "transition_receipt")
    refinement_payload = _extract_mapping(refinement_receipt, "refinement_receipt")
    governance_payload = _extract_mapping(governance_receipt, "governance_receipt")

    if refinement_payload.get("input_policy_hash") != transition_receipt_hash:
        raise ValueError("refinement input_policy_hash must match transition_receipt.stable_hash")
    if governance_payload.get("input_transition_hash") != transition_receipt_hash:
        raise ValueError("governance input_transition_hash must match transition_receipt.stable_hash")
    if governance_payload.get("input_refinement_hash") != refinement_receipt_hash:
        raise ValueError("governance input_refinement_hash must match refinement_receipt.stable_hash")

    verdict = governance_payload.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("governance verdict must be dict")
    if verdict.get("verdict") != "allow":
        raise ValueError("governance verdict must be allow")

    mesh_hash = _get_hash(
        transition_payload,
        "mesh_hash",
        ("mesh_hash", "input_mesh_hash", "input_receipt_hash", "mesh_receipt_hash"),
        required=True,
    )
    stress_hash = _get_hash(
        transition_payload,
        "stress_hash",
        ("stress_hash", "stress_receipt_hash", "input_stress_hash"),
        required=False,
    )
    state_hash = _get_hash(
        transition_payload,
        "state_hash",
        ("state_hash", "input_state_hash", "state_receipt_hash"),
        required=False,
    )

    if stress_hash is None:
        stress_hash = mesh_hash
    if state_hash is None:
        state_hash = mesh_hash
    if refinement_receipt.convergence_metric != round(float(refinement_receipt.convergence_metric), 12):
        raise ValueError("refinement_receipt.convergence_metric must be pre-rounded to 12 decimals")

    action_payload = {
        "selected_transition": {
            "ordering_signature": transition_receipt.selected_decision.selected_ordering_signature,
            "transition_hash": transition_receipt_hash,
        },
        "refined_outcome": {
            "classification": refinement_receipt.classification,
            "convergence_metric": round(float(refinement_receipt.convergence_metric), 12),
            "refinement_hash": refinement_receipt_hash,
        },
        "governance_linkage": {
            "verdict": "allow",
            "verdict_hash": governance_receipt.verdict.stable_hash,
            "governance_hash": governance_receipt_hash,
        },
    }
    bound_constraints = {
        "bounded_refinement": True,
        "max_iterations": refinement_receipt.iteration_count,
        "analysis_only": True,
    }

    descriptor = ActionDescriptor(
        action_type=_ALLOWED_ACTION_TYPE,
        target_scope="orchestration",
        action_payload=action_payload,
        bound_constraints=bound_constraints,
        representation_only=True,
        payload_schema_version="v146.1",
    )

    replay_identity = _compute_replay_identity(
        transition_receipt_hash,
        refinement_receipt_hash,
        governance_receipt_hash,
    )

    capsule_payload = {
        "capsule_version": "v146.1",
        "capsule_kind": "proof_carrying_action_capsule",
        "action_descriptor": descriptor,
        "transition_receipt_hash": transition_receipt_hash,
        "refinement_receipt_hash": refinement_receipt_hash,
        "governance_receipt_hash": governance_receipt_hash,
        "stress_hash": stress_hash,
        "state_hash": state_hash,
        "mesh_hash": mesh_hash,
        "transition_hash": transition_receipt_hash,
        "refinement_hash": refinement_receipt_hash,
        "governance_hash": governance_receipt_hash,
        "governance_verdict": "allow",
        "admissibility_status": "admissible",
        "certification_status": "certified",
        "boundedness_status": "bounded",
        "replay_safety_status": "replay_safe",
        "analysis_only": True,
        "non_executing": True,
        "side_effect_free": True,
        "admissibility_reasons": (
            "governance_allow_verdict",
            "transition_refinement_governance_linkage_valid",
        ),
        "certification_reasons": (
            "capsule_lineage_hashes_valid",
            "bounded_representation_only_action",
        ),
        "validation_notes": (
            "canonical_json_sha256_enforced",
            "non_executing_analysis_layer_representation",
        ),
        "replay_identity": replay_identity,
    }
    capsule_hash = sha256_hex(
        {
            **capsule_payload,
            "action_descriptor": descriptor.to_dict(),
        }
    )
    capsule = ProofCarryingActionCapsule(**capsule_payload, capsule_hash=capsule_hash)
    rederived_capsule_hash = sha256_hex(capsule._payload_without_hash())
    if rederived_capsule_hash != capsule.capsule_hash:
        raise ValueError("capsule_hash re-derivation mismatch")
    if capsule.replay_identity != _compute_replay_identity(
        capsule.transition_receipt_hash,
        capsule.refinement_receipt_hash,
        capsule.governance_receipt_hash,
    ):
        raise ValueError("replay_identity mismatch")

    verification_checks = (
        "transition_receipt_hash_valid",
        "refinement_receipt_hash_valid",
        "governance_receipt_hash_valid",
        "governance_verdict_allow",
        "capsule_hash_self_valid",
        "replay_identity_valid",
    )

    proof_payload = {
        "receipt_version": "v146.1",
        "capsule_hash": capsule.capsule_hash,
        "replay_identity": capsule.replay_identity,
        "transition_receipt_hash": transition_receipt_hash,
        "refinement_receipt_hash": refinement_receipt_hash,
        "governance_receipt_hash": governance_receipt_hash,
        "verification_passed": True,
        "verification_checks": verification_checks,
    }
    proof_receipt = ActionCapsuleProofReceipt(
        **proof_payload,
        proof_receipt_hash=sha256_hex(proof_payload),
    )
    rederived_proof_hash = sha256_hex(proof_receipt._payload_without_hash())
    if rederived_proof_hash != proof_receipt.proof_receipt_hash:
        raise ValueError("proof_receipt_hash re-derivation mismatch")
    if canonical_json(transition_receipt.to_dict()) != transition_snapshot:
        raise ValueError("transition_receipt was mutated during capsule build")
    if canonical_json(refinement_receipt.to_dict()) != refinement_snapshot:
        raise ValueError("refinement_receipt was mutated during capsule build")
    if canonical_json(governance_receipt.to_dict()) != governance_snapshot:
        raise ValueError("governance_receipt was mutated during capsule build")

    return capsule, proof_receipt


__all__ = [
    "ActionCapsuleProofReceipt",
    "ActionDescriptor",
    "ProofCarryingActionCapsule",
    "build_action_capsule",
]
