"""Fail-closed composition gate for QEC evidence trust (post #617, #618, #619).

Individual proof, capability, and firewall verifiers do not themselves assert
that evidence belongs to one operation. This *opt-in* gate requires an external
verifier-owned, precommitted context binding all three. Deriving the context
from untrusted evidence would invalidate the security claim.

The gate is deterministic, side-effect-free and does not alter historical
canonical artifacts or the protected decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from qec.analysis.capability_privilege_boundary_engine import (
    CapabilityGrant, CapabilityReceipt, PrivilegeDecision,
    authorize_action, generate_privilege_receipt, replay_privilege_decision,
)
from qec.controller.execution_proof import (
    authenticate_experiment_result, experiment_result_binding,
)
from qec.orchestration.deterministic_covenant_engine import DeterministicCovenantExecution
from qec.orchestration.governance_memory_io_boundary_auditor import (
    BoundaryAuditRule, GovernanceMemoryIOBoundaryAuditReport,
)
from qec.orchestration.policy_execution_firewall_kernel import (
    FirewallPolicyRule, PolicyExecutionFirewallDecision,
    evaluate_policy_execution_firewall, validate_firewall_decision,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    ProofCarryingAgentActionCapsule,
)


@dataclass(frozen=True)
class TrustedOperationContext:
    """Pins supplied *before* evaluating evidence by an independent verifier.

    The experiment identity is the same value originally signed under the
    domain-specific experiment binding, not a producer-selected substitute.
    The pinned hashes identify exact permissible objects, including policy.
    """

    operation_id: str
    actor_id: str
    action_id: str
    action_type: str
    action_scope: str
    experiment_domain: str
    experiment_identity: Any
    trusted_signer_id: str
    experiment_result_hash: str
    capability_hash: str
    capsule_hash: str
    covenant_execution_hash: str
    boundary_audit_report_hash: str
    firewall_policy_rules_hash: str


@dataclass(frozen=True)
class ComposedTrustDecision:
    verdict: str
    violations: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return self.verdict == "ALLOW" and not self.violations


def evaluate_composed_evidence_trust(
    context: TrustedOperationContext,
    *,
    experiment: Mapping[str, Any],
    grant: CapabilityGrant,
    privilege_decision: PrivilegeDecision,
    privilege_receipt: CapabilityReceipt,
    capsule: ProofCarryingAgentActionCapsule,
    covenant_execution: DeterministicCovenantExecution,
    boundary_audit_report: GovernanceMemoryIOBoundaryAuditReport,
    firewall_rules: Sequence[FirewallPolicyRule],
    firewall_decision: PolicyExecutionFirewallDecision,
    trusted_signers: Mapping[str, bytes],
    trusted_grant_hashes: Collection[str],
    trusted_boundary_rules: Sequence[BoundaryAuditRule] | None = None,
) -> ComposedTrustDecision:
    """Authenticate and cross-bind exact inputs; never trust producer flags.

    A valid sub-receipt from another run, another signer or another issued
    grant is insufficient. Each verification uses caller-owned trust roots;
    the final ALLOW requires all three boundaries and contextual pins.
    """
    violations: set[str] = set()

    if not isinstance(context, TrustedOperationContext) or not context.operation_id:
        return ComposedTrustDecision("DENY", ("invalid_trusted_context",))

    # The context is external authority, not constructed from candidate data.
    try:
        if not isinstance(experiment, Mapping):
            raise TypeError("experiment must be mapping")
        exp = dict(experiment)
        proof = exp.get("proof", {})
        if (not isinstance(proof, dict)
                or proof.get("signed_payload", {}).get("signer_id") != context.trusted_signer_id):
            violations.add("signer_mismatch")
        if not authenticate_experiment_result(
            exp, trusted_signers,
            domain=context.experiment_domain,
            identity=context.experiment_identity,
        ):
            violations.add("unauthenticated_experiment")
        elif experiment_result_binding(
            exp, domain=context.experiment_domain,
            identity=context.experiment_identity,
        )["result_hash"] != context.experiment_result_hash:
            violations.add("experiment_run_mismatch")
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError):
        violations.add("unauthenticated_experiment")

    try:
        if (grant.capability_hash != context.capability_hash
                or grant.actor_id != context.actor_id):
            violations.add("capability_context_mismatch")
        expected = authorize_action(
            grant, context.actor_id, context.action_type, (context.action_scope,),
            trusted_grant_hashes=trusted_grant_hashes,
        )
        if (expected.privilege_verdict != "ALLOW"
                or privilege_decision != expected
                or privilege_decision.action_id != capsule.action_type):
            violations.add("unauthorized_capability")
        elif (replay_privilege_decision(
                privilege_decision.to_canonical_bytes(), grant,
                trusted_grant_hashes=trusted_grant_hashes,
        ) != privilege_decision
                or generate_privilege_receipt(
                    privilege_decision, grant=grant,
                    trusted_grant_hashes=trusted_grant_hashes,
                ) != privilege_receipt):
            violations.add("unbound_capability_receipt")
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError):
        violations.add("unauthorized_capability")

    try:
        if (capsule.action_id != context.action_id
                or capsule.action_type != context.action_type
                or capsule.action_scope != context.action_scope
                or capsule.capsule_hash != context.capsule_hash
                or covenant_execution.stable_hash() != context.covenant_execution_hash
                or boundary_audit_report.report_hash != context.boundary_audit_report_hash):
            violations.add("firewall_context_mismatch")
        recalculated = evaluate_policy_execution_firewall(
            capsule, covenant_execution, boundary_audit_report, tuple(firewall_rules),
            trusted_boundary_rules=trusted_boundary_rules,
        )
        if (recalculated.decision_value != "allow"
                or recalculated != firewall_decision
                or validate_firewall_decision(recalculated)
                or recalculated.execution_receipt.policy_rules_hash
                    != context.firewall_policy_rules_hash):
            violations.add("untrusted_firewall_decision")
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError):
        violations.add("untrusted_firewall_decision")

    ordered = tuple(sorted(violations))
    return ComposedTrustDecision("DENY" if ordered else "ALLOW", ordered)
