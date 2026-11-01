"""Phase 621: mandatory composition preflight for one IRC observation action.

The operator-supplied authority is never constructed from IRC text, nicknames,
SOFA posts, persisted producer flags or other untrusted client-supplied data.

This only gates a *new* read-only NOTICE action. It does not claim to protect
other IRC commands, the internal covenant engine, or a general action runner.
There is no persistent authorization or anti-replay guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from qec.orchestration.evidence_trust_composition import (
    TrustedOperationContext, evaluate_composed_evidence_trust,
)
from qec.orchestration.governance_memory_io_boundary_auditor import BoundaryAuditRule


@dataclass(frozen=True)
class TrustedObservationAuthority:
    """Created out of band by the *server operator* before any IRC request.

    client_id is a local transport identifier, not an authenticated human ID.
    A production integration must bind that identifier to an actual authenticated
    operator. A frozen container does not deep-freeze its nested mappings; the
    composed gate revalidates the candidate on every invocation.
    """
    client_id: str
    context: TrustedOperationContext
    candidate: Mapping[str, Any]
    trusted_signers: Mapping[str, bytes]
    trusted_grant_hashes: Collection[str]
    trusted_boundary_rules: Sequence[BoundaryAuditRule] | None = None


_REQUIRED_CANDIDATE_KEYS = frozenset((
    "experiment", "grant", "privilege_decision", "privilege_receipt",
    "capsule", "covenant_execution", "boundary_audit_report",
    "firewall_rules", "firewall_decision",
))


def authorize_trusted_observation(
    authority: TrustedObservationAuthority | None, client_id: str,
) -> bool:
    """Side-effect-free, fail-closed preflight before the outbound NOTICE.

    Caller identity is an internal session ID fixed by the server, never the
    untrusted IRC nick. Unavailable/malformed/stale evidence means no action.
    Crucially, the candidate cannot supply or override independent trust roots.
    """
    if (not isinstance(authority, TrustedObservationAuthority)
            or not isinstance(client_id, str) or not client_id
            or authority.client_id != client_id
            or not isinstance(authority.context, TrustedOperationContext)
            or authority.context.action_type != "observe"
            or authority.context.action_scope != "observe"):
        return False
    try:
        candidate = authority.candidate
        if (not isinstance(candidate, Mapping)
                or frozenset(candidate) != _REQUIRED_CANDIDATE_KEYS):
            return False
        decision = evaluate_composed_evidence_trust(
            authority.context,
            experiment=candidate["experiment"],
            grant=candidate["grant"],
            privilege_decision=candidate["privilege_decision"],
            privilege_receipt=candidate["privilege_receipt"],
            capsule=candidate["capsule"],
            covenant_execution=candidate["covenant_execution"],
            boundary_audit_report=candidate["boundary_audit_report"],
            firewall_rules=candidate["firewall_rules"],
            firewall_decision=candidate["firewall_decision"],
            trusted_signers=authority.trusted_signers,
            trusted_grant_hashes=authority.trusted_grant_hashes,
            trusted_boundary_rules=authority.trusted_boundary_rules,
        )
        return decision.allowed is True
    except Exception:
        # Even a malformed or hostile mapping/receipt must not allow a NOTICE.
        # Do not trap process-level BaseException (interrupts/termination).
        return False
