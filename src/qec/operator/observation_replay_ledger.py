"""Phase 622: one-use, revocable, in-memory IRC observation admission.

This ledger belongs to the server operator, NEVER to IRC client input. It
serializes the preflight, revocation checks and consumption of a precommitted
operation ID before the Phase 621 fixed outbound NOTICE is constructed.

The receipt chain is local diagnostic evidence, not a signed historical event.
A new process without the same ledger cannot inherit replay protection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any

from .trusted_observation import (
    TrustedObservationAuthority, authorize_trusted_observation,
)

_SCHEMA = "qec-irc-observation-gate-v1"
_GENESIS = "0" * 64
_MAX_RECEIPTS = 256


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ObservationGateReceipt:
    sequence: int
    verdict: str
    reason: str
    operation_digest: str
    prior_receipt_hash: str
    receipt_hash: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "sequence": self.sequence,
            "verdict": self.verdict,
            "reason": self.reason,
            "operation_digest": self.operation_digest,
            "prior_receipt_hash": self.prior_receipt_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "receipt_hash": self.receipt_hash}


def verify_observation_gate_chain(receipts: tuple[ObservationGateReceipt, ...]) -> bool:
    """Check content hashes, contiguous order and exact predecessor binding."""
    prior = _GENESIS
    for sequence, receipt in enumerate(receipts, start=1):
        if (not isinstance(receipt, ObservationGateReceipt)
                or receipt.sequence != sequence
                or receipt.prior_receipt_hash != prior
                or receipt.verdict not in ("ALLOW", "DENY")
                or not isinstance(receipt.reason, str)
                or len(receipt.operation_digest) != 64):
            return False
        if receipt.receipt_hash != hashlib.sha256(
            _canonical_bytes(receipt.payload())
        ).hexdigest():
            return False
        prior = receipt.receipt_hash
    return True


def _require_identifier(value: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("revocation identity must be a non-empty exact string")
    return value


class ObservationReplayLedger:
    """One server-lifetime admission ledger; no implicit expiry or eviction.

    A full ledger denies *all* further protected actions instead of silently
    evicting old operation IDs and accidentally permitting a replay. The only
    mutation on DENY is a bounded diagnostic receipt (or saturated flag).
    """

    def __init__(self, *, max_receipts: int = _MAX_RECEIPTS) -> None:
        if type(max_receipts) is not int or not (1 <= max_receipts <= _MAX_RECEIPTS):
            raise ValueError("invalid observation receipt capacity")
        self._lock = Lock()
        self._limit = max_receipts
        self._used_operations: set[str] = set()
        self._revoked_operations: set[str] = set()
        self._revoked_signers: set[str] = set()
        self._revoked_capabilities: set[str] = set()
        self._revoked_policy_hashes: set[str] = set()
        self._receipts: list[ObservationGateReceipt] = []
        self._saturated = False

    def revoke_operation(self, operation_id: str) -> None:
        with self._lock:
            self._revoked_operations.add(_require_identifier(operation_id))

    def revoke_signer(self, signer_id: str) -> None:
        with self._lock:
            self._revoked_signers.add(_require_identifier(signer_id))

    def revoke_capability(self, capability_hash: str) -> None:
        with self._lock:
            self._revoked_capabilities.add(_require_identifier(capability_hash))

    def revoke_policy(self, policy_rules_hash: str) -> None:
        with self._lock:
            self._revoked_policy_hashes.add(_require_identifier(policy_rules_hash))

    @property
    def receipts(self) -> tuple[ObservationGateReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    @property
    def saturated(self) -> bool:
        with self._lock:
            return self._saturated

    def _record(self, verdict: str, reason: str, operation_id: str) -> None:
        # Called only with the ledger lock. No raw identity or proof is retained.
        operation_digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        previous = self._receipts[-1].receipt_hash if self._receipts else _GENESIS
        item = ObservationGateReceipt(
            sequence=len(self._receipts) + 1,
            verdict=verdict,
            reason=reason,
            operation_digest=operation_digest,
            prior_receipt_hash=previous,
            receipt_hash="",
        )
        item = ObservationGateReceipt(
            item.sequence, item.verdict, item.reason, item.operation_digest,
            item.prior_receipt_hash,
            hashlib.sha256(_canonical_bytes(item.payload())).hexdigest(),
        )
        self._receipts.append(item)

    def admit(
        self, authority: TrustedObservationAuthority | None, client_id: str,
    ) -> bool:
        """Atomically verify/consume one operation, then permit fixed output.

        No await/transport activity occurs while the lock is held. A denied
        attempt may append one diagnostic receipt but never marks an operation
        consumed. An accepted operation is spent before outbound bytes exist.
        """
        with self._lock:
            if len(self._receipts) >= self._limit:
                self._saturated = True
                return False

            operation = ""
            reason = "INVALID_AUTHORITY"
            try:
                if isinstance(authority, TrustedObservationAuthority):
                    ctx = authority.context
                    operation = _require_identifier(ctx.operation_id)
                    if authority.client_id != client_id:
                        reason = "WRONG_CLIENT"
                    elif operation in self._revoked_operations:
                        reason = "REVOKED_OPERATION"
                    elif operation in self._used_operations:
                        reason = "REPLAYED_OPERATION"
                    elif ctx.trusted_signer_id in self._revoked_signers:
                        reason = "REVOKED_SIGNER"
                    elif ctx.capability_hash in self._revoked_capabilities:
                        reason = "REVOKED_CAPABILITY"
                    elif ctx.firewall_policy_rules_hash in self._revoked_policy_hashes:
                        reason = "REVOKED_POLICY"
                    elif not authorize_trusted_observation(authority, client_id):
                        reason = "COMPOSITION_DENIED"
                    else:
                        self._used_operations.add(operation)
                        self._record("ALLOW", "COMPOSED_TRUST_ACCEPTED", operation)
                        return True
            except Exception:
                # A malformed context or failing evidence must never pass.
                reason = "INVALID_AUTHORITY"

            self._record("DENY", reason, operation)
            return False
