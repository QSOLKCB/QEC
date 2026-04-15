"""v137.18.4 — Bounded Autonomy Ledger.

Deterministic persistence and replay-certification layer for longitudinal
bounded-autonomy decisions across firewall evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

LEDGER_VERSION = "v137.18.4"
GENESIS_CONTINUITY_HASH = "0" * 64
VALID_DECISIONS = ("allow", "deny")


EntryLike = Union["AutonomyLedgerEntry", Mapping[str, Any]]
LedgerLike = Union["BoundedAutonomyLedger", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_non_empty(value: Any) -> str:
    return str(value).strip()


def _normalize_hash(value: Any) -> str:
    text = str(value).strip().lower()
    if len(text) != 64:
        raise ValueError("hash must be 64-char sha256 hex")
    int(text, 16)
    return text


def _normalize_decision(value: Any) -> str:
    decision = str(value).strip().lower()
    if decision not in VALID_DECISIONS:
        raise ValueError("decision must be allow or deny")
    return decision


def _normalize_reasons(reasons: Sequence[Any]) -> Tuple[str, ...]:
    normed = [_normalize_non_empty(item) for item in reasons]
    return tuple(sorted({n for n in normed if n}))


@dataclass(frozen=True)
class AutonomyLedgerEntry:
    entry_index: int
    replay_identity: str
    transition_id: str
    boundary_report_hash: str
    firewall_decision_hash: str
    allow: bool
    decision: str
    rule_hit_reasons: Tuple[str, ...]
    prior_continuity_hash: str
    continuity_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_index": self.entry_index,
            "replay_identity": self.replay_identity,
            "transition_id": self.transition_id,
            "boundary_report_hash": self.boundary_report_hash,
            "firewall_decision_hash": self.firewall_decision_hash,
            "allow": self.allow,
            "decision": self.decision,
            "rule_hit_reasons": list(self.rule_hit_reasons),
            "prior_continuity_hash": self.prior_continuity_hash,
            "continuity_hash": self.continuity_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.continuity_hash


@dataclass(frozen=True)
class AutonomyLedgerSnapshot:
    ledger_version: str
    entry_count: int
    continuity_chain: Tuple[str, ...]
    receipt_chain: Tuple[str, ...]
    ledger_hash: str
    snapshot_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "entry_count": self.entry_count,
            "continuity_chain": list(self.continuity_chain),
            "receipt_chain": list(self.receipt_chain),
            "ledger_hash": self.ledger_hash,
            "snapshot_hash": self.snapshot_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.snapshot_hash


@dataclass(frozen=True)
class AutonomyLedgerReceipt:
    receipt_id: str
    ledger_version: str
    entry_index: int
    replay_identity: str
    transition_id: str
    prior_receipt_hash: str
    receipt_hash: str
    snapshot_hash: str
    ledger_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "ledger_version": self.ledger_version,
            "entry_index": self.entry_index,
            "replay_identity": self.replay_identity,
            "transition_id": self.transition_id,
            "prior_receipt_hash": self.prior_receipt_hash,
            "receipt_hash": self.receipt_hash,
            "snapshot_hash": self.snapshot_hash,
            "ledger_hash": self.ledger_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class BoundedAutonomyLedger:
    ledger_version: str
    entries: Tuple[AutonomyLedgerEntry, ...]
    receipt_chain: Tuple[str, ...]
    ledger_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "receipt_chain": list(self.receipt_chain),
            "ledger_hash": self.ledger_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.ledger_hash


def _entry_unsigned_payload(
    entry_index: int,
    replay_identity: str,
    transition_id: str,
    boundary_report_hash: str,
    firewall_decision_hash: str,
    decision: str,
    rule_hit_reasons: Tuple[str, ...],
    prior_continuity_hash: str,
) -> Dict[str, Any]:
    return {
        "entry_index": entry_index,
        "replay_identity": replay_identity,
        "transition_id": transition_id,
        "boundary_report_hash": boundary_report_hash,
        "firewall_decision_hash": firewall_decision_hash,
        "decision": decision,
        "rule_hit_reasons": list(rule_hit_reasons),
        "prior_continuity_hash": prior_continuity_hash,
    }


def _compute_entry_continuity_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_hex(_canonical_json(dict(payload)).encode("utf-8"))


def _compute_ledger_hash(entries: Tuple[AutonomyLedgerEntry, ...], receipt_chain: Tuple[str, ...], ledger_version: str) -> str:
    body = {
        "ledger_version": ledger_version,
        "entries": [entry.to_dict() for entry in entries],
        "receipt_chain": list(receipt_chain),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def _compute_snapshot_hash(entry_hashes: Tuple[str, ...], receipt_chain: Tuple[str, ...], ledger_hash: str, ledger_version: str) -> str:
    payload = {
        "ledger_version": ledger_version,
        "entry_hashes": list(entry_hashes),
        "receipt_chain": list(receipt_chain),
        "ledger_hash": ledger_hash,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _to_entry(raw: EntryLike) -> AutonomyLedgerEntry:
    if isinstance(raw, AutonomyLedgerEntry):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("entry must be mapping or AutonomyLedgerEntry")
    replay_identity = _normalize_non_empty(raw.get("replay_identity", ""))
    transition_id = _normalize_non_empty(raw.get("transition_id", ""))
    reasons_raw = raw.get("rule_hit_reasons", ())
    if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
        raise ValueError("rule_hit_reasons must be a sequence")
    decision = _normalize_decision(raw.get("decision", ""))
    expected_allow = decision == "allow"
    raw_allow = raw.get("allow")
    if raw_allow is not None:
        if not isinstance(raw_allow, bool):
            raise ValueError(f"allow must be bool, got {type(raw_allow).__name__!r}")
        if raw_allow != expected_allow:
            raise ValueError(f"allow={raw_allow!r} contradicts decision={decision!r}")
    return AutonomyLedgerEntry(
        entry_index=int(raw.get("entry_index", -1)),
        replay_identity=replay_identity,
        transition_id=transition_id,
        boundary_report_hash=_normalize_hash(raw.get("boundary_report_hash", "")),
        firewall_decision_hash=_normalize_hash(raw.get("firewall_decision_hash", "")),
        allow=expected_allow,
        decision=decision,
        rule_hit_reasons=_normalize_reasons(reasons_raw),
        prior_continuity_hash=_normalize_hash(raw.get("prior_continuity_hash", "")),
        continuity_hash=_normalize_hash(raw.get("continuity_hash", "")),
    )


def _normalize_ledger(ledger: LedgerLike | None) -> BoundedAutonomyLedger:
    if ledger is None:
        return BoundedAutonomyLedger(
            ledger_version=LEDGER_VERSION,
            entries=(),
            receipt_chain=(),
            ledger_hash=_compute_ledger_hash((), (), LEDGER_VERSION),
        )
    if isinstance(ledger, BoundedAutonomyLedger):
        return ledger
    if not isinstance(ledger, Mapping):
        raise ValueError("ledger must be mapping or BoundedAutonomyLedger")
    entries_raw = ledger.get("entries", ())
    receipt_chain_raw = ledger.get("receipt_chain", ())
    if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
        raise ValueError("entries must be a sequence")
    if not isinstance(receipt_chain_raw, Sequence) or isinstance(receipt_chain_raw, (str, bytes)):
        raise ValueError("receipt_chain must be a sequence")
    entries = tuple(_to_entry(item) for item in entries_raw)
    receipt_chain = tuple(_normalize_hash(item) for item in receipt_chain_raw)
    ledger_version = _normalize_non_empty(ledger.get("ledger_version", LEDGER_VERSION)) or LEDGER_VERSION
    ledger_hash = _normalize_hash(ledger.get("ledger_hash", _compute_ledger_hash(entries, receipt_chain, ledger_version)))
    return BoundedAutonomyLedger(
        ledger_version=ledger_version,
        entries=entries,
        receipt_chain=receipt_chain,
        ledger_hash=ledger_hash,
    )


def append_autonomy_ledger_entry(
    decision_t: Mapping[str, Any],
    prior_ledger: LedgerLike | None = None,
) -> Tuple[BoundedAutonomyLedger, AutonomyLedgerReceipt]:
    ledger = _normalize_ledger(prior_ledger)

    replay_identity = _normalize_non_empty(decision_t.get("replay_identity", ""))
    transition_id = _normalize_non_empty(decision_t.get("transition_id", ""))
    boundary_report_hash = _normalize_hash(decision_t.get("boundary_report_hash", ""))
    firewall_decision_hash = _normalize_hash(decision_t.get("firewall_decision_hash", ""))
    decision = _normalize_decision(decision_t.get("decision", ""))
    reasons_raw = decision_t.get("rule_hit_reasons", ())
    if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
        raise ValueError("rule_hit_reasons must be a sequence")
    reasons = _normalize_reasons(reasons_raw)

    entry_index = len(ledger.entries)
    prior_continuity_hash = ledger.entries[-1].continuity_hash if ledger.entries else GENESIS_CONTINUITY_HASH
    unsigned_entry = _entry_unsigned_payload(
        entry_index=entry_index,
        replay_identity=replay_identity,
        transition_id=transition_id,
        boundary_report_hash=boundary_report_hash,
        firewall_decision_hash=firewall_decision_hash,
        decision=decision,
        rule_hit_reasons=reasons,
        prior_continuity_hash=prior_continuity_hash,
    )
    continuity_hash = _compute_entry_continuity_hash(unsigned_entry)
    entry = AutonomyLedgerEntry(
        entry_index=entry_index,
        replay_identity=replay_identity,
        transition_id=transition_id,
        boundary_report_hash=boundary_report_hash,
        firewall_decision_hash=firewall_decision_hash,
        allow=decision == "allow",
        decision=decision,
        rule_hit_reasons=reasons,
        prior_continuity_hash=prior_continuity_hash,
        continuity_hash=continuity_hash,
    )

    next_entries = ledger.entries + (entry,)
    prior_receipt_hash = ledger.receipt_chain[-1] if ledger.receipt_chain else GENESIS_CONTINUITY_HASH

    provisional_ledger_hash = _compute_ledger_hash(next_entries, ledger.receipt_chain, ledger.ledger_version)
    provisional_snapshot_hash = _compute_snapshot_hash(
        tuple(item.continuity_hash for item in next_entries),
        ledger.receipt_chain,
        provisional_ledger_hash,
        ledger.ledger_version,
    )
    receipt_id = f"autonomy-receipt::{entry_index:08d}::{transition_id}"
    receipt_unsigned = {
        "receipt_id": receipt_id,
        "ledger_version": ledger.ledger_version,
        "entry_index": entry_index,
        "replay_identity": replay_identity,
        "transition_id": transition_id,
        "prior_receipt_hash": prior_receipt_hash,
        "snapshot_hash": provisional_snapshot_hash,
    }
    receipt_hash = _sha256_hex(_canonical_json(receipt_unsigned).encode("utf-8"))

    next_receipt_chain = ledger.receipt_chain + (receipt_hash,)
    next_ledger_hash = _compute_ledger_hash(next_entries, next_receipt_chain, ledger.ledger_version)
    snapshot_hash = _compute_snapshot_hash(
        tuple(item.continuity_hash for item in next_entries),
        next_receipt_chain,
        next_ledger_hash,
        ledger.ledger_version,
    )
    receipt = AutonomyLedgerReceipt(
        receipt_id=receipt_id,
        ledger_version=ledger.ledger_version,
        entry_index=entry_index,
        replay_identity=replay_identity,
        transition_id=transition_id,
        prior_receipt_hash=prior_receipt_hash,
        receipt_hash=receipt_hash,
        snapshot_hash=snapshot_hash,
        ledger_hash=next_ledger_hash,
    )
    next_ledger = BoundedAutonomyLedger(
        ledger_version=ledger.ledger_version,
        entries=next_entries,
        receipt_chain=next_receipt_chain,
        ledger_hash=next_ledger_hash,
    )
    return next_ledger, receipt


def build_autonomy_ledger_receipt(ledger: LedgerLike) -> AutonomyLedgerReceipt:
    normalized = _normalize_ledger(ledger)
    if not normalized.entries:
        raise ValueError("cannot build receipt for empty ledger")
    last = normalized.entries[-1]
    prior_receipt_hash = normalized.receipt_chain[-2] if len(normalized.receipt_chain) > 1 else GENESIS_CONTINUITY_HASH
    prefix_entries = normalized.entries[: last.entry_index + 1]
    prefix_chain_without_last = normalized.receipt_chain[:-1]
    provisional_ledger_hash = _compute_ledger_hash(prefix_entries, prefix_chain_without_last, normalized.ledger_version)
    provisional_snapshot_hash = _compute_snapshot_hash(
        tuple(entry.continuity_hash for entry in prefix_entries),
        prefix_chain_without_last,
        provisional_ledger_hash,
        normalized.ledger_version,
    )
    snapshot = AutonomyLedgerSnapshot(
        ledger_version=normalized.ledger_version,
        entry_count=len(normalized.entries),
        continuity_chain=tuple(entry.continuity_hash for entry in normalized.entries),
        receipt_chain=normalized.receipt_chain,
        ledger_hash=normalized.ledger_hash,
        snapshot_hash=_compute_snapshot_hash(
            tuple(entry.continuity_hash for entry in normalized.entries),
            normalized.receipt_chain,
            normalized.ledger_hash,
            normalized.ledger_version,
        ),
    )
    receipt_id = f"autonomy-receipt::{last.entry_index:08d}::{last.transition_id}"
    unsigned = {
        "receipt_id": receipt_id,
        "ledger_version": normalized.ledger_version,
        "entry_index": last.entry_index,
        "replay_identity": last.replay_identity,
        "transition_id": last.transition_id,
        "prior_receipt_hash": prior_receipt_hash,
        "snapshot_hash": provisional_snapshot_hash,
    }
    receipt_hash = _sha256_hex(_canonical_json(unsigned).encode("utf-8"))
    return AutonomyLedgerReceipt(
        receipt_id=receipt_id,
        ledger_version=normalized.ledger_version,
        entry_index=last.entry_index,
        replay_identity=last.replay_identity,
        transition_id=last.transition_id,
        prior_receipt_hash=prior_receipt_hash,
        receipt_hash=receipt_hash,
        snapshot_hash=snapshot.snapshot_hash,
        ledger_hash=normalized.ledger_hash,
    )


def validate_bounded_autonomy_ledger(ledger: LedgerLike) -> Dict[str, Any]:
    violations = []
    normalized: BoundedAutonomyLedger | None = None
    try:
        normalized = _normalize_ledger(ledger)
    except Exception as exc:  # validator must never raise
        violations.append(f"malformed ledger input: {exc}")
        return {"is_valid": False, "violations": tuple(sorted(violations)), "ledger_hash": None}

    expected_hash = _compute_ledger_hash(normalized.entries, normalized.receipt_chain, normalized.ledger_version)
    if normalized.ledger_hash != expected_hash:
        violations.append("ledger hash drift")

    if normalized.entries and len(normalized.receipt_chain) != len(normalized.entries):
        violations.append("missing receipt chain")

    seen_transition_ids = set()
    seen_decision_hashes = set()
    prior_hash = GENESIS_CONTINUITY_HASH
    for index, entry in enumerate(normalized.entries):
        if entry.entry_index != index:
            violations.append("malformed entry ordering")
        if entry.prior_continuity_hash != prior_hash:
            violations.append("continuity break")
        unsigned = _entry_unsigned_payload(
            entry_index=entry.entry_index,
            replay_identity=entry.replay_identity,
            transition_id=entry.transition_id,
            boundary_report_hash=entry.boundary_report_hash,
            firewall_decision_hash=entry.firewall_decision_hash,
            decision=entry.decision,
            rule_hit_reasons=entry.rule_hit_reasons,
            prior_continuity_hash=entry.prior_continuity_hash,
        )
        if _compute_entry_continuity_hash(unsigned) != entry.continuity_hash:
            violations.append("continuity break")
        if entry.transition_id in seen_transition_ids:
            violations.append("duplicate transition_id")
        seen_transition_ids.add(entry.transition_id)
        if entry.firewall_decision_hash in seen_decision_hashes:
            violations.append("duplicate decision hash")
        seen_decision_hashes.add(entry.firewall_decision_hash)
        if entry.allow != (entry.decision == "allow"):
            violations.append("allow/decision inconsistency")
        prior_hash = entry.continuity_hash

    if normalized.receipt_chain:
        for index, receipt_hash in enumerate(normalized.receipt_chain):
            if index >= len(normalized.entries):
                violations.append("missing receipt chain")
                break
            entry = normalized.entries[index]
            prior_receipt_hash = normalized.receipt_chain[index - 1] if index > 0 else GENESIS_CONTINUITY_HASH
            prefix_chain = normalized.receipt_chain[:index]
            interim_hash = _compute_ledger_hash(normalized.entries[: index + 1], prefix_chain, normalized.ledger_version)
            snap = _compute_snapshot_hash(
                tuple(item.continuity_hash for item in normalized.entries[: index + 1]),
                prefix_chain,
                interim_hash,
                normalized.ledger_version,
            )
            expected = _sha256_hex(
                _canonical_json(
                    {
                        "receipt_id": f"autonomy-receipt::{index:08d}::{entry.transition_id}",
                        "ledger_version": normalized.ledger_version,
                        "entry_index": index,
                        "replay_identity": entry.replay_identity,
                        "transition_id": entry.transition_id,
                        "prior_receipt_hash": prior_receipt_hash,
                        "snapshot_hash": snap,
                    }
                ).encode("utf-8")
            )
            if expected != receipt_hash:
                violations.append("missing receipt chain")
                break

    return {
        "is_valid": not violations,
        "violations": tuple(sorted(set(violations))),
        "ledger_hash": normalized.ledger_hash,
    }


def compare_autonomy_ledger_replay(run_a_ledger: LedgerLike, run_b_ledger: LedgerLike) -> Dict[str, Any]:
    report_a = validate_bounded_autonomy_ledger(run_a_ledger)
    report_b = validate_bounded_autonomy_ledger(run_b_ledger)

    _empty = BoundedAutonomyLedger(LEDGER_VERSION, (), (), _compute_ledger_hash((), (), LEDGER_VERSION))
    a_failed = False
    b_failed = False

    try:
        ledger_a = _normalize_ledger(run_a_ledger)
    except Exception:
        ledger_a = _empty
        a_failed = True

    try:
        ledger_b = _normalize_ledger(run_b_ledger)
    except Exception:
        ledger_b = _empty
        b_failed = True

    if a_failed or b_failed:
        return {
            "replay_stable": False,
            "ledger_hash_match": False,
            "receipt_chain_match": False,
            "replay_identity_match": False,
            "violations": tuple(sorted(set(report_a["violations"] + report_b["violations"]))),
        }

    replay_identity_match = tuple(entry.replay_identity for entry in ledger_a.entries) == tuple(
        entry.replay_identity for entry in ledger_b.entries
    )
    transition_match = tuple(entry.transition_id for entry in ledger_a.entries) == tuple(entry.transition_id for entry in ledger_b.entries)
    receipt_chain_match = ledger_a.receipt_chain == ledger_b.receipt_chain
    ledger_hash_match = ledger_a.ledger_hash == ledger_b.ledger_hash

    violations = []
    if not replay_identity_match:
        violations.append("replay identity mismatch")
    if not receipt_chain_match:
        violations.append("missing receipt chain")
    if not transition_match:
        violations.append("malformed entry ordering")
    if not ledger_hash_match:
        violations.append("ledger hash drift")

    return {
        "replay_stable": report_a["is_valid"] and report_b["is_valid"] and not violations,
        "ledger_hash_match": ledger_hash_match,
        "receipt_chain_match": receipt_chain_match,
        "replay_identity_match": replay_identity_match,
        "violations": tuple(sorted(set(report_a["violations"] + report_b["violations"] + tuple(violations)))),
    }


def summarize_autonomy_ledger(ledger: LedgerLike) -> Dict[str, Any]:
    normalized = _normalize_ledger(ledger)
    decision_counts = {"allow": 0, "deny": 0}
    reason_counts: Dict[str, int] = {}
    replay_identities = []
    for entry in normalized.entries:
        decision_counts[entry.decision] += 1
        replay_identities.append(entry.replay_identity)
        for reason in entry.rule_hit_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    ordered_reasons = tuple(sorted(reason_counts.items(), key=lambda item: (item[0], item[1])))
    ordered_replay_identities = tuple(sorted(set(replay_identities)))
    return {
        "ledger_version": normalized.ledger_version,
        "entry_count": len(normalized.entries),
        "ledger_hash": normalized.ledger_hash,
        "decision_counts": decision_counts,
        "replay_identities": ordered_replay_identities,
        "rule_hit_reason_counts": ordered_reasons,
        "continuity_chain": tuple(entry.continuity_hash for entry in normalized.entries),
        "receipt_chain": normalized.receipt_chain,
    }


__all__ = [
    "AutonomyLedgerEntry",
    "AutonomyLedgerSnapshot",
    "AutonomyLedgerReceipt",
    "BoundedAutonomyLedger",
    "append_autonomy_ledger_entry",
    "validate_bounded_autonomy_ledger",
    "build_autonomy_ledger_receipt",
    "compare_autonomy_ledger_replay",
    "summarize_autonomy_ledger",
]
