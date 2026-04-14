"""v137.17.6 — Deterministic Ledger Replay Certification Pack.

Deterministic certification infrastructure for replay-safe verification of
DataflowResearchLedger artifacts from v137.17.5.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from qec.orchestration.dataflow_research_ledger_kernel import (
    CANONICAL_DATAFLOW_STAGES,
    DataflowLedgerTraversalReceipt,
    DataflowResearchLedger,
    LedgerLike,
    VALID_DATAFLOW_TRAVERSAL_MODES,
    normalize_dataflow_research_ledger,
    traverse_dataflow_research_ledger,
    validate_dataflow_research_ledger,
)


ReceiptLike = Union[DataflowLedgerTraversalReceipt, Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ledger_to_mapping(ledger: LedgerLike) -> Mapping[str, Any]:
    if isinstance(ledger, DataflowResearchLedger):
        return ledger.to_dict()
    if isinstance(ledger, Mapping):
        return dict(ledger)
    raise ValueError("ledger must be mapping or DataflowResearchLedger")


def _require_mode(mode: str) -> str:
    normalized = str(mode).strip()
    if normalized not in VALID_DATAFLOW_TRAVERSAL_MODES:
        raise ValueError(f"invalid traversal mode: {normalized}")
    return normalized


def _normalize_modes(modes: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    for raw in modes:
        seen.add(_require_mode(raw))
    if not seen:
        raise ValueError("at least one traversal mode is required")
    return tuple(mode for mode in VALID_DATAFLOW_TRAVERSAL_MODES if mode in seen)


def _normalize_receipt(raw: ReceiptLike) -> DataflowLedgerTraversalReceipt:
    if isinstance(raw, DataflowLedgerTraversalReceipt):
        parsed = raw
    elif isinstance(raw, Mapping):
        parsed = DataflowLedgerTraversalReceipt(
            receipt_id=str(raw.get("receipt_id", "")).strip(),
            ledger_id=str(raw.get("ledger_id", "")).strip(),
            ledger_hash=str(raw.get("ledger_hash", "")).strip(),
            traversal_mode=_require_mode(str(raw.get("traversal_mode", ""))),
            ordered_stage_trace=tuple(str(v).strip() for v in raw.get("ordered_stage_trace", ())),
            ordered_edge_trace=tuple(str(v).strip() for v in raw.get("ordered_edge_trace", ())),
            traversal_hash=str(raw.get("traversal_hash", "")).strip(),
        )
    else:
        raise ValueError("receipt must be mapping or DataflowLedgerTraversalReceipt")

    if not parsed.receipt_id:
        raise ValueError("receipt_id must be non-empty")
    return parsed


def _normalize_receipt_overrides(overrides: Mapping[str, ReceiptLike] | None) -> Dict[str, DataflowLedgerTraversalReceipt]:
    normalized: Dict[str, DataflowLedgerTraversalReceipt] = {}
    if overrides is None:
        return normalized
    for mode, receipt in sorted(overrides.items(), key=lambda item: str(item[0])):
        normalized_mode = _require_mode(mode)
        parsed = _normalize_receipt(receipt)
        if parsed.traversal_mode != normalized_mode:
            raise ValueError("receipt traversal_mode mismatch")
        normalized[normalized_mode] = parsed
    return normalized


def _continuity_summary_hash(ledger: LedgerLike) -> str:
    mapping = _ledger_to_mapping(ledger)
    summary = mapping.get("continuity_summary", {})
    return _sha256_hex(_canonical_json(summary).encode("utf-8"))


def _is_canonical_stage_order(trace: Tuple[str, ...]) -> bool:
    stages = tuple(item.split(":", 2)[1] for item in trace)
    expected = tuple(stage for stage in CANONICAL_DATAFLOW_STAGES if stage in stages)
    return stages == expected


def _validate_ledger_or_raise(ledger: LedgerLike) -> None:
    try:
        report = validate_dataflow_research_ledger(ledger)
    except ValueError as exc:
        raise ValueError(f"malformed ledger input: {exc}") from exc
    if not report.is_valid:
        raise ValueError(f"malformed ledger input: {','.join(report.violations)}")


@dataclass(frozen=True)
class LedgerReplayCertificationEntry:
    traversal_mode: str
    canonical_bytes_match: bool
    ledger_hash_match: bool
    traversal_hash_match: bool
    receipt_id_match: bool
    continuity_summary_match: bool
    traversal_order_match: bool
    canonical_stage_order_match: bool
    entry_count_match: bool
    replay_stable: bool
    certification_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traversal_mode": self.traversal_mode,
            "canonical_bytes_match": self.canonical_bytes_match,
            "ledger_hash_match": self.ledger_hash_match,
            "traversal_hash_match": self.traversal_hash_match,
            "receipt_id_match": self.receipt_id_match,
            "continuity_summary_match": self.continuity_summary_match,
            "traversal_order_match": self.traversal_order_match,
            "canonical_stage_order_match": self.canonical_stage_order_match,
            "entry_count_match": self.entry_count_match,
            "replay_stable": self.replay_stable,
            "certification_hash": self.certification_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.certification_hash


@dataclass(frozen=True)
class LedgerReplayCertificationReport:
    ledger_id: str
    baseline_ledger_hash: str
    replay_ledger_hash: str
    canonical_bytes_hash_a: str
    canonical_bytes_hash_b: str
    continuity_summary_hash_a: str
    continuity_summary_hash_b: str
    traversal_modes: Tuple[str, ...]
    entries: Tuple[LedgerReplayCertificationEntry, ...]
    replay_stable: bool
    certification_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "baseline_ledger_hash": self.baseline_ledger_hash,
            "replay_ledger_hash": self.replay_ledger_hash,
            "canonical_bytes_hash_a": self.canonical_bytes_hash_a,
            "canonical_bytes_hash_b": self.canonical_bytes_hash_b,
            "continuity_summary_hash_a": self.continuity_summary_hash_a,
            "continuity_summary_hash_b": self.continuity_summary_hash_b,
            "traversal_modes": list(self.traversal_modes),
            "entries": [entry.to_dict() for entry in self.entries],
            "replay_stable": self.replay_stable,
            "certification_hash": self.certification_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.certification_hash


@dataclass(frozen=True)
class LedgerReplayCertificationReceipt:
    receipt_id: str
    ledger_id: str
    certification_hash: str
    replay_stable: bool
    traversal_modes: Tuple[str, ...]
    report_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "ledger_id": self.ledger_id,
            "certification_hash": self.certification_hash,
            "replay_stable": self.replay_stable,
            "traversal_modes": list(self.traversal_modes),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class LedgerReplayCertificationPack:
    report: LedgerReplayCertificationReport
    receipt: LedgerReplayCertificationReceipt
    traversal_receipts_a: Tuple[DataflowLedgerTraversalReceipt, ...]
    traversal_receipts_b: Tuple[DataflowLedgerTraversalReceipt, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "receipt": self.receipt.to_dict(),
            "traversal_receipts_a": [item.to_dict() for item in self.traversal_receipts_a],
            "traversal_receipts_b": [item.to_dict() for item in self.traversal_receipts_b],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


def certify_traversal_replay(
    run_a_ledger: LedgerLike,
    run_b_ledger: LedgerLike,
    traversal_mode: str,
    *,
    receipt_a: ReceiptLike | None = None,
    receipt_b: ReceiptLike | None = None,
) -> LedgerReplayCertificationEntry:
    mode = _require_mode(traversal_mode)
    _validate_ledger_or_raise(run_a_ledger)
    _validate_ledger_or_raise(run_b_ledger)

    baseline_receipt = traverse_dataflow_research_ledger(run_a_ledger, mode)
    replay_receipt = traverse_dataflow_research_ledger(run_b_ledger, mode)
    provided_a = baseline_receipt if receipt_a is None else _normalize_receipt(receipt_a)
    provided_b = replay_receipt if receipt_b is None else _normalize_receipt(receipt_b)

    canonical_a = _canonical_json(_ledger_to_mapping(run_a_ledger))
    canonical_b = _canonical_json(_ledger_to_mapping(run_b_ledger))

    canonical_bytes_match = canonical_a == canonical_b
    ledger_hash_match = provided_a.ledger_hash == provided_b.ledger_hash == baseline_receipt.ledger_hash == replay_receipt.ledger_hash
    traversal_hash_match = provided_a.traversal_hash == provided_b.traversal_hash == baseline_receipt.traversal_hash == replay_receipt.traversal_hash
    receipt_id_match = provided_a.receipt_id == provided_b.receipt_id == baseline_receipt.receipt_id == replay_receipt.receipt_id
    continuity_summary_match = _continuity_summary_hash(run_a_ledger) == _continuity_summary_hash(run_b_ledger)
    traversal_order_match = (
        provided_a.ordered_stage_trace == provided_b.ordered_stage_trace == baseline_receipt.ordered_stage_trace == replay_receipt.ordered_stage_trace
        and provided_a.ordered_edge_trace == provided_b.ordered_edge_trace == baseline_receipt.ordered_edge_trace == replay_receipt.ordered_edge_trace
    )
    canonical_stage_order_match = _is_canonical_stage_order(baseline_receipt.ordered_stage_trace) and _is_canonical_stage_order(replay_receipt.ordered_stage_trace)
    entry_count_match = len(provided_a.ordered_stage_trace) == len(provided_b.ordered_stage_trace) == len(baseline_receipt.ordered_stage_trace) == len(replay_receipt.ordered_stage_trace)

    replay_stable = all(
        (
            canonical_bytes_match,
            ledger_hash_match,
            traversal_hash_match,
            receipt_id_match,
            continuity_summary_match,
            traversal_order_match,
            canonical_stage_order_match,
            entry_count_match,
        )
    )

    payload = {
        "traversal_mode": mode,
        "canonical_bytes_match": canonical_bytes_match,
        "ledger_hash_match": ledger_hash_match,
        "traversal_hash_match": traversal_hash_match,
        "receipt_id_match": receipt_id_match,
        "continuity_summary_match": continuity_summary_match,
        "traversal_order_match": traversal_order_match,
        "canonical_stage_order_match": canonical_stage_order_match,
        "entry_count_match": entry_count_match,
        "replay_stable": replay_stable,
    }
    certification_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))

    return LedgerReplayCertificationEntry(
        traversal_mode=mode,
        canonical_bytes_match=canonical_bytes_match,
        ledger_hash_match=ledger_hash_match,
        traversal_hash_match=traversal_hash_match,
        receipt_id_match=receipt_id_match,
        continuity_summary_match=continuity_summary_match,
        traversal_order_match=traversal_order_match,
        canonical_stage_order_match=canonical_stage_order_match,
        entry_count_match=entry_count_match,
        replay_stable=replay_stable,
        certification_hash=certification_hash,
    )


def compare_replay_runs(
    run_a_ledger: LedgerLike,
    run_b_ledger: LedgerLike,
    *,
    traversal_modes: Sequence[str] = VALID_DATAFLOW_TRAVERSAL_MODES,
    run_a_receipts: Mapping[str, ReceiptLike] | None = None,
    run_b_receipts: Mapping[str, ReceiptLike] | None = None,
) -> LedgerReplayCertificationReport:
    _validate_ledger_or_raise(run_a_ledger)
    _validate_ledger_or_raise(run_b_ledger)
    modes = _normalize_modes(traversal_modes)
    normalized_a = _normalize_receipt_overrides(run_a_receipts)
    normalized_b = _normalize_receipt_overrides(run_b_receipts)

    entries = tuple(
        certify_traversal_replay(
            run_a_ledger,
            run_b_ledger,
            mode,
            receipt_a=normalized_a.get(mode),
            receipt_b=normalized_b.get(mode),
        )
        for mode in modes
    )
    replay_stable = all(entry.replay_stable for entry in entries)

    mapping_a = _ledger_to_mapping(run_a_ledger)
    mapping_b = _ledger_to_mapping(run_b_ledger)
    payload = {
        "ledger_id": str(mapping_a.get("ledger_id", "")).strip(),
        "baseline_ledger_hash": str(mapping_a.get("ledger_hash", "")).strip(),
        "replay_ledger_hash": str(mapping_b.get("ledger_hash", "")).strip(),
        "canonical_bytes_hash_a": _sha256_hex(_canonical_json(mapping_a).encode("utf-8")),
        "canonical_bytes_hash_b": _sha256_hex(_canonical_json(mapping_b).encode("utf-8")),
        "continuity_summary_hash_a": _continuity_summary_hash(mapping_a),
        "continuity_summary_hash_b": _continuity_summary_hash(mapping_b),
        "traversal_modes": list(modes),
        "entries": [entry.to_dict() for entry in entries],
        "replay_stable": replay_stable,
    }
    certification_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))

    return LedgerReplayCertificationReport(
        ledger_id=payload["ledger_id"],
        baseline_ledger_hash=payload["baseline_ledger_hash"],
        replay_ledger_hash=payload["replay_ledger_hash"],
        canonical_bytes_hash_a=payload["canonical_bytes_hash_a"],
        canonical_bytes_hash_b=payload["canonical_bytes_hash_b"],
        continuity_summary_hash_a=payload["continuity_summary_hash_a"],
        continuity_summary_hash_b=payload["continuity_summary_hash_b"],
        traversal_modes=modes,
        entries=entries,
        replay_stable=replay_stable,
        certification_hash=certification_hash,
    )


def build_replay_certification_receipt(report: LedgerReplayCertificationReport) -> LedgerReplayCertificationReceipt:
    report_hash = _sha256_hex(report.to_canonical_json().encode("utf-8"))
    return LedgerReplayCertificationReceipt(
        receipt_id=f"ledger-replay-cert::{report_hash[:16]}",
        ledger_id=report.ledger_id,
        certification_hash=report.certification_hash,
        replay_stable=report.replay_stable,
        traversal_modes=report.traversal_modes,
        report_hash=report_hash,
    )


def validate_ledger_replay_certification(pack: LedgerReplayCertificationPack) -> bool:
    report_copy = pack.report.to_dict()
    stored_hash = report_copy.pop("certification_hash")
    recomputed_hash = _sha256_hex(_canonical_json(report_copy).encode("utf-8"))
    if recomputed_hash != stored_hash:
        raise ValueError(
            f"report certification_hash mismatch: stored={stored_hash!r}, recomputed={recomputed_hash!r}"
        )
    expected_receipt = build_replay_certification_receipt(pack.report)
    if pack.receipt.receipt_id != expected_receipt.receipt_id:
        raise ValueError(
            f"receipt receipt_id mismatch: got={pack.receipt.receipt_id!r}, expected={expected_receipt.receipt_id!r}"
        )
    if pack.receipt.ledger_id != expected_receipt.ledger_id:
        raise ValueError(
            f"receipt ledger_id mismatch: got={pack.receipt.ledger_id!r}, expected={expected_receipt.ledger_id!r}"
        )
    if pack.receipt.certification_hash != expected_receipt.certification_hash:
        raise ValueError(
            f"receipt certification_hash mismatch: got={pack.receipt.certification_hash!r}, expected={expected_receipt.certification_hash!r}"
        )
    if pack.receipt.replay_stable != expected_receipt.replay_stable:
        raise ValueError(
            f"receipt replay_stable mismatch: got={pack.receipt.replay_stable!r}, expected={expected_receipt.replay_stable!r}"
        )
    if pack.receipt.traversal_modes != expected_receipt.traversal_modes:
        raise ValueError(
            f"receipt traversal_modes mismatch: got={pack.receipt.traversal_modes!r}, expected={expected_receipt.traversal_modes!r}"
        )
    if pack.receipt.report_hash != expected_receipt.report_hash:
        raise ValueError(
            f"receipt report_hash mismatch: got={pack.receipt.report_hash!r}, expected={expected_receipt.report_hash!r}"
        )
    return True


def certify_ledger_replay(
    ledger: LedgerLike,
    *,
    traversal_modes: Sequence[str] = VALID_DATAFLOW_TRAVERSAL_MODES,
    expected_receipts: Mapping[str, ReceiptLike] | None = None,
) -> LedgerReplayCertificationPack:
    _validate_ledger_or_raise(ledger)

    normalized_ledger = normalize_dataflow_research_ledger(ledger)
    baseline_mapping = normalized_ledger.to_dict()
    reconstructed_mapping = json.loads(_canonical_json(baseline_mapping))
    _validate_ledger_or_raise(reconstructed_mapping)

    modes = _normalize_modes(traversal_modes)
    normalized_expected_receipts = _normalize_receipt_overrides(expected_receipts)
    unexpected_modes = tuple(mode for mode in normalized_expected_receipts if mode not in modes)
    if unexpected_modes:
        raise ValueError(
            f"expected_receipts contains modes not present in traversal_modes: {','.join(unexpected_modes)}"
        )
    baseline_receipts = {mode: traverse_dataflow_research_ledger(baseline_mapping, mode) for mode in modes}
    reconstructed_receipts = {mode: traverse_dataflow_research_ledger(reconstructed_mapping, mode) for mode in modes}

    for mode, receipt in normalized_expected_receipts.items():
        baseline_receipts[mode] = receipt

    report = compare_replay_runs(
        baseline_mapping,
        reconstructed_mapping,
        traversal_modes=modes,
        run_a_receipts=baseline_receipts,
        run_b_receipts=reconstructed_receipts,
    )
    if not report.replay_stable:
        failing = tuple(entry.traversal_mode for entry in report.entries if not entry.replay_stable)
        raise ValueError(f"replay certification failed for modes: {','.join(failing)}")

    receipt = build_replay_certification_receipt(report)
    pack = LedgerReplayCertificationPack(
        report=report,
        receipt=receipt,
        traversal_receipts_a=tuple(baseline_receipts[mode] for mode in modes),
        traversal_receipts_b=tuple(reconstructed_receipts[mode] for mode in modes),
    )
    validate_ledger_replay_certification(pack)
    return pack
