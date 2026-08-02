# SPDX-License-Identifier: MPL-2.0
"""Canonical receipts for deterministic Strowger exchange routes."""

from __future__ import annotations

from qec.sonify.canonical import canonical_sha256, validate_sha256

from .events import Event
from .model import CLAIM_BOUNDARY, ExchangeConfig, RouteRequest


SCHEMA = "qec.strowger-route-receipt.v1"
QEC_VERSION = "170.3.0"


def build_receipt(
    *,
    config: ExchangeConfig,
    request: RouteRequest,
    mode: str,
    linefinder: int | None,
    selector_trunks: tuple[int, ...],
    connector: tuple[int, int] | None,
    expected_tones: dict[str, int] | None,
    observed_tones: dict[str, int] | None,
    tone_verification: dict[str, object] | None,
    outcome: str,
    events: tuple[Event, ...],
    operator_commands: tuple[dict[str, object], ...],
    fault_plan: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "qec_version": QEC_VERSION,
        "config": config.as_dict(),
        "request": request.as_dict(),
        "request_sha256": request.sha256(),
        "mode": mode,
        "route": {
            "linefinder": linefinder,
            "selector_trunks": list(selector_trunks),
            "connector": list(connector) if connector is not None else None,
        },
        "tones": {
            "expected": expected_tones,
            "observed": observed_tones,
            "verification": tone_verification,
        },
        "outcome": outcome,
        "events": [event.as_dict() for event in events],
        "operator_commands": list(operator_commands),
        "fault_plan": fault_plan,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_receipt(receipt: dict[str, object]) -> dict[str, object]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unexpected Strowger receipt schema")
    if receipt.get("qec_version") != QEC_VERSION:
        raise ValueError("unexpected QEC version")
    observed = validate_sha256(receipt.get("sha256"), "receipt.sha256")
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    if canonical_sha256(unsigned) != observed:
        raise ValueError("Strowger receipt hash mismatch")
    if receipt.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("Strowger claim boundary mismatch")
    request = receipt.get("request")
    if not isinstance(request, dict):
        raise ValueError("receipt request is missing")
    request_hash = validate_sha256(
        receipt.get("request_sha256"), "receipt.request_sha256"
    )
    if canonical_sha256(request) != request_hash:
        raise ValueError("request identity mismatch")
    events = receipt.get("events")
    if not isinstance(events, list):
        raise ValueError("receipt events are missing")
    previous: str | None = None
    for sequence, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("receipt event must be an object")
        event_hash = validate_sha256(
            event.get("event_sha256"), f"events[{sequence}].event_sha256"
        )
        unsigned_event = dict(event)
        unsigned_event.pop("event_sha256", None)
        if unsigned_event.get("sequence") != sequence:
            raise ValueError("event sequence is not contiguous")
        if unsigned_event.get("previous_event_sha256") != previous:
            raise ValueError("event hash chain is broken")
        if canonical_sha256(unsigned_event) != event_hash:
            raise ValueError("event hash mismatch")
        previous = event_hash
    return {"valid": True, "sha256": observed, "events": len(events)}
