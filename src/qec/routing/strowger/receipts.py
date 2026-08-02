# SPDX-License-Identifier: MPL-2.0
"""Canonical receipts for deterministic Strowger exchange routes."""

from __future__ import annotations

from qec.sonify.canonical import canonical_sha256, validate_sha256

from .events import Event
from .model import (
    CLAIM_BOUNDARY,
    ExchangeConfig,
    ExchangeMode,
    FaultPlan,
    RouteRequest,
    TrunkState,
)
from .operator import OperatorCommand


SCHEMA = "qec.strowger-route-receipt.v1"
QEC_VERSION = "170.3.0"


def build_receipt(
    *,
    config: ExchangeConfig,
    request: RouteRequest,
    mode: str,
    initial_state: dict[str, object],
    pre_route_state: dict[str, object],
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
        "initial_state": initial_state,
        "pre_route_state": pre_route_state,
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
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _validate_event_chain(events: object) -> int:
    if not isinstance(events, list):
        raise ValueError("receipt events are missing")
    previous: str | None = None
    previous_tick = -1
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
        tick = unsigned_event.get("tick")
        if type(tick) is not int or tick < previous_tick:
            raise ValueError("event ticks must be non-negative and monotonic")
        if unsigned_event.get("previous_event_sha256") != previous:
            raise ValueError("event hash chain is broken")
        if canonical_sha256(unsigned_event) != event_hash:
            raise ValueError("event hash mismatch")
        previous = event_hash
        previous_tick = tick
    return len(events)


def _restore_state(
    config: ExchangeConfig,
    payload: object,
    *,
    label: str,
) -> tuple[list[TrunkState], dict[str, list[TrunkState]]]:
    if not isinstance(payload, dict) or set(payload) != {"linefinders", "trunks"}:
        raise ValueError(f"{label} must contain linefinders and trunks")
    linefinders_raw = payload["linefinders"]
    trunks_raw = payload["trunks"]
    if not isinstance(linefinders_raw, list):
        raise ValueError(f"{label} linefinders must be a list")
    if len(linefinders_raw) != config.linefinders:
        raise ValueError(f"{label} linefinder count does not match config")
    try:
        linefinders = [TrunkState(value) for value in linefinders_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} linefinder state") from exc
    if not isinstance(trunks_raw, dict):
        raise ValueError(f"{label} trunks must be an object")
    expected_names = {stage.name for stage in config.selectors}
    if set(trunks_raw) != expected_names:
        raise ValueError(f"{label} trunk stages do not match config")
    trunks: dict[str, list[TrunkState]] = {}
    for stage in config.selectors:
        values = trunks_raw[stage.name]
        if not isinstance(values, list) or len(values) != stage.trunks:
            raise ValueError(f"{label} trunk count does not match stage {stage.name}")
        try:
            trunks[stage.name] = [TrunkState(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {label} trunk state for {stage.name}") from exc
    return linefinders, trunks


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
    if receipt.get("claim_boundary") != dict(CLAIM_BOUNDARY):
        raise ValueError("Strowger claim boundary mismatch")

    config = ExchangeConfig.from_dict(receipt.get("config"))
    request = RouteRequest.from_dict(receipt.get("request"))
    request.validate_against(config)
    request_hash = validate_sha256(
        receipt.get("request_sha256"), "receipt.request_sha256"
    )
    if request.sha256() != request_hash:
        raise ValueError("request identity mismatch")
    event_count = _validate_event_chain(receipt.get("events"))

    try:
        mode = ExchangeMode(receipt.get("mode"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid exchange mode in receipt") from exc
    fault_plan = FaultPlan.from_dict(receipt.get("fault_plan"))
    commands_raw = receipt.get("operator_commands")
    if not isinstance(commands_raw, list):
        raise ValueError("operator commands must be a list")
    commands = tuple(OperatorCommand.from_dict(item) for item in commands_raw)
    if mode is ExchangeMode.AUTOMATIC and commands:
        raise ValueError("automatic receipts may not contain operator commands")

    initial_linefinders, initial_trunks = _restore_state(
        config, receipt.get("initial_state"), label="initial state"
    )
    _restore_state(config, receipt.get("pre_route_state"), label="pre-route state")

    from .exchange import StrowgerExchange

    exchange = StrowgerExchange(config, mode=mode)
    exchange.linefinder_states = initial_linefinders
    exchange.trunk_states = initial_trunks

    def replay_operator(desk, trunk_states) -> None:
        for command in commands:
            desk.apply(command, trunk_states)

    replay = exchange.route(
        request,
        faults=fault_plan,
        prepare_operator=replay_operator if commands else None,
    )
    if replay.receipt != receipt:
        raise ValueError("Strowger receipt replay mismatch")
    return {
        "valid": True,
        "replayed": True,
        "sha256": observed,
        "events": event_count,
    }
