# SPDX-License-Identifier: MPL-2.0
"""Regression tests for the Codex review hardening pass."""

from __future__ import annotations

import copy

import pytest

from qec.adapters.nexus.version import QEC_PACKAGE_VERSION
from qec.routing.strowger import (
    CLAIM_BOUNDARY,
    ExchangeConfig,
    ExchangeMode,
    FaultPlan,
    RouteOutcome,
    RouteRequest,
    StageConfig,
    StrowgerExchange,
    TrunkState,
    validate_receipt,
)
from qec.sonify.canonical import canonical_sha256


def config() -> ExchangeConfig:
    return ExchangeConfig(
        linefinders=2,
        selectors=(
            StageConfig("code-family", 3, trunks=3),
            StageConfig("syndrome-sector", 4, trunks=3),
        ),
        connector_vertical_radix=10,
        connector_rotary_radix=16,
    )


def request() -> RouteRequest:
    return RouteRequest(
        request_id="codex-hardening",
        digits=(2, 3, 4, 11),
        epoch=9,
        destination="ququart/site-4/pauli-11",
    )


def resign(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    receipt["sha256"] = canonical_sha256(unsigned)


def rehash_events(receipt: dict[str, object]) -> None:
    events = receipt["events"]
    assert isinstance(events, list)
    previous = None
    for sequence, event in enumerate(events):
        assert isinstance(event, dict)
        unsigned = dict(event)
        unsigned.pop("event_sha256", None)
        unsigned["sequence"] = sequence
        unsigned["previous_event_sha256"] = previous
        event.clear()
        event.update(unsigned)
        event_hash = canonical_sha256(unsigned)
        event["event_sha256"] = event_hash
        previous = event_hash


def test_connector_pulse_fault_is_detected_before_commit() -> None:
    connector_vertical_stage = len(config().selectors)
    result = StrowgerExchange(config()).route(
        request(),
        faults=FaultPlan(missed_pulses=((connector_vertical_stage, 1),)),
    )
    assert result.outcome is RouteOutcome.CONNECTOR_FAULT
    assert result.receipt["route"]["connector"] is None
    assert any(
        event["device"] == "connector-vertical"
        and event["action"] == "pulse_count_mismatch"
        for event in result.receipt["events"]
    )
    assert validate_receipt(result.receipt)["replayed"] is True


def test_fault_references_outside_encoded_train_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown route stage"):
        StrowgerExchange(config()).route(
            request(), faults=FaultPlan(missed_pulses=((99, 1),))
        )
    with pytest.raises(ValueError, match="nonexistent pulse"):
        StrowgerExchange(config()).route(
            request(), faults=FaultPlan(duplicate_pulses=((0, 3),))
        )


def test_linefinder_exhaustion_has_distinct_outcome() -> None:
    exchange = StrowgerExchange(config())
    exchange.linefinder_states = [TrunkState.BUSY, TrunkState.BUSY]
    result = exchange.route(request())
    assert result.outcome is RouteOutcome.NO_LINEFINDER_AVAILABLE
    assert result.receipt["outcome"] == "no_linefinder_available"
    assert result.receipt["route"]["linefinder"] is None
    assert validate_receipt(result.receipt)["replayed"] is True


def test_claim_boundary_is_not_mutable_through_a_receipt() -> None:
    receipt = StrowgerExchange(config()).route(request()).receipt
    receipt["claim_boundary"]["operator_may_force_accept"] = True
    assert CLAIM_BOUNDARY["operator_may_force_accept"] is False
    resign(receipt)
    with pytest.raises(ValueError, match="claim boundary mismatch"):
        validate_receipt(receipt)


def test_resigned_operator_target_forgery_fails_actual_replay() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.SUPERVISED)

    def prepare(desk, trunks) -> None:
        desk.quarantine(
            trunks,
            selector="code-family",
            contact=0,
            operator_id="operator-1",
            reason="contact bounce",
        )

    receipt = exchange.route(request(), prepare_operator=prepare).receipt
    forged = copy.deepcopy(receipt)
    forged["operator_commands"][0]["target"] = "code-family:1"
    for event in forged["events"]:
        if event["action"] == "operator_quarantine_trunk":
            event["details"]["target"] = "code-family:1"
    rehash_events(forged)
    resign(forged)
    with pytest.raises(ValueError, match="replay mismatch"):
        validate_receipt(forged)


def test_receipt_preserves_encoded_pulse_timing() -> None:
    receipt = StrowgerExchange(config()).route(request()).receipt
    pulse_events = [
        event
        for event in receipt["events"]
        if event["action"] in {"pulse_received", "missed_pulse", "duplicate_pulse"}
    ]
    assert pulse_events
    assert max(event["tick"] for event in pulse_events) > 0
    assert all(
        event["tick"] == event["details"]["encoded_tick"]
        for event in pulse_events
    )
    assert [event["tick"] for event in receipt["events"]] == sorted(
        event["tick"] for event in receipt["events"]
    )
    assert validate_receipt(receipt)["replayed"] is True


def test_public_package_identity_matches_distribution_release() -> None:
    assert QEC_PACKAGE_VERSION == "170.3.0"
