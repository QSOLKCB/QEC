# SPDX-License-Identifier: MPL-2.0
"""Deterministic exchange, fault, tone, and receipt tests."""

from __future__ import annotations

import copy

import pytest

from qec.routing.strowger import (
    ExchangeConfig,
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
        request_id="syndrome-001",
        digits=(2, 3, 4, 11),
        epoch=7,
        destination="ququart/site-4/pauli-11",
    )


def resign(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    receipt["sha256"] = canonical_sha256(unsigned)


def test_same_input_produces_same_receipt() -> None:
    first = StrowgerExchange(config()).route(request())
    second = StrowgerExchange(config()).route(request())
    assert first.outcome is RouteOutcome.COMMITTED
    assert first.receipt == second.receipt
    validation = validate_receipt(first.receipt)
    assert validation["valid"] is True
    assert validation["replayed"] is True
    assert first.receipt["route"]["selector_trunks"] == [0, 0]


def test_first_free_trunk_hunting_skips_busy_and_quarantined() -> None:
    exchange = StrowgerExchange(config())
    exchange.trunk_states["code-family"][0] = TrunkState.BUSY
    exchange.trunk_states["code-family"][1] = TrunkState.QUARANTINED
    result = exchange.route(request())
    assert result.outcome is RouteOutcome.COMMITTED
    assert result.receipt["route"]["selector_trunks"][0] == 2
    assert validate_receipt(result.receipt)["replayed"] is True


def test_all_trunks_busy_fails_closed() -> None:
    exchange = StrowgerExchange(config())
    exchange.trunk_states["syndrome-sector"] = [TrunkState.BUSY] * 3
    result = exchange.route(request())
    assert result.outcome is RouteOutcome.ALL_TRUNKS_BUSY
    assert result.receipt["tones"]["expected"] is None
    assert validate_receipt(result.receipt)["replayed"] is True


def test_missed_pulse_is_detected() -> None:
    result = StrowgerExchange(config()).route(
        request(),
        faults=FaultPlan(missed_pulses=((0, 1),)),
    )
    assert result.outcome is RouteOutcome.SELECTOR_FAULT
    assert any(
        event["action"] == "pulse_count_mismatch"
        for event in result.receipt["events"]
    )
    assert validate_receipt(result.receipt)["replayed"] is True


def test_tone_mismatch_is_rejected() -> None:
    result = StrowgerExchange(config()).route(
        request(),
        faults=FaultPlan(tone_offsets_hz=(1, 0, 0)),
    )
    assert result.outcome is RouteOutcome.TONE_MISMATCH
    assert result.receipt["tones"]["verification"]["verified"] is False
    assert validate_receipt(result.receipt)["replayed"] is True


def test_receipt_tampering_breaks_hash() -> None:
    receipt = StrowgerExchange(config()).route(request()).receipt
    tampered = copy.deepcopy(receipt)
    tampered["outcome"] = "tone_mismatch"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_receipt(tampered)


def test_resigned_receipt_forgery_fails_replay() -> None:
    receipt = StrowgerExchange(config()).route(request()).receipt
    tampered = copy.deepcopy(receipt)
    tampered["outcome"] = "tone_mismatch"
    resign(tampered)
    with pytest.raises(ValueError, match="replay mismatch"):
        validate_receipt(tampered)
