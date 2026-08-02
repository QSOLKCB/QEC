# SPDX-License-Identifier: MPL-2.0
"""Optional Operator Desk permissions and evidence tests."""

from __future__ import annotations

import pytest

from qec.routing.strowger import (
    ExchangeConfig,
    ExchangeMode,
    OperatorAction,
    OperatorCommand,
    RouteRequest,
    StageConfig,
    StrowgerExchange,
    validate_receipt,
)


def config() -> ExchangeConfig:
    return ExchangeConfig(
        linefinders=1,
        selectors=(StageConfig("family", 3, trunks=2),),
        connector_vertical_radix=10,
        connector_rotary_radix=10,
    )


def request() -> RouteRequest:
    return RouteRequest("operator-demo", (1, 2, 3), 0, "qutrit/site-2/X")


def test_automatic_mode_rejects_operator_callback() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.AUTOMATIC)
    with pytest.raises(PermissionError, match="requires supervised"):
        exchange.route(request(), prepare_operator=lambda desk, trunks: None)


def test_supervised_operator_quarantine_is_hash_chained() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.SUPERVISED)

    def prepare(desk, trunks) -> None:
        desk.quarantine(
            trunks,
            selector="family",
            contact=0,
            operator_id="local-console",
            reason="contact-bounce-detected",
        )
        desk.inspect(
            operator_id="local-console",
            target="family:1",
            reason="confirm alternate trunk",
        )

    result = exchange.route(request(), prepare_operator=prepare)
    assert result.receipt["route"]["selector_trunks"] == [1]
    assert result.receipt["initial_state"]["trunks"]["family"] == ["free", "free"]
    assert result.receipt["pre_route_state"]["trunks"]["family"] == [
        "quarantined",
        "free",
    ]
    assert len(result.receipt["operator_commands"]) == 2
    assert result.receipt["operator_commands"][0]["action"] == "quarantine_trunk"
    assert any(
        event["action"] == "operator_quarantine_trunk"
        for event in result.receipt["events"]
    )
    assert validate_receipt(result.receipt)["replayed"] is True


def test_manual_actions_require_manual_mode() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.SUPERVISED)

    def prepare(desk, trunks) -> None:
        desk.record(
            OperatorCommand(
                action=OperatorAction.MANUAL_STEP,
                operator_id="operator-1",
                target="selector-0",
                reason="maintenance exercise",
                value=1,
            )
        )

    with pytest.raises(PermissionError, match="requires manual"):
        exchange.route(request(), prepare_operator=prepare)


def test_manual_step_and_seizure_are_executed_but_cannot_force_accept() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.MANUAL)

    def prepare(desk, trunks) -> None:
        desk.manual_step(
            operator_id="operator-1",
            target="selector-0",
            reason="demonstration",
            value=1,
        )
        desk.seize(
            trunks,
            selector="family",
            contact=0,
            operator_id="operator-1",
            reason="maintenance exercise",
        )

    result = exchange.route(request(), prepare_operator=prepare)
    assert result.receipt["route"]["selector_trunks"] == [1]
    assert result.receipt["pre_route_state"]["trunks"]["family"] == [
        "busy",
        "free",
    ]
    assert result.receipt["claim_boundary"]["operator_may_force_accept"] is False
    assert validate_receipt(result.receipt)["replayed"] is True


def test_unrecorded_operator_state_mutation_is_rejected() -> None:
    exchange = StrowgerExchange(config(), mode=ExchangeMode.MANUAL)

    def prepare(_desk, trunks) -> None:
        trunks["family"][0] = trunks["family"][1]

    with pytest.raises(ValueError, match="not explained"):
        exchange.route(request(), prepare_operator=prepare)
