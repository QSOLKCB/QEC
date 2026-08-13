# SPDX-License-Identifier: MPL-2.0
from qec.routing.panel import (
    MotorGroup, PanelBank, PanelExchange, PanelFaultPlan, PanelPath, PanelRequest,
    PanelTopology, TranslationEntry, TranslationTable, validate_route_receipt,
)

DEST = "ququart/site-4/pauli-11"
DIGITS = (2, 3, 4, 11)

def test_unavailable_path_keeps_healthy_sibling_selectable():
    topology = PanelTopology(
        "same-bank",
        (MotorGroup("motor-a", ("bank-a",)),),
        (PanelBank("bank-a", "motor-a", 2, 0, 9),),
        (
            PanelPath("path-a", "bank-a", 4, DEST),
            PanelPath("path-b", "bank-a", 5, DEST),
        ),
    )
    table = TranslationTable(
        "same-bank-table",
        "1",
        "first_declared_free_path",
        (TranslationEntry(DIGITS, DEST, ("path-a", "path-b")),),
    )
    request = PanelRequest("same-bank", DIGITS, 0, DEST, b"demo")
    result = PanelExchange(topology, table).route(
        request,
        faults=PanelFaultPlan(unavailable_paths=("path-a",)),
    )
    assert result.outcome == "committed"
    assert result.receipt["route"]["path_id"] == "path-b"
    assert any(event["action"] == "path_unavailable" for event in result.receipt["events"])
    assert validate_route_receipt(result.receipt)["replayed"] is True
