# SPDX-License-Identifier: MPL-2.0
from qec.routing.panel import PanelExchange, PanelFaultPlan, PanelRequest, build_fault_battery, demo_topology, demo_translation

DEST = "ququart/site-4/pauli-11"
DIGITS = (2, 3, 4, 11)

def test_declared_panel_failure_cases_are_reproducible():
    request = PanelRequest("panel-test", DIGITS, 0, DEST, b"demo")
    exchange = PanelExchange(demo_topology(DEST), demo_translation(DIGITS, DEST))
    assert exchange.route(request, faults=PanelFaultPlan(busy_banks=("bank-a",))).receipt["route"]["path_id"] == "path-b"
    assert exchange.route(request, faults=PanelFaultPlan(busy_banks=("bank-a", "bank-b"))).outcome == "capacity_exhausted"
    assert exchange.route(request, faults=PanelFaultPlan(stalled_motor_groups=("motor-a",))).outcome == "motor_stall"
    assert exchange.route(request, faults=PanelFaultPlan(translation_corruption=True)).outcome == "translation_corruption"
    assert exchange.route(request, faults=PanelFaultPlan(sender_disagreement=True)).outcome == "sender_disagreement"
    assert build_fault_battery(exchange, request) == build_fault_battery(exchange, request)
