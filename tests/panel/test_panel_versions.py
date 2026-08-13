# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

from qec.routing.panel import PanelExchange, PanelRequest, build_claim_validation, build_fault_battery, demo_topology, demo_translation


def test_v171_contract_versions():
    request = PanelRequest("versions", (2, 3, 4, 11), 0, "correction/demo", b"demo")
    exchange = PanelExchange(demo_topology(request.destination), demo_translation(request.digits, request.destination))
    receipt = exchange.route(request).receipt
    assert receipt["topology"]["contract_version"] == "171.0"
    assert receipt["digit_register"]["contract_version"] == "171.1"
    assert receipt["sender_program"]["contract_version"] == "171.1"
    assert receipt["contract_version"] == "171.2"
    assert receipt["translation_table"]["contract_version"] == "171.3"
    assert build_fault_battery(exchange, request)["contract_version"] == "171.4"
    assert build_claim_validation(receipt)["contract_version"] == "171.4"


def test_package_description_distinguishes_candidate_from_published_stable():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    package_readme = Path("PACKAGE_README.md").read_text(encoding="utf-8")
    assert 'version = "171.5.0"' in pyproject
    assert 'readme = "PACKAGE_README.md"' in pyproject
    assert "171.5.0 development/package candidate" in package_readme
    assert "published stable release remains **v170.3.0**" in package_readme
