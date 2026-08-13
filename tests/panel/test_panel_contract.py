# SPDX-License-Identifier: MPL-2.0
import pytest
from qec.routing.panel import (
    MotorGroup, PanelBank, PanelExchange, PanelPath, PanelRequest, PanelTopology,
    TranslationEntry, TranslationTable, build_claim_validation, build_sender_register_receipt,
    compile_sender_program, demo_topology, demo_translation, seal_digit_register,
    validate_route_receipt,
)
from qec.sonify.canonical import canonical_sha256

DEST = "ququart/site-4/pauli-11"
DIGITS = (2, 3, 4, 11)

def make_request():
    return PanelRequest("panel-test", DIGITS, 7, DEST, b"correction-payload")

def make_exchange():
    return PanelExchange(demo_topology(DEST), demo_translation(DIGITS, DEST))

def test_sender_program_is_deterministic_and_translation_bound():
    request = make_request(); topology = demo_topology(DEST); register = seal_digit_register(request)
    first = compile_sender_program(topology, demo_translation(DIGITS, DEST, version="1"), register)
    repeat = compile_sender_program(topology, demo_translation(DIGITS, DEST, version="1"), register)
    changed = compile_sender_program(topology, demo_translation(DIGITS, DEST, version="2"), register)
    assert first == repeat
    assert first["sha256"] != changed["sha256"]

def test_sender_seals_before_actuation_and_payload_identity_is_stable():
    request = make_request(); result = make_exchange().route(request); events = result.receipt["events"]
    sender = next(i for i, event in enumerate(events) if event["action"] == "sender_program_sealed")
    actuation = min(i for i, event in enumerate(events) if event["phase"] == "actuation")
    assert sender < actuation
    assert result.receipt["payload_identity"]["before_sha256"] == result.receipt["payload_identity"]["after_sha256"]

def test_receipt_replay_and_claim_validation():
    result = make_exchange().route(make_request())
    assert validate_route_receipt(result.receipt)["replayed"] is True
    assert make_exchange().route(make_request()).receipt == result.receipt
    assert build_claim_validation(result.receipt)["all_passed"] is True

def test_fallback_requires_declared_deterministic_rule():
    with pytest.raises(ValueError, match="declared deterministic fallback"):
        TranslationTable("bad", "1", "none", (TranslationEntry(DIGITS, DEST, ("path-a", "path-b")),))

def test_selector_inventory_is_bounded():
    with pytest.raises(ValueError, match="bounded selector movement"):
        PanelTopology("bad", (MotorGroup("motor-a", ("bank-a",)),), (PanelBank("bank-a", "motor-a", 1, 0, 3),), (PanelPath("path-a", "bank-a", 4, DEST),))

def test_paths_may_not_share_the_same_actuation_coordinate():
    with pytest.raises(ValueError, match="actuation coordinate"):
        PanelTopology(
            "duplicate-coordinate",
            (MotorGroup("motor-a", ("bank-a",)),),
            (PanelBank("bank-a", "motor-a", 2, 0, 9),),
            (
                PanelPath("path-a", "bank-a", 4, DEST),
                PanelPath("path-b", "bank-a", 4, "different/destination"),
            ),
        )

def test_sender_register_receipt_requires_complete_canonical_program():
    register = seal_digit_register(make_request())
    program = compile_sender_program(demo_topology(DEST), demo_translation(DIGITS, DEST), register)
    incomplete = dict(program)
    incomplete.pop("contract_version")
    unsigned = dict(incomplete)
    unsigned.pop("sha256")
    incomplete["sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="fields are not canonical"):
        build_sender_register_receipt(register, incomplete)
