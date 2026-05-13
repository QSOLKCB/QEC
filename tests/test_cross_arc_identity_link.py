from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.cross_arc_identity_link import (
    CrossArcIdentityLink,
    CrossArcIdentityLinkReceipt,
    build_cross_arc_identity_link,
    build_cross_arc_identity_link_receipt,
    get_reality_loop_link_definitions,
    validate_cross_arc_identity_link,
    validate_cross_arc_identity_link_receipt,
    validate_cross_arc_identity_link_receipt_with_composition_spec,
    validate_cross_arc_identity_link_with_composition_spec,
)
from qec.analysis.reality_loop_composition_spec import (
    build_reality_loop_composition_spec,
    get_reality_loop_slot_definitions,
)


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _spec(offset: int = 1):
    mapping = {field: _h(i + offset) for i, (_, _, field) in enumerate(get_reality_loop_slot_definitions())}
    return build_reality_loop_composition_spec(mapping)


def test_cross_arc_identity_link_basics():
    spec = _spec()
    defs = get_reality_loop_link_definitions()
    assert len(defs) == 18
    assert defs[0] == (0, 1, "LINK_000_001")
    assert defs[13] == (13, 14, "LINK_013_014")
    assert defs[17] == (17, 18, "LINK_017_018")

    a = build_cross_arc_identity_link(spec, 0)
    b = build_cross_arc_identity_link(spec, 0)
    assert a.cross_arc_identity_link_hash == b.cross_arc_identity_link_hash
    assert len([build_cross_arc_identity_link(spec, i) for i in range(18)]) == 18

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, cross_arc_identity_link_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_cross_arc_identity_link(replace(a, cross_arc_identity_link_hash=_h(999)))

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, source_receipt_hash="bad"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, target_receipt_hash="bad"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, source_composition_slot_hash="bad"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, target_composition_slot_hash="bad"))

    for bad_idx in (True, -1, 18):
        with pytest.raises(ValueError, match="INVALID_LINK_INDEX"):
            build_cross_arc_identity_link(spec, bad_idx)

    with pytest.raises(ValueError, match="INVALID_LINK_KIND"):
        validate_cross_arc_identity_link(replace(a, link_kind="BAD"))
    with pytest.raises(ValueError, match="LINK_DEFINITION_MISMATCH|SLOT_LINK_MISMATCH"):
        validate_cross_arc_identity_link(replace(a, source_slot_index=1))
    with pytest.raises(ValueError, match="LINK_DEFINITION_MISMATCH|SLOT_LINK_MISMATCH"):
        validate_cross_arc_identity_link(replace(a, target_slot_index=3))
    with pytest.raises(ValueError, match="LINK_DEFINITION_MISMATCH"):
        validate_cross_arc_identity_link(replace(a, link_label="LINK_777_888"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link(replace(a, source_receipt_hash="A" * 64))

    with pytest.raises(FrozenInstanceError):
        a.link_index = 1

    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_cross_arc_identity_link_receipt_basics():
    spec = _spec()
    receipt = build_cross_arc_identity_link_receipt(spec)
    receipt2 = build_cross_arc_identity_link_receipt(spec)
    assert receipt.cross_arc_identity_link_receipt_hash == receipt2.cross_arc_identity_link_receipt_hash
    assert len(receipt.cross_arc_identity_links) == 18
    assert [l.link_index for l in receipt.cross_arc_identity_links] == list(range(18))
    assert receipt.link_count == 18
    assert receipt.required_link_count == 18

    with pytest.raises(ValueError, match="CROSS_ARC_LINK_COUNT_MISMATCH"):
        validate_cross_arc_identity_link_receipt(replace(receipt, link_count=True))
    with pytest.raises(ValueError, match="CROSS_ARC_LINK_COUNT_MISMATCH"):
        validate_cross_arc_identity_link_receipt(replace(receipt, required_link_count=True))

    links = list(receipt.cross_arc_identity_links)
    dup = replace(links[-1])
    object.__setattr__(dup, "link_index", 0)
    dup_idx = tuple([links[0], *links[1:-1], dup])
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_links=dup_idx))

    with pytest.raises(ValueError, match="CROSS_ARC_LINK_COUNT_MISMATCH"):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_links=receipt.cross_arc_identity_links[:-1]))

    unsorted_links = tuple(reversed(receipt.cross_arc_identity_links))
    with pytest.raises(ValueError, match="CROSS_ARC_LINK_ORDER_MISMATCH"):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_links=unsorted_links))

    assert receipt.first_composition_slot_hash == spec.composition_slots[0].composition_slot_hash
    assert receipt.final_composition_slot_hash == spec.composition_slots[18].composition_slot_hash

    tampered_link = replace(receipt.cross_arc_identity_links[0])
    object.__setattr__(tampered_link, "source_receipt_hash", _h(999))
    object.__setattr__(tampered_link, "cross_arc_identity_link_hash", _h(123))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_links=(tampered_link, *receipt.cross_arc_identity_links[1:])))

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_link_receipt_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_cross_arc_identity_link_receipt(replace(receipt, cross_arc_identity_link_receipt_hash=_h(9999)))

    with pytest.raises(FrozenInstanceError):
        receipt.link_count = 1

    assert receipt.to_canonical_json() == receipt2.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_complete_validators_and_boundaries():
    spec = _spec()
    other_spec = _spec(500)
    link = build_cross_arc_identity_link(spec, 1)
    receipt = build_cross_arc_identity_link_receipt(spec)
    assert validate_cross_arc_identity_link_with_composition_spec(link, spec) is True
    assert validate_cross_arc_identity_link_receipt_with_composition_spec(receipt, spec) is True

    with pytest.raises(ValueError, match="SLOT_LINK_MISMATCH"):
        validate_cross_arc_identity_link_with_composition_spec(link, other_spec)
    with pytest.raises(ValueError, match="CROSS_ARC_IDENTITY_LINK_RECEIPT_MISMATCH"):
        validate_cross_arc_identity_link_receipt_with_composition_spec(receipt, other_spec)

    bad_link = replace(link)
    object.__setattr__(bad_link, "source_receipt_hash", _h(777))
    object.__setattr__(bad_link, "cross_arc_identity_link_hash", _h(1))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_with_composition_spec(bad_link, spec)
    bad_link2 = replace(link)
    object.__setattr__(bad_link2, "target_receipt_hash", _h(888))
    object.__setattr__(bad_link2, "cross_arc_identity_link_hash", _h(2))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_with_composition_spec(bad_link2, spec)

    reordered = replace(receipt)
    object.__setattr__(reordered, "cross_arc_identity_links", tuple(reversed(receipt.cross_arc_identity_links)))
    object.__setattr__(reordered, "cross_arc_identity_link_receipt_hash", _h(3))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_receipt_with_composition_spec(reordered, spec)
    bad_first = replace(receipt)
    object.__setattr__(bad_first, "first_composition_slot_hash", _h(4))
    object.__setattr__(bad_first, "cross_arc_identity_link_receipt_hash", _h(5))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_receipt_with_composition_spec(bad_first, spec)
    bad_final = replace(receipt)
    object.__setattr__(bad_final, "final_composition_slot_hash", _h(6))
    object.__setattr__(bad_final, "cross_arc_identity_link_receipt_hash", _h(7))
    with pytest.raises(ValueError):
        validate_cross_arc_identity_link_receipt_with_composition_spec(bad_final, spec)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_cross_arc_identity_link(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_cross_arc_identity_link_receipt(object())

    malformed_child = replace(receipt)
    object.__setattr__(malformed_child, "cross_arc_identity_links", (object(), *receipt.cross_arc_identity_links[1:]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_cross_arc_identity_link_receipt(malformed_child)


def test_scope_boundary_scan():
    with open("src/qec/analysis/cross_arc_identity_link.py", "r", encoding="utf-8") as f:
        text = f.read()
    forbidden = [
        "RealityLoopProofReceipt", "GlobalTruthReceipt", "GlobalValidationIndex", "GlobalThresholdContract",
        "GlobalReplayProof", "global_truth", "global_validation", "runtime_replay_execution", "recursive_execution",
        "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess",
        "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy",
    ]
    for token in forbidden:
        assert token not in text

    allowed = ["CrossArcIdentityLink", "CrossArcIdentityLinkReceipt", "cross_arc_identity_link_hash", "cross_arc_identity_link_receipt_hash"]
    for token in allowed:
        assert token in text
