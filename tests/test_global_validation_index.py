from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.global_validation_index import (
    GlobalValidationEntry,
    GlobalValidationIndex,
    build_global_validation_entry,
    build_global_validation_index,
    build_global_validation_index_from_ordered_hashes,
    get_global_validation_entry_definitions,
    validate_global_validation_entry,
    validate_global_validation_index,
    validate_global_validation_index_matches_hashes,
)


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _mapping() -> dict[str, str]:
    return {field: _h(i + 1) for i, _, field in get_global_validation_entry_definitions()}


def test_global_validation_entry_basics_and_constraints():
    defs = get_global_validation_entry_definitions()
    assert len(defs) == 48
    assert defs[0] == (0, "v151", "canonical_hash")
    assert defs[39] == (39, "v156", "game_world_interaction_report_hash")
    assert defs[42] == (42, "v157", "perturbation_stability_proof_hash")
    assert defs[44] == (44, "v158", "substrate_state_receipt_hash")
    assert defs[47] == (47, "v160", "reality_loop_proof_receipt_hash")
    e1 = build_global_validation_entry(0, _h(1))
    e2 = build_global_validation_entry(0, _h(1))
    assert e1.global_validation_entry_hash == e2.global_validation_entry_hash
    assert all(build_global_validation_entry(i, _h(i + 1)).entry_index == i for i in range(48))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_global_validation_entry(0, "abc")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_global_validation_entry(0, "A" * 64)
    with pytest.raises(ValueError, match="INVALID_ENTRY_INDEX"):
        build_global_validation_entry(True, _h(1))
    with pytest.raises(ValueError, match="INVALID_ENTRY_INDEX"):
        build_global_validation_entry(-1, _h(1))
    with pytest.raises(ValueError, match="INVALID_ENTRY_INDEX"):
        build_global_validation_entry(48, _h(1))
    with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
        GlobalValidationEntry(0, "v152", "canonical_hash", _h(1), e1.global_validation_entry_hash)
    with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
        GlobalValidationEntry(0, "v151", "x", _h(1), e1.global_validation_entry_hash)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_validation_entry(replace(e1, global_validation_entry_hash="xyz"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_validation_entry(replace(e1, global_validation_entry_hash=_h(999)))
    with pytest.raises(FrozenInstanceError):
        e1.entry_index = 3
    assert e1.to_canonical_json() == e1.to_canonical_json()
    assert e1.to_canonical_bytes() == e1.to_canonical_bytes()


def test_global_validation_index_basics_and_constraints():
    m = _mapping()
    i1 = build_global_validation_index(m)
    i2 = build_global_validation_index(m)
    assert i1.global_validation_index_hash == i2.global_validation_index_hash
    assert i1.entry_count == 48
    assert i1.required_entry_count == 48
    assert i1.first_global_validation_entry_hash == i1.global_validation_entries[0].global_validation_entry_hash
    assert i1.final_global_validation_entry_hash == i1.global_validation_entries[-1].global_validation_entry_hash
    assert i1.reality_loop_proof_receipt_hash == i1.global_validation_entries[47].receipt_hash
    with pytest.raises(ValueError, match="MISSING_GLOBAL_VALIDATION_ENTRY"):
        build_global_validation_index({})
    with pytest.raises(ValueError, match="MISSING_GLOBAL_VALIDATION_ENTRY"):
        mm = dict(m)
        mm.pop("canonical_hash")
        build_global_validation_index(mm)
    with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
        mm = dict(m)
        mm["unknown_field"] = _h(555)
        build_global_validation_index(mm)
    with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
        mm = dict(m)
        mm["global_truth_receipt_hash"] = _h(555)
        build_global_validation_index(mm)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        mm = dict(m)
        mm["canonical_hash"] = "abc"
        build_global_validation_index(mm)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        mm = dict(m)
        mm["canonical_hash"] = "A"*64
        build_global_validation_index(mm)
    with pytest.raises(ValueError, match="INVALID_INDEX_MODE"):
        validate_global_validation_index(replace(i1, index_mode="BAD"))
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH"):
        validate_global_validation_index(replace(i1, global_validation_entries=tuple(reversed(i1.global_validation_entries))))
    with pytest.raises(ValueError, match="DUPLICATE_GLOBAL_VALIDATION_ENTRY"):
        dup = list(i1.global_validation_entries)
        dup[1] = dup[0]
        validate_global_validation_index(replace(i1, global_validation_entries=tuple(dup)))
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH"):
        miss = tuple(e for e in i1.global_validation_entries if e.entry_index != 0)
        validate_global_validation_index(replace(i1, global_validation_entries=miss, entry_count=47, required_entry_count=47))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        bad_entry = replace(i1.global_validation_entries[0], receipt_hash=_h(777), global_validation_entry_hash=_h(888))
        entries = (bad_entry,) + i1.global_validation_entries[1:]
        validate_global_validation_index(replace(i1, global_validation_entries=entries))
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH"):
        validate_global_validation_index(replace(i1, first_global_validation_entry_hash=_h(1)))
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH"):
        validate_global_validation_index(replace(i1, final_global_validation_entry_hash=_h(1)))
    with pytest.raises(ValueError, match="REALITY_LOOP_PROOF_HASH_MISMATCH"):
        validate_global_validation_index(replace(i1, reality_loop_proof_receipt_hash=_h(1)))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_validation_index(replace(i1, global_validation_index_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_validation_index(replace(i1, global_validation_index_hash=_h(1234)))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_validation_index(replace(i1, entry_count=True))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_validation_index(replace(i1, required_entry_count=True))
    with pytest.raises(FrozenInstanceError):
        i1.entry_count = 0
    assert i1.to_canonical_json() == i1.to_canonical_json()
    assert i1.to_canonical_bytes() == i1.to_canonical_bytes()


def test_complete_validator_and_ordered_builder_and_boundaries():
    m = _mapping()
    idx = build_global_validation_index(m)
    assert validate_global_validation_index_matches_hashes(idx, m) is True
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_INDEX_MISMATCH"):
        mm = dict(m)
        mm["canonical_hash"] = _h(500)
        validate_global_validation_index_matches_hashes(idx, mm)
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_INDEX_MISMATCH"):
        rehashed = build_global_validation_index(dict(m, canonical_hash=_h(444)))
        validate_global_validation_index_matches_hashes(rehashed, m)
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_INDEX_MISMATCH"):
        validate_global_validation_index_matches_hashes(replace(idx, index_mode=idx.index_mode), dict(m, resonance_hash=_h(1)))
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH"):
        validate_global_validation_index(replace(idx, global_validation_entries=tuple(reversed(idx.global_validation_entries))))
    with pytest.raises(ValueError, match="REALITY_LOOP_PROOF_HASH_MISMATCH"):
        validate_global_validation_index(replace(idx, reality_loop_proof_receipt_hash=_h(2)))

    ordered = [_h(i + 1) for i in range(48)]
    i2 = build_global_validation_index_from_ordered_hashes(ordered)
    assert i2.global_validation_entries[0].receipt_field_name == "canonical_hash"
    assert i2.global_validation_entries[39].receipt_field_name == "game_world_interaction_report_hash"
    assert i2.global_validation_entries[47].receipt_field_name == "reality_loop_proof_receipt_hash"
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH"):
        build_global_validation_index_from_ordered_hashes(ordered[:-1])
    with pytest.raises(ValueError, match="GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH"):
        build_global_validation_index_from_ordered_hashes(ordered + [_h(99)])
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        bad = list(ordered)
        bad[0] = "z"
        build_global_validation_index_from_ordered_hashes(bad)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_validation_entry(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_validation_index(object())


def test_scope_boundary_forbidden_fields():
    """Verify that fields outside the canonical v151→v160.2 scope are rejected at the builder.

    Tests behavior rather than source-text scanning: each of the named forbidden fields
    must cause build_global_validation_index to raise ENTRY_DEFINITION_MISMATCH.
    """
    m = _mapping()
    forbidden_fields = [
        "global_threshold_contract_hash",
        "global_truth_receipt_hash",
        "replay_record_hash",
        "global_replay_proof_hash",
    ]
    for field in forbidden_fields:
        mm = dict(m)
        mm[field] = _h(1)
        with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
            build_global_validation_index(mm)

    with pytest.raises(ValueError, match="ENTRY_DEFINITION_MISMATCH"):
        mm = dict(m)
        mm["unknown_extra_field"] = _h(99)
        build_global_validation_index(mm)
