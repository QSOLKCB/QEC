from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.reality_loop_composition_spec import (
    COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION,
    CompositionSlot,
    RealityLoopCompositionSpec,
    build_composition_slot,
    build_reality_loop_composition_spec,
    build_reality_loop_composition_spec_from_ordered_hashes,
    get_reality_loop_slot_definitions,
    validate_composition_slot,
    validate_reality_loop_composition_spec,
    validate_reality_loop_composition_spec_matches_hashes,
)


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _mapping() -> dict[str, str]:
    return {field: _h(i + 1) for i, (_, _, field) in enumerate(get_reality_loop_slot_definitions())}


def test_composition_slot_basics():
    defs = get_reality_loop_slot_definitions()
    assert len(defs) == 19
    assert defs[0] == (0, "v151", "canonical_hash")
    assert defs[13] == (13, "v156", "game_world_interaction_report_hash")
    assert defs[18] == (18, "v159", "loop_termination_proof_hash")

    a = build_composition_slot(0, _h(10))
    b = build_composition_slot(0, _h(10))
    assert a.composition_slot_hash == b.composition_slot_hash

    slots = [build_composition_slot(i, _h(i + 100)) for i in range(19)]
    assert len(slots) == 19

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_composition_slot(0, "bad")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_composition_slot(0, "A" * 64)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_composition_slot(replace(a, composition_slot_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_composition_slot(replace(a, composition_slot_hash=_h(999)))

    for bad_idx in (True, -1, 19):
        with pytest.raises(ValueError, match="INVALID_SLOT_INDEX"):
            build_composition_slot(bad_idx, _h(1))

    with pytest.raises(ValueError, match="SLOT_DEFINITION_MISMATCH"):
        CompositionSlot(0, "v152", "canonical_hash", _h(1), a.composition_slot_hash)
    with pytest.raises(ValueError, match="SLOT_DEFINITION_MISMATCH"):
        CompositionSlot(0, "v151", "semantic_field_hash", _h(1), a.composition_slot_hash)

    with pytest.raises(FrozenInstanceError):
        a.slot_index = 1

    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_reality_loop_spec_basics():
    mapping = _mapping()
    spec = build_reality_loop_composition_spec(mapping)
    spec2 = build_reality_loop_composition_spec(mapping)
    assert spec.composition_spec_hash == spec2.composition_spec_hash
    assert spec.slot_count == 19
    assert spec.required_slot_count == 19

    with pytest.raises(ValueError, match="MISSING_COMPOSITION_SLOT"):
        missing = dict(mapping)
        missing.pop("canonical_hash")
        build_reality_loop_composition_spec(missing)

    with pytest.raises(ValueError, match="SLOT_DEFINITION_MISMATCH|INVALID_RECEIPT_FIELD_NAME"):
        extra = dict(mapping)
        extra["unknown_field"] = _h(123)
        build_reality_loop_composition_spec(extra)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        bad = dict(mapping)
        bad["canonical_hash"] = "bad"
        build_reality_loop_composition_spec(bad)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        bad = dict(mapping)
        bad["canonical_hash"] = "A" * 64
        build_reality_loop_composition_spec(bad)

    with pytest.raises(ValueError, match="INVALID_COMPOSITION_MODE"):
        validate_reality_loop_composition_spec(replace(spec, composition_mode="BAD"))
    with pytest.raises(ValueError, match="COMPOSITION_SLOT_COUNT_MISMATCH"):
        validate_reality_loop_composition_spec(replace(spec, slot_count=True))
    with pytest.raises(ValueError, match="COMPOSITION_SLOT_COUNT_MISMATCH"):
        validate_reality_loop_composition_spec(replace(spec, required_slot_count=True))

    unsorted_slots = tuple(reversed(spec.composition_slots))
    with pytest.raises(ValueError, match="COMPOSITION_SLOT_ORDER_MISMATCH"):
        validate_reality_loop_composition_spec(replace(spec, composition_slots=unsorted_slots))

    spec_dup = build_reality_loop_composition_spec(mapping)
    dup_idx = tuple(spec_dup.composition_slots)
    object.__setattr__(dup_idx[1], "slot_index", 0)
    with pytest.raises(ValueError, match="SLOT_DEFINITION_MISMATCH|DUPLICATE_COMPOSITION_SLOT"):
        validate_reality_loop_composition_spec(replace(spec_dup, composition_slots=dup_idx))

    spec_dup2 = build_reality_loop_composition_spec(mapping)
    dup_field = tuple(spec_dup2.composition_slots)
    object.__setattr__(dup_field[1], "receipt_field_name", spec_dup2.composition_slots[0].receipt_field_name)
    with pytest.raises(ValueError):
        validate_reality_loop_composition_spec(replace(spec_dup2, composition_slots=dup_field))

    with pytest.raises(ValueError, match="COMPOSITION_SLOT_COUNT_MISMATCH"):
        validate_reality_loop_composition_spec(replace(spec, composition_slots=spec.composition_slots[:-1]))

    spec_tampered = build_reality_loop_composition_spec(mapping)
    object.__setattr__(spec_tampered.composition_slots[0], "receipt_hash", _h(4242))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_reality_loop_composition_spec(spec_tampered)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_reality_loop_composition_spec(replace(spec, composition_spec_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_reality_loop_composition_spec(replace(spec, composition_spec_hash=_h(9999)))

    with pytest.raises(FrozenInstanceError):
        spec.slot_count = 10

    assert spec.to_canonical_json() == spec2.to_canonical_json()
    assert spec.to_canonical_bytes() == spec2.to_canonical_bytes()


def test_complete_validator_and_ordered_builder_and_boundaries():
    mapping = _mapping()
    spec = build_reality_loop_composition_spec(mapping)
    assert validate_reality_loop_composition_spec_matches_hashes(spec, mapping) is True

    wrong = dict(mapping)
    wrong["canonical_hash"] = _h(777)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_reality_loop_composition_spec_matches_hashes(spec, wrong)

    altered_slot = replace(spec.composition_slots[0], receipt_hash=_h(888), composition_slot_hash=build_composition_slot(0, _h(888)).composition_slot_hash)
    altered = replace(
        spec,
        composition_slots=(altered_slot, *spec.composition_slots[1:]),
        composition_spec_hash=build_reality_loop_composition_spec({**mapping, "canonical_hash": _h(888)}).composition_spec_hash,
    )
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_reality_loop_composition_spec_matches_hashes(altered, mapping)

    altered_mode = replace(spec)
    object.__setattr__(altered_mode, "composition_mode", "BAD_MODE")
    with pytest.raises(ValueError, match="INVALID_COMPOSITION_MODE"):
        validate_reality_loop_composition_spec_matches_hashes(altered_mode, mapping)

    reordered = replace(spec)
    object.__setattr__(reordered, "composition_slots", tuple(reversed(spec.composition_slots)))
    payload_hashes = [spec.composition_slots[i].receipt_hash for i in range(19)]
    with pytest.raises(ValueError, match="COMPOSITION_SLOT_ORDER_MISMATCH"):
        validate_reality_loop_composition_spec_matches_hashes(reordered, _mapping())

    built = build_reality_loop_composition_spec_from_ordered_hashes(payload_hashes)
    assert built.composition_slots[0].receipt_field_name == "canonical_hash"
    assert built.composition_slots[13].receipt_field_name == "game_world_interaction_report_hash"
    assert built.composition_slots[18].receipt_field_name == "loop_termination_proof_hash"

    with pytest.raises(ValueError, match="COMPOSITION_SLOT_COUNT_MISMATCH"):
        build_reality_loop_composition_spec_from_ordered_hashes(payload_hashes[:-1])
    with pytest.raises(ValueError, match="COMPOSITION_SLOT_COUNT_MISMATCH"):
        build_reality_loop_composition_spec_from_ordered_hashes(payload_hashes + [_h(1)])
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        bad = list(payload_hashes)
        bad[0] = "bad"
        build_reality_loop_composition_spec_from_ordered_hashes(bad)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_composition_slot(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_reality_loop_composition_spec(object())

    bad_child = replace(spec)
    object.__setattr__(bad_child, "composition_slots", (object(), *spec.composition_slots[1:]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_reality_loop_composition_spec(bad_child)


def test_scope_boundary_scan():
    text = open("src/qec/analysis/reality_loop_composition_spec.py", "r", encoding="utf-8").read()
    forbidden = [
        "CrossArcIdentityLink", "RealityLoopProofReceipt", "GlobalTruthReceipt",
        "GlobalValidationIndex", "GlobalThresholdContract", "GlobalReplayProof",
        "global_truth", "global_validation", "runtime_replay_execution",
        "recursive_execution", "gameplay", "render", "step_world", "execute_action",
        "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(",
        "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy",
    ]
    for token in forbidden:
        assert token not in text

    allowed = ["RealityLoopCompositionSpec", "CompositionSlot", "composition_slot_hash", "composition_spec_hash"]
    for token in allowed:
        assert token in text
