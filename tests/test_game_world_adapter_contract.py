from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import zipfile

import pytest

from qec.analysis.game_world_intake_contract import build_game_world_archive, build_game_world_corpus_manifest, build_game_world_intake_receipt
import qec.analysis.game_world_adapter_contract as gwac
from qec.analysis.game_world_adapter_contract import (
    ActionAlphabet, ActionAtom, WorldAdapterContractReceipt, WorldAdapterSpec,
    build_action_alphabet, build_action_atom, build_world_adapter_contract_receipt, build_world_adapter_spec,
    validate_action_alphabet, validate_action_atom, validate_world_adapter_contract_receipt, validate_world_adapter_spec,
)
from qec.analysis.canonical_hashing import sha256_hex


def _mkzip(name: str) -> str:
    d = tempfile.mkdtemp()
    p = Path(d) / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("main.py", "print('x')")
    return str(p)


def _mk_manifest_and_receipt(names=("doom.zip", "atari.zip")):
    manifest = build_game_world_corpus_manifest([_mkzip(n) for n in names])
    receipt = build_game_world_intake_receipt(manifest, "a" * 64)
    return manifest, receipt

# ActionAtom tests

def test_action_atom_determinism():
    hs = {build_action_atom("NO_OP", "META", ()).action_atom_hash for _ in range(10)}
    assert len(hs) == 1

def test_invalid_action_code_rejected():
    for code in ["", "abc", "A B", "A-1", "_A"]:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            build_action_atom(code, "META", ())

def test_invalid_action_kind_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_action_atom("NO_OP", "BAD", ())

def test_parameter_slots_sorted_and_deduplicated_by_builder():
    atom = build_action_atom("KEY_PRESS", "PIXEL_BUTTON", ["z", "a"]) 
    assert atom.parameter_slots == ("a", "z")

def test_duplicate_parameter_slots_rejected():
    with pytest.raises(ValueError, match="DUPLICATE_PARAMETER_SLOT"):
        build_action_atom("KEY_PRESS", "PIXEL_BUTTON", ["a", "a"])

def test_direct_unsorted_parameter_slots_rejected():
    payload = {"action_code": "KEY_PRESS", "action_kind": "PIXEL_BUTTON", "parameter_slots": ["z", "a"]}
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ActionAtom("KEY_PRESS", "PIXEL_BUTTON", ("z", "a"), sha256_hex(payload))

def test_action_atom_hash_tamper_detected():
    good = build_action_atom("NO_OP", "META", ())
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(good, action_atom_hash="b" * 64)

def test_action_atom_malformed_hash_rejected():
    good = build_action_atom("NO_OP", "META", ())
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(good, action_atom_hash="zzz")

# Alphabet tests

def test_action_alphabet_determinism_for_all_world_families():
    for wf in sorted(gwac._ALLOWED_WORLD_FAMILIES):
        hs = {build_action_alphabet(wf).action_alphabet_hash for _ in range(3)}
        assert len(hs) == 1

def test_unknown_world_family_gets_noop_alphabet():
    a = build_action_alphabet("UNKNOWN")
    assert a.action_count == 1 and a.actions[0].action_code == "NO_OP"

def test_action_alphabet_sorted():
    a = build_action_alphabet("RAYCAST_FPS")
    assert a.actions == tuple(sorted(a.actions, key=lambda x: (x.action_code, x.action_atom_hash)))

def test_duplicate_action_code_rejected():
    atom = build_action_atom("NO_OP", "META", ())
    payload = {"world_family": "UNKNOWN", "actions": [atom.to_dict(), atom.to_dict()], "action_count": 2}
    with pytest.raises(ValueError, match="DUPLICATE_ACTION"):
        ActionAlphabet("UNKNOWN", (atom, atom), 2, sha256_hex(payload))

def test_action_count_mismatch_detected():
    a = build_action_alphabet("UNKNOWN")
    with pytest.raises(ValueError, match="ACTION_COUNT_MISMATCH"):
        replace(a, action_count=2)

def test_alphabet_deep_validates_child_actions():
    alphabet = build_action_alphabet("UNKNOWN")
    object.__setattr__(alphabet.actions[0], "action_atom_hash", "c" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH|INVALID_HASH_FORMAT"):
        validate_action_alphabet(alphabet)

def test_invalid_world_family_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_action_alphabet("BAD")

# Spec/receipt

def test_world_adapter_spec_determinism():
    a = build_game_world_archive(_mkzip("doom.zip"))
    hs = {build_world_adapter_spec(a).adapter_spec_hash for _ in range(3)}
    assert len(hs) == 1

def test_world_adapter_spec_binds_archive_family_to_alphabet():
    a = build_game_world_archive(_mkzip("doom.zip"))
    s = build_world_adapter_spec(a)
    assert s.world_family == a.world_family == s.action_alphabet.world_family

def test_adapter_family_mismatch_rejected():
    a = build_game_world_archive(_mkzip("doom.zip"))
    s = build_world_adapter_spec(a)
    bad_alph = build_action_alphabet("UNKNOWN")
    payload = {"archive_manifest_hash": s.archive_manifest_hash, "world_family": s.world_family, "adapter_mode": s.adapter_mode, "action_alphabet": bad_alph.to_dict()}
    with pytest.raises(ValueError, match="ADAPTER_FAMILY_MISMATCH"):
        WorldAdapterSpec(s.archive_manifest_hash, s.world_family, s.adapter_mode, bad_alph, sha256_hex(payload))

def test_invalid_adapter_mode_rejected():
    a = build_game_world_archive(_mkzip("doom.zip"))
    s = build_world_adapter_spec(a)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(s, adapter_mode="RUNTIME")

def test_adapter_spec_hash_tamper_detected():
    s = build_world_adapter_spec(build_game_world_archive(_mkzip("doom.zip")))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(s, adapter_spec_hash="d" * 64)

def test_adapter_spec_malformed_hash_rejected():
    s = build_world_adapter_spec(build_game_world_archive(_mkzip("doom.zip")))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(s, adapter_spec_hash="oops")

def test_world_adapter_contract_receipt_determinism():
    m, r = _mk_manifest_and_receipt()
    hs = {build_world_adapter_contract_receipt(m, r).adapter_contract_receipt_hash for _ in range(3)}
    assert len(hs) == 1

def test_contract_receipt_builds_one_adapter_per_archive():
    m, r = _mk_manifest_and_receipt()
    c = build_world_adapter_contract_receipt(m, r)
    assert c.total_adapters == len(m.archives)

def test_contract_receipt_sorts_adapter_specs():
    m, r = _mk_manifest_and_receipt(("doom.zip", "atari.zip"))
    c = build_world_adapter_contract_receipt(m, r)
    assert c.adapter_specs == tuple(sorted(c.adapter_specs, key=lambda s: (s.archive_manifest_hash, s.adapter_spec_hash)))

def test_corpus_receipt_mismatch_rejected():
    m1, _ = _mk_manifest_and_receipt(("doom.zip",))
    m2, r2 = _mk_manifest_and_receipt(("atari.zip",))
    assert m1.corpus_manifest_hash != m2.corpus_manifest_hash
    with pytest.raises(ValueError, match="CORPUS_RECEIPT_MISMATCH"):
        build_world_adapter_contract_receipt(m1, r2)

def test_duplicate_adapter_rejected():
    m, r = _mk_manifest_and_receipt(("doom.zip",))
    s = build_world_adapter_spec(m.archives[0])
    payload = {"corpus_manifest_hash": m.corpus_manifest_hash, "intake_receipt_hash": r.receipt_hash, "adapter_specs": [s.to_dict(), s.to_dict()], "total_adapters": 2}
    with pytest.raises(ValueError, match="DUPLICATE_ADAPTER"):
        WorldAdapterContractReceipt(m.corpus_manifest_hash, r.receipt_hash, (s, s), 2, sha256_hex(payload))

def test_adapter_count_mismatch_detected():
    m, r = _mk_manifest_and_receipt(("doom.zip",))
    c = build_world_adapter_contract_receipt(m, r)
    with pytest.raises(ValueError, match="ADAPTER_COUNT_MISMATCH"):
        replace(c, total_adapters=3)

def test_contract_deep_validates_adapter_specs():
    m, r = _mk_manifest_and_receipt(("doom.zip",))
    c = build_world_adapter_contract_receipt(m, r)
    orig = c.adapter_specs[0]
    tampered = {"archive_manifest_hash": orig.archive_manifest_hash, "world_family": orig.world_family, "adapter_mode": orig.adapter_mode, "action_alphabet": orig.action_alphabet, "adapter_spec_hash": "e" * 64}
    payload = {"corpus_manifest_hash": m.corpus_manifest_hash, "intake_receipt_hash": r.receipt_hash, "adapter_specs": [{**orig.to_dict(), "adapter_spec_hash": "e" * 64}], "total_adapters": 1}
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        WorldAdapterContractReceipt(m.corpus_manifest_hash, r.receipt_hash, (WorldAdapterSpec(**tampered),), 1, sha256_hex(payload))

def test_contract_receipt_hash_tamper_detected():
    c = build_world_adapter_contract_receipt(*_mk_manifest_and_receipt(("doom.zip",)))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(c, adapter_contract_receipt_hash="f" * 64)

def test_contract_receipt_malformed_hash_rejected():
    c = build_world_adapter_contract_receipt(*_mk_manifest_and_receipt(("doom.zip",)))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(c, adapter_contract_receipt_hash="bad")

def test_no_game_execution_or_import_behavior():
    forbidden = {"run_game","execute_action","step_world","import_world","load_world","load_module_from_zip","observe_world"}
    for n in forbidden:
        assert n not in globals()
    src = Path("src/qec/analysis/game_world_adapter_contract.py").read_text(encoding="utf-8")
    for needle in ["zipfile.ZipFile", ".extract(", ".extractall(", "importlib", "__import__(", "subprocess", "exec(", "eval(", "pygame", "gym", "universe"]:
        assert needle not in src

def test_no_observation_or_episode_trace_artifacts():
    for n in ["ObservationSnapshot", "EpisodeTrace", "StrategyProbeReceipt", "ChaosReplayVerdict"]:
        assert n not in globals()

def test_artifacts_are_frozen():
    a = build_action_atom("NO_OP", "META", ())
    with pytest.raises(FrozenInstanceError):
        a.action_code = "X"

def test_same_process_determinism():
    a1 = build_action_alphabet("UNKNOWN")
    a2 = build_action_alphabet("UNKNOWN")
    assert a1.to_canonical_json() == a2.to_canonical_json()
    assert a1.to_canonical_bytes() == a2.to_canonical_bytes()

def test_validators_reject_non_objects():
    for fn in [validate_action_atom, validate_action_alphabet, validate_world_adapter_spec, validate_world_adapter_contract_receipt]:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            fn(object())
