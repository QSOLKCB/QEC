from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import zipfile

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt
import qec.analysis.game_world_observation_snapshot as gwos
from qec.analysis.game_world_observation_snapshot import *


def _mkzip(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("main.py", "print(1)")
    return str(p)


@pytest.fixture()
def adapter_ctx(tmp_path):
    m = build_game_world_corpus_manifest([_mkzip(tmp_path, "doom.zip")])
    r = build_game_world_intake_receipt(m, "a" * 64)
    c = build_world_adapter_contract_receipt(m, r)
    return c, c.adapter_specs[0]


def test_channel_spec_basics():
    s = build_observation_channel_spec("TEXT_EVENT", "EVENTS", 1024)
    assert s.observation_channel_hash == build_observation_channel_spec("TEXT_EVENT", "EVENTS", 1024).observation_channel_hash
    assert isinstance(get_allowed_observation_channel_types(), frozenset)
    with pytest.raises(ValueError, match="INVALID_CHANNEL_TYPE"): build_observation_channel_spec("BAD")
    with pytest.raises(ValueError, match="INVALID_INPUT"): build_observation_channel_spec("TEXT_EVENT", "bad")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): replace(s, observation_channel_hash="A" * 64)
    with pytest.raises(FrozenInstanceError): s.channel_label = "X"


def test_snapshot_canonical_and_tamper(adapter_ctx):
    c, s = adapter_ctx
    ch = build_observation_channel_spec("SYMBOLIC_STATE")
    a = build_observation_snapshot(c, s, ch, 1, {"b": 2, "a": 1})
    b = build_observation_snapshot(c, s, ch, 1, {"a": 1, "b": 2})
    assert a.canonical_observation_payload == b.canonical_observation_payload
    assert a.observation_snapshot_hash == b.observation_snapshot_hash
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"):
        replace(a, canonical_observation_payload="{not-json", observation_payload_hash=sha256_hex("{not-json"), observation_snapshot_hash=sha256_hex(gwos._observation_snapshot_payload(a.adapter_contract_receipt_hash, a.adapter_spec_hash, a.observation_index, a.observation_channel, "{not-json", sha256_hex("{not-json"))))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(a, canonical_observation_payload=canonical_json({"z": 9}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(a, observation_index=5)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(a, observation_payload_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(a, observation_snapshot_hash="b" * 64)


def test_payload_channel_edges(adapter_ctx):
    c, s = adapter_ctx
    assert build_observation_snapshot(c, s, build_observation_channel_spec("PIXEL_HASH"), 0, "a" * 64)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): build_observation_snapshot(c, s, build_observation_channel_spec("PIXEL_HASH"), 0, "A" * 64)
    t = build_observation_snapshot(c, s, build_observation_channel_spec("TEXT_EVENT"), 0, "  Hi  ")
    assert json.loads(t.canonical_observation_payload) == "  Hi  "
    with pytest.raises(ValueError, match="PAYLOAD_TOO_LARGE"): build_observation_snapshot(c, s, build_observation_channel_spec("TEXT_EVENT", max_payload_bytes=4), 0, "abcdef")
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("SCORE_VALUE"), 0, True)
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("SCORE_VALUE"), 0, 1.0)
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("POSITION_VECTOR"), 0, (1, 2))
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("POSITION_VECTOR"), 0, [1, True])
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("POSITION_VECTOR"), 0, [1, 1.2])


def test_action_mask_and_adapter_validators(adapter_ctx, tmp_path):
    c, s = adapter_ctx
    ch = build_observation_channel_spec("ACTION_MASK")
    codes = [a.action_code for a in s.action_alphabet.actions]
    snap = build_observation_snapshot(c, s, ch, 0, codes)
    rec = build_observation_snapshot_receipt(c, s, ch, 0, codes)
    st = build_observation_snapshot_set(c, s, [rec])
    assert validate_spec_in_contract(c, s)
    assert validate_observation_snapshot_with_adapter(snap, c, s)
    assert validate_observation_snapshot_receipt_with_adapter(rec, c, s)
    assert validate_observation_snapshot_set_with_adapter(st, c, s)
    m2 = build_game_world_corpus_manifest([_mkzip(tmp_path, "atari.zip")])
    r2 = build_game_world_intake_receipt(m2, "b" * 64)
    c2 = build_world_adapter_contract_receipt(m2, r2)
    with pytest.raises(ValueError, match="ADAPTER_SPEC_NOT_IN_CONTRACT"): validate_spec_in_contract(c2, s)
    if len(codes) > 0:
        bad = ["NOPE"]
        cp = canonical_json(bad); ph = sha256_hex(cp)
        s_bad = ObservationSnapshot(c.adapter_contract_receipt_hash, s.adapter_spec_hash, 0, ch, cp, ph, sha256_hex(gwos._observation_snapshot_payload(c.adapter_contract_receipt_hash, s.adapter_spec_hash, 0, ch, cp, ph)))
        assert validate_observation_snapshot(s_bad)
        with pytest.raises(ValueError, match="ACTION_MASK_UNKNOWN_ACTION"): validate_observation_snapshot_with_adapter(s_bad, c, s)
    if len(codes) > 1:
        oo = list(reversed(codes)); cp = canonical_json(oo); ph = sha256_hex(cp)
        s_oo = ObservationSnapshot(c.adapter_contract_receipt_hash, s.adapter_spec_hash, 0, ch, cp, ph, sha256_hex(gwos._observation_snapshot_payload(c.adapter_contract_receipt_hash, s.adapter_spec_hash, 0, ch, cp, ph)))
        with pytest.raises(ValueError, match="ACTION_MASK_ORDER_MISMATCH"): validate_observation_snapshot_with_adapter(s_oo, c, s)


def test_tuple_rejection_and_set_rules(adapter_ctx, monkeypatch):
    c, s = adapter_ctx
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("SYMBOLIC_STATE"), 0, {"x": (1, 2)})
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("UNKNOWN"), 0, (1, 2))
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_PAYLOAD"): build_observation_snapshot(c, s, build_observation_channel_spec("ACTION_MASK"), 0, ("A",))
    t = build_observation_channel_spec("TEXT_EVENT")
    r0 = build_observation_snapshot_receipt(c, s, t, 0, "x")
    r1 = build_observation_snapshot_receipt(c, s, t, 1, "y")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ObservationSnapshotSet(c.adapter_contract_receipt_hash, s.adapter_spec_hash, (r1, r0), 2, sha256_hex(gwos._observation_snapshot_set_payload(c.adapter_contract_receipt_hash, s.adapter_spec_hash, (r1, r0), 2)))
    with pytest.raises(ValueError, match="DUPLICATE_OBSERVATION"):
        build_observation_snapshot_set(c, s, [r0, build_observation_snapshot_receipt(c, s, t, 0, "z")])
    monkeypatch.setattr("qec.analysis.game_world_observation_snapshot._MAX_OBSERVATION_SNAPSHOTS", 1)
    with pytest.raises(ValueError, match="INVALID_INPUT"): build_observation_snapshot_set(c, s, [r0, r1])


def test_receipt_set_and_binding_tamper(adapter_ctx):
    c, s = adapter_ctx
    t = build_observation_channel_spec("TEXT_EVENT")
    r = build_observation_snapshot_receipt(c, s, t, 0, "x")
    st = build_observation_snapshot_set(c, s, [r])
    assert validate_observation_snapshot_set(st)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): replace(st, observation_snapshot_set_hash="SHORT")
    with pytest.raises(ValueError, match="HASH_MISMATCH"): replace(st, observation_snapshot_set_hash="c" * 64)
    bad_snap = replace(r.observation_snapshot, adapter_contract_receipt_hash="d" * 64, observation_snapshot_hash=sha256_hex(gwos._observation_snapshot_payload("d" * 64, r.observation_snapshot.adapter_spec_hash, r.observation_snapshot.observation_index, r.observation_snapshot.observation_channel, r.observation_snapshot.canonical_observation_payload, r.observation_snapshot.observation_payload_hash)))
    with pytest.raises(ValueError, match="SNAPSHOT_ADAPTER_MISMATCH"):
        replace(r, observation_snapshot=bad_snap, observation_snapshot_receipt_hash=sha256_hex(gwos._observation_snapshot_receipt_payload(r.adapter_contract_receipt_hash, r.adapter_spec_hash, bad_snap)))


def test_scope_boundary_scan():
    source_path = Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_observation_snapshot.py"
    src = source_path.read_text(encoding="utf-8")
    for needle in ["zipfile.ZipFile", ".extract(", ".extractall(", "importlib", "__import__(", "subprocess", "exec(", "eval(", "pygame", "gym", "render", "step_world", "execute_action", "run_game", "play_game", "train_policy", "StrategyProbe", "EpisodeTrace", "ChaosReplayVerdict"]:
        assert needle not in src
