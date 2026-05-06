from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import zipfile

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.game_world_adapter_contract import build_action_atom, build_world_adapter_contract_receipt
from qec.analysis.game_world_episode_trace import (
    EpisodeTrace,
    _episode_step_payload,
    _episode_trace_payload,
    _episode_trace_receipt_payload,
    _MAX_EPISODE_STEPS,
    build_episode_step,
    build_episode_trace,
    build_episode_trace_receipt,
    validate_episode_step,
    validate_episode_step_with_adapter,
    validate_episode_trace,
    validate_episode_trace_receipt,
    validate_episode_trace_receipt_with_adapter,
    validate_episode_trace_with_adapter,
)
from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_observation_snapshot import build_observation_channel_spec, build_observation_snapshot_receipt


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


def _step(c, s, idx, txt="x", terminal=False):
    obs = build_observation_snapshot_receipt(c, s, build_observation_channel_spec("TEXT_EVENT"), idx, txt)
    return build_episode_step(c, s, idx, obs, s.action_alphabet.actions[0], terminal)


def test_episode_step_rules(adapter_ctx, tmp_path):
    c, s = adapter_ctx
    a = _step(c, s, 0)
    b = _step(c, s, 0)
    assert a.episode_step_hash == b.episode_step_hash
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(a, episode_step_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(a, episode_step_hash="a" * 64)
    with pytest.raises(ValueError, match="STEP_INDEX_OUT_OF_BOUNDS"):
        build_episode_step(c, s, -1, a.observation_snapshot_receipt, a.action_atom)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_episode_step(c, s, True, a.observation_snapshot_receipt, a.action_atom)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_episode_step(c, s, 1, a.observation_snapshot_receipt, a.action_atom, terminal_flag=1)
    with pytest.raises(FrozenInstanceError):
        a.step_index = 2
    fake = build_action_atom("NOPE", "META", ())
    with pytest.raises(ValueError, match="ACTION_NOT_IN_ALPHABET"):
        build_episode_step(c, s, 2, a.observation_snapshot_receipt, fake)
    m2 = build_game_world_corpus_manifest([_mkzip(tmp_path, "atari.zip")])
    r2 = build_game_world_intake_receipt(m2, "b" * 64)
    c2 = build_world_adapter_contract_receipt(m2, r2)
    with pytest.raises(ValueError, match="OBSERVATION_ADAPTER_MISMATCH|ADAPTER"):
        build_episode_step(c2, c2.adapter_specs[0], 0, a.observation_snapshot_receipt, c2.adapter_specs[0].action_alphabet.actions[0])
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_episode_trace_rules(adapter_ctx, monkeypatch):
    c, s = adapter_ctx
    s0, s1 = _step(c, s, 0), _step(c, s, 1)
    t1 = build_episode_trace(c, s, [s0, s1])
    t2 = build_episode_trace(c, s, [s0, s1])
    assert t1.episode_trace_hash == t2.episode_trace_hash
    t3 = build_episode_trace(c, s, [s1, s0])
    assert [x.step_index for x in t3.episode_steps] == [0, 1]
    with pytest.raises(ValueError, match="STEP_ORDER_MISMATCH"):
        EpisodeTrace(c.adapter_contract_receipt_hash, s.adapter_spec_hash, (s1, s0), 2, None, sha256_hex(_episode_trace_payload(c.adapter_contract_receipt_hash, s.adapter_spec_hash, (s1, s0), 2, None)))
    with pytest.raises(ValueError, match="DUPLICATE_STEP"):
        build_episode_trace(c, s, [s0, replace(s0, episode_step_hash=s0.episode_step_hash)])
    with pytest.raises(ValueError, match="STEP_ORDER_MISMATCH"):
        build_episode_trace(c, s, [_step(c, s, 0), _step(c, s, 2)])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_episode_trace(c, s, [])
    monkeypatch.setattr("qec.analysis.game_world_episode_trace._MAX_EPISODE_STEPS", 1)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_episode_trace(c, s, [s0, s1])
    monkeypatch.setattr("qec.analysis.game_world_episode_trace._MAX_EPISODE_STEPS", _MAX_EPISODE_STEPS)
    with pytest.raises(ValueError, match="STEP_COUNT_MISMATCH"):
        replace(t1, step_count=1)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(t1, episode_trace_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(t1, episode_trace_hash="b" * 64)
    with pytest.raises(ValueError, match="TRACE_ADAPTER_MISMATCH"):
        replace(t1, adapter_contract_receipt_hash="c" * 64)
    with pytest.raises(ValueError, match="TRACE_ADAPTER_MISMATCH"):
        replace(t1, adapter_spec_hash="d" * 64)
    with pytest.raises(FrozenInstanceError):
        t1.step_count = 3


def test_terminal_rules(adapter_ctx):
    c, s = adapter_ctx
    s0, s1 = _step(c, s, 0), _step(c, s, 1, terminal=True)
    t = build_episode_trace(c, s, [s0, s1])
    assert t.terminal_step_index == 1
    nt = build_episode_trace(c, s, [s0])
    assert nt.terminal_step_index is None
    with pytest.raises(ValueError, match="MULTIPLE_TERMINAL_STEPS"):
        build_episode_trace(c, s, [_step(c, s, 0, terminal=True), _step(c, s, 1, terminal=True)])
    with pytest.raises(ValueError, match="POST_TERMINAL_STEP"):
        build_episode_trace(c, s, [_step(c, s, 0, terminal=True), _step(c, s, 1)])
    with pytest.raises(ValueError, match="TERMINAL_STEP_MISMATCH"):
        replace(t, terminal_step_index=0)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(t, terminal_step_index=True)


def test_trace_receipt_and_complete_validators(adapter_ctx, tmp_path):
    c, s = adapter_ctx
    tr = build_episode_trace(c, s, [_step(c, s, 0)])
    r1 = build_episode_trace_receipt(c, s, tr)
    r2 = build_episode_trace_receipt(c, s, tr)
    assert r1.episode_trace_receipt_hash == r2.episode_trace_receipt_hash
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(r1, episode_trace_receipt_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(r1, episode_trace_receipt_hash="e" * 64)
    with pytest.raises(ValueError, match="STEP_COUNT_MISMATCH"):
        replace(tr, step_count=2, episode_trace_hash=sha256_hex(_episode_trace_payload(tr.adapter_contract_receipt_hash, tr.adapter_spec_hash, tr.episode_steps, 2, tr.terminal_step_index)))
    m2 = build_game_world_corpus_manifest([_mkzip(tmp_path, "atari.zip")])
    rint2 = build_game_world_intake_receipt(m2, "b" * 64)
    c2 = build_world_adapter_contract_receipt(m2, rint2)
    with pytest.raises(ValueError, match="TRACE_ADAPTER_MISMATCH|ADAPTER"):
        validate_episode_trace_receipt_with_adapter(r1, c2, c2.adapter_specs[0])
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()
    assert validate_episode_step_with_adapter(tr.episode_steps[0], c, s)
    assert validate_episode_trace_with_adapter(tr, c, s)
    assert validate_episode_trace_receipt_with_adapter(r1, c, s)
    bad_action = build_action_atom("NOPE", "META", ())
    bad_step = replace(
        tr.episode_steps[0],
        action_atom=bad_action,
        episode_step_hash=sha256_hex(
            _episode_step_payload(
                tr.episode_steps[0].adapter_contract_receipt_hash,
                tr.episode_steps[0].adapter_spec_hash,
                tr.episode_steps[0].step_index,
                tr.episode_steps[0].observation_snapshot_receipt,
                bad_action,
                tr.episode_steps[0].terminal_flag,
            )
        ),
    )
    assert validate_episode_step(bad_step)
    with pytest.raises(ValueError, match="ACTION_NOT_IN_ALPHABET"):
        validate_episode_step_with_adapter(bad_step, c, s)


def test_scope_boundary_scan():
    text = Path("src/qec/analysis/game_world_episode_trace.py").read_text(encoding="utf-8")
    forbidden = [
        "zipfile.ZipFile", ".extract(", ".extractall(", "importlib", "__import__(", "subprocess", "exec(", "eval(", "pygame", "gym", "render", "step_world", "execute_action", "run_game", "play_game", "train_policy", "StrategyProbe", "ChaosReplayVerdict", "GameWorldInteractionReport", "policy", "reward", "score_heuristic",
    ]
    for token in forbidden:
        assert token not in text
