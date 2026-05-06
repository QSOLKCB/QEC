from dataclasses import replace
from pathlib import Path
import zipfile

import pytest

from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt, validate_game_world_intake_receipt
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt, validate_world_adapter_contract_receipt
from qec.analysis.game_world_observation_snapshot import build_observation_channel_spec, build_observation_snapshot_receipt, validate_observation_snapshot_receipt_with_adapter
from qec.analysis.game_world_episode_trace import build_episode_step, build_episode_trace, build_episode_trace_receipt, validate_episode_trace_receipt_with_adapter
from qec.analysis.game_world_strategy_probe import build_strategy_probe_receipt, validate_strategy_probe_receipt_with_adapter


def _mkzip(tmp_path: Path, name: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("main.py", "print(1)")
    return str(p)


def _bundle(tmp_path: Path, name: str):
    m = build_game_world_corpus_manifest([_mkzip(tmp_path, name)])
    i = build_game_world_intake_receipt(m, "a" * 64)
    c = build_world_adapter_contract_receipt(m, i)
    s = c.adapter_specs[0]
    o = build_observation_snapshot_receipt(c, s, build_observation_channel_spec("TEXT_EVENT"), 0, "x")
    step = build_episode_step(c, s, 0, o, s.action_alphabet.actions[0], False)
    tr = build_episode_trace_receipt(c, s, build_episode_trace(c, s, [step]))
    pr = build_strategy_probe_receipt(c, s, tr, "NO_OP_BASELINE", "NO_OP_BASELINE")
    return i, c, o, tr, pr, s


def test_v156x_validation_matrix(tmp_path):
    i, c, o, tr, pr, s = _bundle(tmp_path / "a", "doom.zip")
    _, c2, _, _, _, s2 = _bundle(tmp_path / "b", "atari.zip")

    assert validate_game_world_intake_receipt(i)
    assert validate_world_adapter_contract_receipt(c)
    assert validate_observation_snapshot_receipt_with_adapter(o, c, s)
    assert validate_episode_trace_receipt_with_adapter(tr, c, s)
    assert validate_strategy_probe_receipt_with_adapter(pr, c, s, tr)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        replace(i, receipt_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        replace(c, adapter_contract_receipt_hash="a" * 64)
    with pytest.raises(ValueError, match="MISMATCH|ADAPTER"):
        validate_observation_snapshot_receipt_with_adapter(o, c2, s2)
    with pytest.raises(ValueError, match="MISMATCH|ADAPTER"):
        validate_episode_trace_receipt_with_adapter(tr, c2, s2)
    with pytest.raises(ValueError, match="MISMATCH|ADAPTER"):
        validate_strategy_probe_receipt_with_adapter(pr, c2, s2, tr)
