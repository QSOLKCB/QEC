from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json, zipfile

import pytest

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt
from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_observation_snapshot import build_observation_channel_spec, build_observation_snapshot_receipt
from qec.analysis.game_world_episode_trace import build_episode_step, build_episode_trace, build_episode_trace_receipt
from qec.analysis.game_world_strategy_probe import (
    build_strategy_probe_spec, build_strategy_probe_result, build_strategy_probe_receipt,
    validate_strategy_probe_receipt_with_adapter, validate_strategy_probe_result_with_adapter,
    validate_strategy_probe_spec_with_adapter, get_allowed_strategy_probe_types
)

def _mkzip(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf: zf.writestr("main.py", "print(1)")
    return str(p)

@pytest.fixture()
def ctx(tmp_path):
    m=build_game_world_corpus_manifest([_mkzip(tmp_path,"doom.zip")]); ir=build_game_world_intake_receipt(m,"a"*64); c=build_world_adapter_contract_receipt(m,ir); s=c.adapter_specs[0]
    ch=build_observation_channel_spec("TEXT_EVENT")
    s0=build_episode_step(c,s,0,build_observation_snapshot_receipt(c,s,ch,0,"x"),s.action_alphabet.actions[0],False)
    s1=build_episode_step(c,s,1,build_observation_snapshot_receipt(c,s,ch,1,"y"),s.action_alphabet.actions[1],True)
    tr=build_episode_trace(c,s,[s0,s1]); rr=build_episode_trace_receipt(c,s,tr)
    return c,s,rr

def test_spec_result_receipt_determinism(ctx):
    c,s,tr=ctx
    sp1=build_strategy_probe_spec(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE")
    sp2=build_strategy_probe_spec(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE")
    assert sp1.strategy_probe_spec_hash==sp2.strategy_probe_spec_hash
    rs1=build_strategy_probe_result(c,s,tr,sp1); rs2=build_strategy_probe_result(c,s,tr,sp1)
    assert rs1.strategy_probe_result_hash==rs2.strategy_probe_result_hash
    rc1=build_strategy_probe_receipt(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE")
    rc2=build_strategy_probe_receipt(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE")
    assert rc1.strategy_probe_receipt_hash==rc2.strategy_probe_receipt_hash

def test_probe_types_and_semantics(ctx):
    c,s,tr=ctx
    for t in get_allowed_strategy_probe_types():
        p={} if t not in {"REPEAT_ACTION_STABILITY","UNKNOWN_ACTION_REJECTION","TRACE_DIVERGENCE_SCAN"} else ({"action_code":s.action_alphabet.actions[0].action_code} if t!="TRACE_DIVERGENCE_SCAN" else {"expected_episode_trace_hash":tr.episode_trace.episode_trace_hash})
        receipt=build_strategy_probe_receipt(c,s,tr,t,t,p)
        assert validate_strategy_probe_receipt_with_adapter(receipt,c,s,tr)
        payload=json.loads(receipt.strategy_probe_result.canonical_result_payload)
        assert "best_action" not in payload and "optimal_action" not in payload

def test_errors_and_mutation(ctx):
    c,s,tr=ctx
    with pytest.raises(ValueError, match="INVALID_PROBE_TYPE"): build_strategy_probe_spec(c,s,tr,"BAD","BAD")
    with pytest.raises(ValueError): build_strategy_probe_spec(c,s,tr,"NO_OP_BASELINE","bad")
    with pytest.raises(ValueError, match="ACTION_NOT_IN_ALPHABET"):
        build_strategy_probe_spec(c,s,tr,"REPEAT_ACTION_STABILITY","REPEAT_ACTION_STABILITY",{"action_code":"NOPE"})
    sp=build_strategy_probe_spec(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE")
    with pytest.raises(FrozenInstanceError): sp.probe_type="X"
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): replace(sp, strategy_probe_spec_hash="BAD")
    with pytest.raises(ValueError, match="HASH_MISMATCH"): replace(sp, strategy_probe_spec_hash="a"*64)


def test_tamper_and_binding(ctx,tmp_path):
    c,s,tr=ctx
    sp=build_strategy_probe_spec(c,s,tr,"TRACE_DIVERGENCE_SCAN","TRACE_DIVERGENCE_SCAN",{"expected_episode_trace_hash":"b"*64})
    rs=build_strategy_probe_result(c,s,tr,sp)
    with pytest.raises(ValueError):
        tam=replace(rs, canonical_result_payload=canonical_json({"x":1}))
        validate_strategy_probe_result_with_adapter(tam,sp,c,s,tr)
    m2=build_game_world_corpus_manifest([_mkzip(tmp_path,"atari.zip")]); ir2=build_game_world_intake_receipt(m2,"b"*64); c2=build_world_adapter_contract_receipt(m2,ir2)
    with pytest.raises(ValueError): validate_strategy_probe_spec_with_adapter(sp,c2,c2.adapter_specs[0],tr)

def test_scope_boundary_scan():
    text=Path("src/qec/analysis/game_world_strategy_probe.py").read_text(encoding="utf-8")
    forbidden=["zipfile.ZipFile",".extract(",".extractall(","importlib","__import__(","subprocess","exec(","eval(","pygame","gym","render","step_world","execute_action","run_game","play_game","train_policy","learned_policy","neural","probability","probabilistic","best_action","optimal_action","reward","score_heuristic","ChaosReplayVerdict","GameWorldInteractionReport"]
    for t in forbidden: assert t not in text
