from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import zipfile
import pytest

from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt
from qec.analysis.game_world_observation_snapshot import build_observation_channel_spec, build_observation_snapshot_receipt
from qec.analysis.game_world_episode_trace import build_episode_step, build_episode_trace, build_episode_trace_receipt
from qec.analysis.game_world_strategy_probe import build_strategy_probe_receipt
from qec.analysis.game_world_chaos_replay_verdict import (
    build_replay_comparison_point,
    validate_replay_comparison_point,
    build_chaos_replay_verdict,
    validate_chaos_replay_verdict,
    validate_chaos_replay_verdict_with_artifacts,
    build_chaos_replay_verdict_receipt,
    validate_chaos_replay_verdict_receipt,
    validate_chaos_replay_verdict_receipt_with_artifacts,
)

def _mkzip(tmp_path: Path, name: str, body: str="print(1)") -> str:
    p=tmp_path/name
    zi=zipfile.ZipInfo("main.py"); zi.date_time=(2020,1,1,0,0,0)
    with zipfile.ZipFile(p,"w") as zf: zf.writestr(zi, body)
    return str(p)

@pytest.fixture()
def ctx(tmp_path):
    m=build_game_world_corpus_manifest([_mkzip(tmp_path,"a.zip")]); ir=build_game_world_intake_receipt(m,"a"*64); c=build_world_adapter_contract_receipt(m,ir); s=c.adapter_specs[0]; ch=build_observation_channel_spec("TEXT_EVENT")
    e0=build_episode_step(c,s,0,build_observation_snapshot_receipt(c,s,ch,0,"x"),s.action_alphabet.actions[0],False)
    e1=build_episode_step(c,s,1,build_observation_snapshot_receipt(c,s,ch,1,"y"),s.action_alphabet.actions[0],True)
    tr=build_episode_trace_receipt(c,s,build_episode_trace(c,s,[e0,e1]))
    p=build_strategy_probe_receipt(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE",{})
    return c,s,tr,[p],m

def test_point_basics(ctx):
    h=ctx[0].adapter_contract_receipt_hash
    p1=build_replay_comparison_point(0,"ADAPTER_CONTRACT","ADAPTER_CONTRACT",h,h); p2=build_replay_comparison_point(0,"ADAPTER_CONTRACT","ADAPTER_CONTRACT",h,h)
    assert p1.replay_comparison_point_hash==p2.replay_comparison_point_hash and not p1.drifted
    assert build_replay_comparison_point(1,"ACTION_SELECTION","STEP_000000_ACTION","a"*64,"b"*64).drift_class=="ACTION_DRIFT"
    assert build_replay_comparison_point(2,"EPISODE_TRACE","EPISODE_TRACE","a"*64,None).drift_class=="INCOMPLETE_REPLAY"
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): build_replay_comparison_point(3,"EPISODE_TRACE","EPISODE_TRACE","bad","a"*64)
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): build_replay_comparison_point(3,"EPISODE_TRACE","EPISODE_TRACE","a"*64,"bad")
    with pytest.raises(ValueError,match="INVALID_COMPARISON_KIND"): build_replay_comparison_point(0,"BAD","X","a"*64,"a"*64)
    with pytest.raises(ValueError): build_replay_comparison_point(0,"EPISODE_TRACE","bad","a"*64,"a"*64)
    with pytest.raises(ValueError): build_replay_comparison_point(True,"EPISODE_TRACE","EPISODE_TRACE","a"*64,"a"*64)
    with pytest.raises(ValueError): build_replay_comparison_point(1_000_001,"EPISODE_TRACE","EPISODE_TRACE","a"*64,"a"*64)
    with pytest.raises(FrozenInstanceError): p1.comparison_label="X"
    assert p1.to_canonical_json()==p2.to_canonical_json() and p1.to_canonical_bytes()==p2.to_canonical_bytes()
    with pytest.raises(ValueError,match="HASH_MISMATCH"): validate_replay_comparison_point(replace(p1,replay_comparison_point_hash="a"*64))

def test_verdict_and_receipt(ctx,tmp_path):
    c,s,tr,probes,m=ctx
    v=build_chaos_replay_verdict(c,s,tr,c,s,tr,probes,probes); assert v.verdict_class=="CHAOS_REPLAY_CLEAN" and v.drift_count==0 and v.comparison_count==len(v.comparison_points)
    assert validate_chaos_replay_verdict(v) and validate_chaos_replay_verdict_with_artifacts(v,c,s,tr,c,s,tr,probes,probes)
    assert build_chaos_replay_verdict(c,s,tr,c,s,tr,probes,probes).chaos_replay_verdict_hash==v.chaos_replay_verdict_hash
    r=build_chaos_replay_verdict_receipt(c,s,tr,c,s,tr,probes,probes)
    assert validate_chaos_replay_verdict_receipt(r) and validate_chaos_replay_verdict_receipt_with_artifacts(r,c,s,tr,c,s,tr,probes,probes)
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): validate_chaos_replay_verdict(replace(v,chaos_replay_verdict_hash="bad"))
    with pytest.raises(ValueError,match="HASH_MISMATCH"): validate_chaos_replay_verdict(replace(v,chaos_replay_verdict_hash="a"*64))
    with pytest.raises(ValueError,match="COMPARISON_COUNT_MISMATCH"): validate_chaos_replay_verdict(replace(v,comparison_count=v.comparison_count+1))
    with pytest.raises(ValueError,match="DRIFT_COUNT_MISMATCH"): validate_chaos_replay_verdict(replace(v,drift_count=v.drift_count+1))
    with pytest.raises(ValueError,match="VERDICT_CLASS_MISMATCH"): validate_chaos_replay_verdict(replace(v,verdict_class="TRACE_DRIFT"))
    with pytest.raises(FrozenInstanceError): r.chaos_replay_verdict_receipt_hash="x"
    with pytest.raises(ValueError,match="HASH_MISMATCH"): validate_chaos_replay_verdict_receipt(replace(r,chaos_replay_verdict_receipt_hash="a"*64))
    with pytest.raises(ValueError): validate_chaos_replay_verdict_with_artifacts(replace(v,comparison_points=tuple(reversed(v.comparison_points))),c,s,tr,c,s,tr,probes,probes)

    m2=build_game_world_corpus_manifest([_mkzip(tmp_path,"b.zip","print(2)")]); ir2=build_game_world_intake_receipt(m2,"b"*64); c2=build_world_adapter_contract_receipt(m2,ir2); s2=c2.adapter_specs[0]
    ch2=build_observation_channel_spec("TEXT_EVENT")
    f0=build_episode_step(c2,s2,0,build_observation_snapshot_receipt(c2,s2,ch2,0,"x"),s2.action_alphabet.actions[0],False)
    f1=build_episode_step(c2,s2,1,build_observation_snapshot_receipt(c2,s2,ch2,1,"y"),s2.action_alphabet.actions[0],True)
    tr2=build_episode_trace_receipt(c2,s2,build_episode_trace(c2,s2,[f0,f1]))
    p2=build_strategy_probe_receipt(c2,s2,tr2,"NO_OP_BASELINE","NO_OP_BASELINE",{})
    ov=build_chaos_replay_verdict(c,s,tr,c2,s2,tr2,probes,[p2]); assert ov.verdict_class=="INTAKE_DRIFT"
    iv=build_chaos_replay_verdict(c,s,tr,None,None,None,probes,[]); assert iv.verdict_class=="INCOMPLETE_REPLAY"

def test_probe_pairing_and_scope(ctx):
    c,s,tr,probes,_=ctx
    p2=build_strategy_probe_receipt(c,s,tr,"TERMINAL_STATE_SCAN","TERMINAL_STATE_SCAN",{})
    v=build_chaos_replay_verdict(c,s,tr,c,s,tr,[probes[0],p2],[probes[0]])
    assert v.verdict_class=="INCOMPLETE_REPLAY"
    with pytest.raises(ValueError,match="PROBE_PAIRING_MISMATCH"): build_chaos_replay_verdict(c,s,tr,c,s,tr,[probes[0],probes[0]],[probes[0]])
    with pytest.raises(ValueError,match="PROBE_PAIRING_MISMATCH"): build_chaos_replay_verdict(c,s,tr,c,s,tr,[probes[0]],[probes[0],probes[0]])
    text=Path("src/qec/analysis/game_world_chaos_replay_verdict.py").read_text(encoding="utf-8")
    for t in ["zipfile.ZipFile",".extract(",".extractall(","importlib","__import__(","subprocess","exec(","eval(","pygame","gym","render","step_world","execute_action","run_game","play_game","train_policy","learned_policy","neural","fuzzy","tolerance","auto_repair","repair_replay","instability_metric","probability","probabilistic","best_action","optimal_action","reward","score_heuristic","GameWorldInteractionReport"]: assert t not in text
