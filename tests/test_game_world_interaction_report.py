from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import zipfile

import pytest

from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt
from qec.analysis.game_world_observation_snapshot import build_observation_channel_spec, build_observation_snapshot_receipt
from qec.analysis.game_world_episode_trace import build_episode_step, build_episode_trace, build_episode_trace_receipt
from qec.analysis.game_world_strategy_probe import build_strategy_probe_receipt
from qec.analysis.game_world_chaos_replay_verdict import build_chaos_replay_verdict_receipt
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.game_world_interaction_report import (
    GameWorldInteractionReport,
    GameWorldInteractionReportReceipt,
    build_game_world_interaction_report,
    build_game_world_interaction_report_receipt,
    get_game_world_interaction_report_sections,
    get_allowed_interaction_report_verdicts,
    validate_game_world_interaction_report,
    validate_game_world_interaction_report_receipt,
    validate_game_world_interaction_report_with_artifacts,
    validate_game_world_interaction_report_receipt_with_artifacts,
)

def _rehash_report(r: GameWorldInteractionReport, **changes) -> GameWorldInteractionReport:
    d=r.to_dict(); d.update(changes); d.pop("game_world_interaction_report_hash",None)
    d["game_world_interaction_report_hash"]=sha256_hex(d)
    d["observation_snapshot_receipt_hashes"]=tuple(d["observation_snapshot_receipt_hashes"]); d["strategy_probe_receipt_hashes"]=tuple(d["strategy_probe_receipt_hashes"])
    return GameWorldInteractionReport(**d)

def _mkzip(tmp_path: Path, name: str, body: str="print(1)") -> str:
    p=tmp_path/name; zi=zipfile.ZipInfo("main.py"); zi.date_time=(2020,1,1,0,0,0)
    with zipfile.ZipFile(p,"w") as zf: zf.writestr(zi, body)
    return str(p)

@pytest.fixture()
def ctx(tmp_path):
    m=build_game_world_corpus_manifest([_mkzip(tmp_path,"a.zip")]); ir=build_game_world_intake_receipt(m,"a"*64); c=build_world_adapter_contract_receipt(m,ir); s=c.adapter_specs[0]
    ch=build_observation_channel_spec("TEXT_EVENT")
    e0=build_episode_step(c,s,0,build_observation_snapshot_receipt(c,s,ch,0,"x"),s.action_alphabet.actions[0],False)
    e1=build_episode_step(c,s,1,build_observation_snapshot_receipt(c,s,ch,1,"y"),s.action_alphabet.actions[0],True)
    tr=build_episode_trace_receipt(c,s,build_episode_trace(c,s,[e0,e1]))
    p1=build_strategy_probe_receipt(c,s,tr,"NO_OP_BASELINE","NO_OP_BASELINE",{})
    p2=build_strategy_probe_receipt(c,s,tr,"TERMINAL_STATE_SCAN","TERMINAL_STATE_SCAN",{})
    cv=build_chaos_replay_verdict_receipt(c,s,tr,c,s,tr,[p1,p2],[p1,p2])
    return ir,c,s,tr,[p1,p2],cv

def test_report_determinism_and_hashes(ctx):
    ir,c,s,tr,probes,cv=ctx
    r1=build_game_world_interaction_report(ir,c,s,tr,probes,cv); r2=build_game_world_interaction_report(ir,c,s,tr,probes,cv)
    assert r1.game_world_interaction_report_hash==r2.game_world_interaction_report_hash and r1.to_canonical_json()==r2.to_canonical_json() and r1.to_canonical_bytes()==r2.to_canonical_bytes()
    rr1=build_game_world_interaction_report_receipt(ir,c,s,tr,probes,cv); rr2=build_game_world_interaction_report_receipt(ir,c,s,tr,probes,cv)
    assert rr1.game_world_interaction_report_receipt_hash==rr2.game_world_interaction_report_receipt_hash
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): validate_game_world_interaction_report(replace(r1,game_world_interaction_report_hash="bad"))
    with pytest.raises(ValueError,match="HASH_MISMATCH"): validate_game_world_interaction_report(replace(r1,game_world_interaction_report_hash="a"*64))
    with pytest.raises(ValueError,match="INVALID_INTERACTION_REPORT_VERDICT"): validate_game_world_interaction_report(replace(r1,final_report_verdict="BAD"))
    with pytest.raises(ValueError,match="REPORT_VERDICT_MISMATCH"): validate_game_world_interaction_report_with_artifacts(_rehash_report(r1,final_report_verdict="INTERACTION_INVALID"),ir,c,s,tr,probes,cv)
    with pytest.raises(FrozenInstanceError): r1.final_report_verdict="X"

def test_sections_and_complete_validator(ctx):
    ir,c,s,tr,probes,cv=ctx; r=build_game_world_interaction_report(ir,c,s,tr,probes,cv)
    assert get_game_world_interaction_report_sections()==("source_intake","adapter_contract","observation_summary","episode_trace_summary","strategy_probe_summary","chaos_replay_summary","scope_boundary","failure_summary")
    for n in get_game_world_interaction_report_sections(): assert isinstance(getattr(r,n),str) and json.loads(getattr(r,n)) is not None
    with pytest.raises(ValueError,match="INVALID_REPORT_SECTION"): validate_game_world_interaction_report(replace(r,source_intake='{"b":1,"a":2}'))
    with pytest.raises(ValueError,match="INVALID_REPORT_SECTION"): validate_game_world_interaction_report(replace(r,source_intake='{"a":'))
    with pytest.raises(ValueError,match="REPORT_SECTION_TOO_LARGE"): validate_game_world_interaction_report(replace(r,source_intake='"'+('a'*20000)+'"'))
    for k,e in (("source_intake","INTAKE_REPORT_MISMATCH"),("adapter_contract","ADAPTER_REPORT_MISMATCH"),("observation_summary","OBSERVATION_REPORT_MISMATCH"),("episode_trace_summary","EPISODE_TRACE_REPORT_MISMATCH"),("strategy_probe_summary","STRATEGY_PROBE_REPORT_MISMATCH"),("chaos_replay_summary","CHAOS_REPLAY_REPORT_MISMATCH"),("scope_boundary","REPORT_SECTION_MISMATCH"),("failure_summary","REPORT_SECTION_MISMATCH")):
        with pytest.raises(ValueError,match=e): validate_game_world_interaction_report_with_artifacts(_rehash_report(r,**{k:'{}'}),ir,c,s,tr,probes,cv)

def test_verdict_mapping_and_chain_binding(ctx,tmp_path):
    ir,c,s,tr,probes,cv=ctx
    assert build_game_world_interaction_report(ir,c,s,tr,probes,cv).final_report_verdict=="INTERACTION_REPLAY_CLEAN"
    iv=build_chaos_replay_verdict_receipt(c,s,tr,None,None,None,probes,[])
    assert build_game_world_interaction_report(ir,c,s,tr,probes,iv).final_report_verdict=="INTERACTION_INCOMPLETE"
    m2=build_game_world_corpus_manifest([_mkzip(tmp_path,"b.zip","print(2)")]); ir2=build_game_world_intake_receipt(m2,"b"*64); c2=build_world_adapter_contract_receipt(m2,ir2); s2=c2.adapter_specs[0]
    ch=build_observation_channel_spec("TEXT_EVENT")
    t2=build_episode_trace_receipt(c2,s2,build_episode_trace(c2,s2,[build_episode_step(c2,s2,0,build_observation_snapshot_receipt(c2,s2,ch,0,"x"),s2.action_alphabet.actions[0],True)]))
    p2=build_strategy_probe_receipt(c2,s2,t2,"NO_OP_BASELINE","NO_OP_BASELINE",{})
    dv=build_chaos_replay_verdict_receipt(c,s,tr,c2,s2,t2,probes,[p2]); assert build_game_world_interaction_report(ir,c,s,tr,probes,dv).final_report_verdict in {"INTERACTION_REPLAY_DRIFTED","INTERACTION_INCOMPLETE"}
    with pytest.raises(ValueError): build_game_world_interaction_report(ir2,c,s,tr,probes,cv)
    with pytest.raises(ValueError): build_game_world_interaction_report(ir,c2,s,tr,probes,cv)
    with pytest.raises(ValueError): build_game_world_interaction_report(ir,c,s2,tr,probes,cv)
    with pytest.raises(ValueError): build_game_world_interaction_report(ir,c,s,t2,probes,cv)
    with pytest.raises(ValueError): build_game_world_interaction_report(ir,c,s,tr,[p2],cv)

def test_failure_summary_receipt_and_scope(ctx):
    ir,c,s,tr,probes,cv=ctx
    r=build_game_world_interaction_report(ir,c,s,tr,probes,cv); fs=json.loads(r.failure_summary); assert fs["failure_count"]==0 and not fs["incomplete_replay"]
    iv=build_chaos_replay_verdict_receipt(c,s,tr,None,None,None,probes,[])
    ifs=json.loads(build_game_world_interaction_report(ir,c,s,tr,probes,iv).failure_summary); assert ifs["incomplete_replay"]
    rr=build_game_world_interaction_report_receipt(ir,c,s,tr,probes,cv); assert validate_game_world_interaction_report_receipt_with_artifacts(rr,ir,c,s,tr,probes,cv)
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): validate_game_world_interaction_report_receipt(replace(rr,game_world_interaction_report_receipt_hash="bad"))
    with pytest.raises(ValueError,match="HASH_MISMATCH"): validate_game_world_interaction_report_receipt(replace(rr,game_world_interaction_report_receipt_hash="a"*64))
    with pytest.raises(ValueError): validate_game_world_interaction_report_receipt(replace(rr,game_world_interaction_report=replace(r,source_intake='{}')))
    with pytest.raises(ValueError): validate_game_world_interaction_report_receipt_with_artifacts(replace(rr,game_world_interaction_report=replace(r,source_intake='{}')),ir,c,s,tr,probes,cv)
    with pytest.raises(FrozenInstanceError): rr.game_world_interaction_report_receipt_hash="x"
    text=Path("src/qec/analysis/game_world_interaction_report.py").read_text(encoding="utf-8")
    for t in ["zipfile.ZipFile",".extract(",".extractall(","importlib","__import__(","subprocess","exec(","eval(","pygame","gym","step_world","execute_action","run_game","play_game","train_policy","learned_policy","neural","fuzzy","tolerance","repair_replay","instability_metric","probability","best_action","optimal_action","reward","score_heuristic","PerturbationContract","SubstrateContract","RecursiveProofReceipt","RealityLoopProofReceipt","GlobalTruthReceipt"]: assert t not in text
