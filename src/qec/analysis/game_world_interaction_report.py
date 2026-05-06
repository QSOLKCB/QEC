from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_adapter_contract import WorldAdapterContractReceipt, WorldAdapterSpec, validate_world_adapter_contract_receipt, validate_world_adapter_spec
from .game_world_chaos_replay_verdict import ChaosReplayVerdictReceipt, validate_chaos_replay_verdict_receipt, validate_chaos_replay_verdict_receipt_with_artifacts
from .game_world_episode_trace import EpisodeTraceReceipt, validate_episode_trace_receipt_with_adapter
from .game_world_intake_contract import GameWorldIntakeReceipt, validate_game_world_intake_receipt
from .game_world_observation_snapshot import validate_spec_in_contract
from .game_world_strategy_probe import StrategyProbeReceipt, validate_strategy_probe_receipt_with_adapter

_ERR_INVALID_INPUT="INVALID_INPUT"; _ERR_INVALID_HASH_FORMAT="INVALID_HASH_FORMAT"; _ERR_HASH_MISMATCH="HASH_MISMATCH"; _ERR_INVALID_REPORT_SECTION="INVALID_REPORT_SECTION"; _ERR_REPORT_SECTION_MISMATCH="REPORT_SECTION_MISMATCH"; _ERR_REPORT_SECTION_TOO_LARGE="REPORT_SECTION_TOO_LARGE"; _ERR_INVALID_INTERACTION_REPORT_VERDICT="INVALID_INTERACTION_REPORT_VERDICT"; _ERR_REPORT_VERDICT_MISMATCH="REPORT_VERDICT_MISMATCH"; _ERR_INTAKE_REPORT_MISMATCH="INTAKE_REPORT_MISMATCH"; _ERR_ADAPTER_REPORT_MISMATCH="ADAPTER_REPORT_MISMATCH"; _ERR_OBSERVATION_REPORT_MISMATCH="OBSERVATION_REPORT_MISMATCH"; _ERR_EPISODE_TRACE_REPORT_MISMATCH="EPISODE_TRACE_REPORT_MISMATCH"; _ERR_STRATEGY_PROBE_REPORT_MISMATCH="STRATEGY_PROBE_REPORT_MISMATCH"; _ERR_CHAOS_REPLAY_REPORT_MISMATCH="CHAOS_REPLAY_REPORT_MISMATCH"; _ERR_DUPLICATE_REPORT_ENTRY="DUPLICATE_REPORT_ENTRY"; _ERR_REPORT_COUNT_MISMATCH="REPORT_COUNT_MISMATCH"; _ERR_INTERACTION_REPORT_RECEIPT_MISMATCH="INTERACTION_REPORT_RECEIPT_MISMATCH"
_MAX_REPORT_SECTION_BYTES=16_384; _MAX_OBSERVATION_RECEIPTS=10_000; _MAX_STRATEGY_PROBE_RECEIPTS=1_000; _MAX_FAILURE_ENTRIES=10_000
_SHA256_RE=re.compile(r"^[0-9a-f]{64}$")
_SECTIONS=("source_intake","adapter_contract","observation_summary","episode_trace_summary","strategy_probe_summary","chaos_replay_summary","scope_boundary","failure_summary")
_ALLOWED_VERDICTS=frozenset({"INTERACTION_REPLAY_CLEAN","INTERACTION_REPLAY_DRIFTED","INTERACTION_INCOMPLETE","INTERACTION_INVALID"})


def get_game_world_interaction_report_sections()->tuple[str,...]: return _SECTIONS

def get_allowed_interaction_report_verdicts()->frozenset[str]: return _ALLOWED_VERDICTS

def _validate_hash(v:object)->None:
    if not isinstance(v,str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)

def _probe_key(r:StrategyProbeReceipt)->tuple[str,str]: return (r.strategy_probe_spec.probe_type,r.strategy_probe_spec.probe_label)

def _report_verdict(chaos_verdict:str)->str:
    if chaos_verdict=="CHAOS_REPLAY_CLEAN": return "INTERACTION_REPLAY_CLEAN"
    if chaos_verdict=="INCOMPLETE_REPLAY": return "INTERACTION_INCOMPLETE"
    if chaos_verdict in {"OBSERVATION_DRIFT","ACTION_DRIFT","TRACE_DRIFT","PROBE_DRIFT","ADAPTER_DRIFT","INTAKE_DRIFT"}: return "INTERACTION_REPLAY_DRIFTED"
    raise ValueError(_ERR_INVALID_INPUT)

def _canonical_section(payload:dict[str,Any])->str:
    s=canonical_json(payload)
    if len(s.encode("utf-8"))>_MAX_REPORT_SECTION_BYTES: raise ValueError(_ERR_REPORT_SECTION_TOO_LARGE)
    return s

def _validate_section(s:object)->dict[str,Any]:
    if not isinstance(s,str): raise ValueError(_ERR_INVALID_REPORT_SECTION)
    if len(s.encode("utf-8"))>_MAX_REPORT_SECTION_BYTES: raise ValueError(_ERR_REPORT_SECTION_TOO_LARGE)
    try: p=json.loads(s)
    except Exception: raise ValueError(_ERR_INVALID_REPORT_SECTION)
    if canonical_json(p)!=s: raise ValueError(_ERR_INVALID_REPORT_SECTION)
    return p

def _game_world_interaction_report_payload(game_world_intake_receipt_hash:str,world_adapter_contract_receipt_hash:str,adapter_spec_hash:str,episode_trace_receipt_hash:str,chaos_replay_verdict_receipt_hash:str,observation_snapshot_receipt_hashes:tuple[str,...],strategy_probe_receipt_hashes:tuple[str,...],source_intake:str,adapter_contract:str,observation_summary:str,episode_trace_summary:str,strategy_probe_summary:str,chaos_replay_summary:str,scope_boundary:str,failure_summary:str,final_report_verdict:str)->dict[str,Any]:
    return {"game_world_intake_receipt_hash":game_world_intake_receipt_hash,"world_adapter_contract_receipt_hash":world_adapter_contract_receipt_hash,"adapter_spec_hash":adapter_spec_hash,"episode_trace_receipt_hash":episode_trace_receipt_hash,"chaos_replay_verdict_receipt_hash":chaos_replay_verdict_receipt_hash,"observation_snapshot_receipt_hashes":list(observation_snapshot_receipt_hashes),"strategy_probe_receipt_hashes":list(strategy_probe_receipt_hashes),"source_intake":source_intake,"adapter_contract":adapter_contract,"observation_summary":observation_summary,"episode_trace_summary":episode_trace_summary,"strategy_probe_summary":strategy_probe_summary,"chaos_replay_summary":chaos_replay_summary,"scope_boundary":scope_boundary,"failure_summary":failure_summary,"final_report_verdict":final_report_verdict}

def _game_world_interaction_report_receipt_payload(game_world_interaction_report:"GameWorldInteractionReport")->dict[str,Any]: return {"game_world_interaction_report":game_world_interaction_report.to_dict()}

@dataclass(frozen=True)
class GameWorldInteractionReport:
    game_world_intake_receipt_hash:str; world_adapter_contract_receipt_hash:str; adapter_spec_hash:str; episode_trace_receipt_hash:str; chaos_replay_verdict_receipt_hash:str; observation_snapshot_receipt_hashes:tuple[str,...]; strategy_probe_receipt_hashes:tuple[str,...]; source_intake:str; adapter_contract:str; observation_summary:str; episode_trace_summary:str; strategy_probe_summary:str; chaos_replay_summary:str; scope_boundary:str; failure_summary:str; final_report_verdict:str; game_world_interaction_report_hash:str
    def __post_init__(self)->None: validate_game_world_interaction_report(self)
    def _hash_payload(self)->dict[str,Any]: return _game_world_interaction_report_payload(self.game_world_intake_receipt_hash,self.world_adapter_contract_receipt_hash,self.adapter_spec_hash,self.episode_trace_receipt_hash,self.chaos_replay_verdict_receipt_hash,self.observation_snapshot_receipt_hashes,self.strategy_probe_receipt_hashes,self.source_intake,self.adapter_contract,self.observation_summary,self.episode_trace_summary,self.strategy_probe_summary,self.chaos_replay_summary,self.scope_boundary,self.failure_summary,self.final_report_verdict)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"game_world_interaction_report_hash":self.game_world_interaction_report_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class GameWorldInteractionReportReceipt:
    game_world_interaction_report:GameWorldInteractionReport; game_world_interaction_report_receipt_hash:str
    def __post_init__(self)->None: validate_game_world_interaction_report_receipt(self)
    def _hash_payload(self)->dict[str,Any]: return _game_world_interaction_report_receipt_payload(self.game_world_interaction_report)
    def to_dict(self)->dict[str,Any]: return {"game_world_interaction_report":self.game_world_interaction_report.to_dict(),"game_world_interaction_report_receipt_hash":self.game_world_interaction_report_receipt_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

def build_game_world_interaction_report(game_world_intake_receipt:GameWorldIntakeReceipt,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...],chaos_replay_verdict_receipt:ChaosReplayVerdictReceipt)->GameWorldInteractionReport:
    validate_game_world_intake_receipt(game_world_intake_receipt); validate_world_adapter_contract_receipt(adapter_contract_receipt); validate_world_adapter_spec(adapter_spec); validate_spec_in_contract(adapter_contract_receipt,adapter_spec)
    if adapter_contract_receipt.intake_receipt_hash!=game_world_intake_receipt.receipt_hash: raise ValueError(_ERR_INTAKE_REPORT_MISMATCH)
    validate_episode_trace_receipt_with_adapter(episode_trace_receipt,adapter_contract_receipt,adapter_spec)
    if len(strategy_probe_receipts)>_MAX_STRATEGY_PROBE_RECEIPTS: raise ValueError(_ERR_INVALID_INPUT)
    for r in strategy_probe_receipts: validate_strategy_probe_receipt_with_adapter(r,adapter_contract_receipt,adapter_spec,episode_trace_receipt)
    validate_chaos_replay_verdict_receipt(chaos_replay_verdict_receipt)
    v=chaos_replay_verdict_receipt.chaos_replay_verdict
    if v.expected_intake_receipt_hash!=game_world_intake_receipt.receipt_hash: raise ValueError(_ERR_INTAKE_REPORT_MISMATCH)
    if v.expected_adapter_contract_receipt_hash!=adapter_contract_receipt.adapter_contract_receipt_hash or v.expected_adapter_spec_hash!=adapter_spec.adapter_spec_hash: raise ValueError(_ERR_ADAPTER_REPORT_MISMATCH)
    if v.expected_episode_trace_receipt_hash!=episode_trace_receipt.episode_trace_receipt_hash: raise ValueError(_ERR_EPISODE_TRACE_REPORT_MISMATCH)
    steps=episode_trace_receipt.episode_trace.episode_steps
    obs=tuple(s.observation_snapshot_receipt.observation_snapshot_receipt_hash for s in steps)
    if len(obs)>_MAX_OBSERVATION_RECEIPTS: raise ValueError(_ERR_REPORT_COUNT_MISMATCH)
    for h in obs: _validate_hash(h)
    ordered=tuple(sorted(strategy_probe_receipts,key=lambda r:(r.strategy_probe_spec.probe_type,r.strategy_probe_spec.probe_label,r.strategy_probe_receipt_hash)))
    keys=set(); probes=[]
    for r in ordered:
        k=_probe_key(r)
        if k in keys: raise ValueError(_ERR_DUPLICATE_REPORT_ENTRY)
        keys.add(k); probes.append(r.strategy_probe_receipt_hash)
    ph=tuple(probes)
    for h in ph: _validate_hash(h)
    ai=adapter_spec.action_alphabet
    src=_canonical_section({"game_world_intake_receipt_hash":game_world_intake_receipt.receipt_hash,"corpus_manifest_hash":game_world_intake_receipt.corpus_manifest_hash,"intake_bound":True,"execution_permitted":False})
    adp=_canonical_section({"world_adapter_contract_receipt_hash":adapter_contract_receipt.adapter_contract_receipt_hash,"adapter_spec_hash":adapter_spec.adapter_spec_hash,"adapter_mode":adapter_spec.adapter_mode,"action_alphabet_hash":ai.action_alphabet_hash,"action_count":len(ai.actions),"action_codes":[a.action_code for a in ai.actions]})
    obs_types=[s.observation_snapshot_receipt.observation_snapshot.observation_channel.channel_type for s in steps]
    obs_sum={"observation_count":len(obs),"observation_snapshot_receipt_hashes":list(obs),"observation_channel_types":obs_types}
    tc=sum(1 for t in obs_types if t=="TERMINAL_FLAG"); amc=sum(1 for t in obs_types if t=="ACTION_MASK")
    if tc: obs_sum["terminal_observation_count"]=tc
    if amc: obs_sum["action_mask_observation_count"]=amc
    obsec=_canonical_section(obs_sum)
    et=_canonical_section({"episode_trace_receipt_hash":episode_trace_receipt.episode_trace_receipt_hash,"step_count":len(steps),"terminal_step_index":episode_trace_receipt.episode_trace.terminal_step_index,"terminal_present":episode_trace_receipt.episode_trace.terminal_step_index is not None,"action_atom_hashes":[s.action_atom.action_atom_hash for s in steps],"action_codes":[s.action_atom.action_code for s in steps]})
    sp=_canonical_section({"strategy_probe_count":len(ph),"strategy_probe_receipt_hashes":list(ph),"probe_types":[r.strategy_probe_spec.probe_type for r in ordered],"probe_labels":[r.strategy_probe_spec.probe_label for r in ordered]})
    points=v.comparison_points
    probe_hashes={p.expected_hash for p in points if p.comparison_kind=="STRATEGY_PROBE" and p.expected_hash is not None}
    for h in ph:
        if h not in probe_hashes: raise ValueError(_ERR_STRATEGY_PROBE_REPORT_MISMATCH)
    drifts=[p for p in points if p.drifted]
    ch=_canonical_section({"chaos_replay_verdict_receipt_hash":chaos_replay_verdict_receipt.chaos_replay_verdict_receipt_hash,"chaos_replay_verdict_hash":chaos_replay_verdict_receipt.chaos_replay_verdict.chaos_replay_verdict_hash,"chaos_replay_verdict_class":chaos_replay_verdict_receipt.chaos_replay_verdict.verdict_class,"comparison_count":len(points),"drift_count":len(drifts),"drift_classes_present":sorted({p.drift_class for p in drifts})})
    sb=_canonical_section({"analysis_layer_only":True,"archive_extraction":False,"auto_repair":False,"dynamic_imports":False,"gameplay_execution":False,"global_truth_claim":False,"policy_optimization":False,"probabilistic_scoring":False,"rendering":False,"runtime_replay_execution":False,"substrate_claim":False,"v157_perturbation":False})
    dcc={}
    for p in drifts: dcc[p.drift_class]=dcc.get(p.drift_class,0)+1
    fs=_canonical_section({"failure_count":len(drifts),"incomplete_replay":chaos_replay_verdict_receipt.chaos_replay_verdict.verdict_class=="INCOMPLETE_REPLAY","drift_count":len(drifts),"drift_class_counts":{k:dcc[k] for k in sorted(dcc)},"drifted_comparison_labels":[p.comparison_label for p in drifts],"drifted_comparison_kinds":[p.comparison_kind for p in drifts]})
    if len(drifts)>_MAX_FAILURE_ENTRIES: raise ValueError(_ERR_REPORT_COUNT_MISMATCH)
    fv=_report_verdict(chaos_replay_verdict_receipt.chaos_replay_verdict.verdict_class)
    payload=_game_world_interaction_report_payload(game_world_intake_receipt.receipt_hash,adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,chaos_replay_verdict_receipt.chaos_replay_verdict_receipt_hash,obs,ph,src,adp,obsec,et,sp,ch,sb,fs,fv)
    return GameWorldInteractionReport(game_world_intake_receipt.receipt_hash,adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,chaos_replay_verdict_receipt.chaos_replay_verdict_receipt_hash,obs,ph,src,adp,obsec,et,sp,ch,sb,fs,fv,sha256_hex(payload))

def build_game_world_interaction_report_receipt(game_world_intake_receipt:GameWorldIntakeReceipt,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...],chaos_replay_verdict_receipt:ChaosReplayVerdictReceipt)->GameWorldInteractionReportReceipt:
    r=build_game_world_interaction_report(game_world_intake_receipt,adapter_contract_receipt,adapter_spec,episode_trace_receipt,strategy_probe_receipts,chaos_replay_verdict_receipt); validate_game_world_interaction_report(r)
    return GameWorldInteractionReportReceipt(r,sha256_hex(_game_world_interaction_report_receipt_payload(r)))

def validate_game_world_interaction_report(report:GameWorldInteractionReport)->bool:
    if not isinstance(report,GameWorldInteractionReport): raise ValueError(_ERR_INVALID_INPUT)
    for h in (report.game_world_intake_receipt_hash,report.world_adapter_contract_receipt_hash,report.adapter_spec_hash,report.episode_trace_receipt_hash,report.chaos_replay_verdict_receipt_hash,report.game_world_interaction_report_hash): _validate_hash(h)
    if not isinstance(report.observation_snapshot_receipt_hashes,tuple) or len(report.observation_snapshot_receipt_hashes)>_MAX_OBSERVATION_RECEIPTS: raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(report.strategy_probe_receipt_hashes,tuple) or len(report.strategy_probe_receipt_hashes)>_MAX_STRATEGY_PROBE_RECEIPTS: raise ValueError(_ERR_INVALID_INPUT)
    for h in report.observation_snapshot_receipt_hashes+report.strategy_probe_receipt_hashes: _validate_hash(h)
    for name in _SECTIONS: _validate_section(getattr(report,name))
    if report.final_report_verdict not in _ALLOWED_VERDICTS: raise ValueError(_ERR_INVALID_INTERACTION_REPORT_VERDICT)
    if report.game_world_interaction_report_hash!=sha256_hex(report._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_game_world_interaction_report_receipt(receipt:GameWorldInteractionReportReceipt)->bool:
    if not isinstance(receipt,GameWorldInteractionReportReceipt): raise ValueError(_ERR_INVALID_INPUT)
    validate_game_world_interaction_report(receipt.game_world_interaction_report); _validate_hash(receipt.game_world_interaction_report_receipt_hash)
    if receipt.game_world_interaction_report_receipt_hash!=sha256_hex(receipt._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_game_world_interaction_report_with_artifacts(report:GameWorldInteractionReport,game_world_intake_receipt:GameWorldIntakeReceipt,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...],chaos_replay_verdict_receipt:ChaosReplayVerdictReceipt)->bool:
    validate_game_world_interaction_report(report)
    rb=build_game_world_interaction_report(game_world_intake_receipt,adapter_contract_receipt,adapter_spec,episode_trace_receipt,strategy_probe_receipts,chaos_replay_verdict_receipt)
    for n,e in (("source_intake",_ERR_INTAKE_REPORT_MISMATCH),("adapter_contract",_ERR_ADAPTER_REPORT_MISMATCH),("observation_summary",_ERR_OBSERVATION_REPORT_MISMATCH),("episode_trace_summary",_ERR_EPISODE_TRACE_REPORT_MISMATCH),("strategy_probe_summary",_ERR_STRATEGY_PROBE_REPORT_MISMATCH),("chaos_replay_summary",_ERR_CHAOS_REPLAY_REPORT_MISMATCH),("scope_boundary",_ERR_REPORT_SECTION_MISMATCH),("failure_summary",_ERR_REPORT_SECTION_MISMATCH)):
        if getattr(report,n)!=getattr(rb,n): raise ValueError(e)
    if report.final_report_verdict!=rb.final_report_verdict: raise ValueError(_ERR_REPORT_VERDICT_MISMATCH)
    if report.to_dict()!=rb.to_dict(): raise ValueError(_ERR_REPORT_SECTION_MISMATCH)
    return True

def validate_game_world_interaction_report_receipt_with_artifacts(receipt:GameWorldInteractionReportReceipt,game_world_intake_receipt:GameWorldIntakeReceipt,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...],chaos_replay_verdict_receipt:ChaosReplayVerdictReceipt)->bool:
    validate_game_world_interaction_report_receipt(receipt)
    rb=build_game_world_interaction_report_receipt(game_world_intake_receipt,adapter_contract_receipt,adapter_spec,episode_trace_receipt,strategy_probe_receipts,chaos_replay_verdict_receipt)
    if receipt.game_world_interaction_report_receipt_hash!=rb.game_world_interaction_report_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    if receipt.to_dict()!=rb.to_dict(): raise ValueError(_ERR_INTERACTION_REPORT_RECEIPT_MISMATCH)
    return True
