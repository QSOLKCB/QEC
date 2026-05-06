from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_adapter_contract import WorldAdapterContractReceipt, WorldAdapterSpec, validate_world_adapter_contract_receipt, validate_world_adapter_spec
from .game_world_observation_snapshot import validate_spec_in_contract
from .game_world_episode_trace import EpisodeTraceReceipt, validate_episode_trace_receipt, validate_episode_trace_receipt_with_adapter
from .game_world_strategy_probe import StrategyProbeReceipt, validate_strategy_probe_receipt, validate_strategy_probe_receipt_with_adapter

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_COMPARISON_KIND = "INVALID_COMPARISON_KIND"
_ERR_INVALID_VERDICT_CLASS = "INVALID_VERDICT_CLASS"
_ERR_COMPARISON_INDEX_OUT_OF_BOUNDS = "COMPARISON_INDEX_OUT_OF_BOUNDS"
_ERR_COMPARISON_COUNT_MISMATCH = "COMPARISON_COUNT_MISMATCH"
_ERR_DRIFT_COUNT_MISMATCH = "DRIFT_COUNT_MISMATCH"
_ERR_DUPLICATE_COMPARISON_POINT = "DUPLICATE_COMPARISON_POINT"
_ERR_COMPARISON_ORDER_MISMATCH = "COMPARISON_ORDER_MISMATCH"
_ERR_DRIFT_CLASS_MISMATCH = "DRIFT_CLASS_MISMATCH"
_ERR_VERDICT_CLASS_MISMATCH = "VERDICT_CLASS_MISMATCH"
_ERR_REPLAY_ARTIFACT_MISMATCH = "REPLAY_ARTIFACT_MISMATCH"
_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT = "ADAPTER_SPEC_NOT_IN_CONTRACT"
_ERR_PROBE_PAIRING_MISMATCH = "PROBE_PAIRING_MISMATCH"
_ERR_INCOMPLETE_REPLAY = "INCOMPLETE_REPLAY"
_MAX_COMPARISON_POINTS = 10_000
_MAX_COMPARISON_INDEX = 1_000_000
_MAX_COMPARISON_LABEL_LENGTH = 96
_MAX_STRATEGY_PROBE_RECEIPTS = 1_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_VERDICTS=frozenset({"CHAOS_REPLAY_CLEAN","OBSERVATION_DRIFT","ACTION_DRIFT","TRACE_DRIFT","PROBE_DRIFT","ADAPTER_DRIFT","INTAKE_DRIFT","INCOMPLETE_REPLAY"})
_ALLOWED_KINDS=frozenset({"INTAKE_RECEIPT","ADAPTER_CONTRACT","ADAPTER_SPEC","EPISODE_TRACE","OBSERVATION_SNAPSHOT","ACTION_SELECTION","TERMINAL_FLAG","STRATEGY_PROBE"})
_KIND_TO_DRIFT={"INTAKE_RECEIPT":"INTAKE_DRIFT","ADAPTER_CONTRACT":"ADAPTER_DRIFT","ADAPTER_SPEC":"ADAPTER_DRIFT","EPISODE_TRACE":"TRACE_DRIFT","OBSERVATION_SNAPSHOT":"OBSERVATION_DRIFT","ACTION_SELECTION":"ACTION_DRIFT","TERMINAL_FLAG":"TRACE_DRIFT","STRATEGY_PROBE":"PROBE_DRIFT"}
_PRECEDENCE=("INCOMPLETE_REPLAY","INTAKE_DRIFT","ADAPTER_DRIFT","OBSERVATION_DRIFT","ACTION_DRIFT","PROBE_DRIFT","TRACE_DRIFT","CHAOS_REPLAY_CLEAN")

def get_allowed_chaos_replay_verdict_classes()->frozenset[str]: return _ALLOWED_VERDICTS
def get_allowed_replay_comparison_kinds()->frozenset[str]: return _ALLOWED_KINDS

def _validate_hash_string(v: object)->None:
    if not isinstance(v,str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)

def _comparison_point_order_key(point:"ReplayComparisonPoint")->tuple[int,str,str,str]: return (point.comparison_index,point.comparison_kind,point.comparison_label,point.replay_comparison_point_hash)

def _terminal_flag_hash(v: bool)->str: return sha256_hex({"terminal_flag":v})

def _derive_drift(kind:str, expected_hash:str|None, observed_hash:str|None)->tuple[bool,str]:
    if expected_hash is None or observed_hash is None: return True, _ERR_INCOMPLETE_REPLAY
    if expected_hash==observed_hash: return False,"CHAOS_REPLAY_CLEAN"
    return True,_KIND_TO_DRIFT[kind]

def _replay_comparison_point_payload(comparison_index:int,comparison_kind:str,comparison_label:str,expected_hash:str|None,observed_hash:str|None,drifted:bool,drift_class:str)->dict[str,Any]:
    return {"comparison_index":comparison_index,"comparison_kind":comparison_kind,"comparison_label":comparison_label,"expected_hash":expected_hash,"observed_hash":observed_hash,"drifted":drifted,"drift_class":drift_class}

def _chaos_replay_verdict_payload(expected_intake_receipt_hash:str,observed_intake_receipt_hash:str|None,expected_adapter_contract_receipt_hash:str,observed_adapter_contract_receipt_hash:str|None,expected_adapter_spec_hash:str,observed_adapter_spec_hash:str|None,expected_episode_trace_receipt_hash:str,observed_episode_trace_receipt_hash:str|None,comparison_points:tuple["ReplayComparisonPoint",...],comparison_count:int,drift_count:int,verdict_class:str)->dict[str,Any]:
    return {"expected_intake_receipt_hash":expected_intake_receipt_hash,"observed_intake_receipt_hash":observed_intake_receipt_hash,"expected_adapter_contract_receipt_hash":expected_adapter_contract_receipt_hash,"observed_adapter_contract_receipt_hash":observed_adapter_contract_receipt_hash,"expected_adapter_spec_hash":expected_adapter_spec_hash,"observed_adapter_spec_hash":observed_adapter_spec_hash,"expected_episode_trace_receipt_hash":expected_episode_trace_receipt_hash,"observed_episode_trace_receipt_hash":observed_episode_trace_receipt_hash,"comparison_points":[p.to_dict() for p in comparison_points],"comparison_count":comparison_count,"drift_count":drift_count,"verdict_class":verdict_class}

def _chaos_replay_verdict_receipt_payload(chaos_replay_verdict:"ChaosReplayVerdict")->dict[str,Any]: return {"chaos_replay_verdict":chaos_replay_verdict.to_dict()}

def _validate_common_inputs(idx:object,kind:object,label:object,eh:object,oh:object)->None:
    if not isinstance(idx,int) or isinstance(idx,bool): raise ValueError(_ERR_INVALID_INPUT)
    if idx<0 or idx>_MAX_COMPARISON_INDEX: raise ValueError(_ERR_COMPARISON_INDEX_OUT_OF_BOUNDS)
    if not isinstance(kind,str) or kind not in _ALLOWED_KINDS: raise ValueError(_ERR_INVALID_COMPARISON_KIND)
    if not isinstance(label,str) or len(label)<1 or len(label)>_MAX_COMPARISON_LABEL_LENGTH or _LABEL_RE.fullmatch(label) is None: raise ValueError(_ERR_INVALID_INPUT)
    if eh is None and oh is None: raise ValueError(_ERR_INCOMPLETE_REPLAY)
    if eh is not None: _validate_hash_string(eh)
    if oh is not None: _validate_hash_string(oh)

def _derive_verdict_class(points:tuple["ReplayComparisonPoint",...])->str:
    drifts={p.drift_class for p in points if p.drifted}
    if not drifts: return "CHAOS_REPLAY_CLEAN"
    for v in _PRECEDENCE:
        if v in drifts: return v
    raise ValueError(_ERR_INVALID_VERDICT_CLASS)

@dataclass(frozen=True)
class ReplayComparisonPoint:
    comparison_index:int; comparison_kind:str; comparison_label:str; expected_hash:str|None; observed_hash:str|None; drifted:bool; drift_class:str; replay_comparison_point_hash:str
    def __post_init__(self)->None: validate_replay_comparison_point(self)
    def _hash_payload(self)->dict[str,Any]: return _replay_comparison_point_payload(self.comparison_index,self.comparison_kind,self.comparison_label,self.expected_hash,self.observed_hash,self.drifted,self.drift_class)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"replay_comparison_point_hash":self.replay_comparison_point_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class ChaosReplayVerdict:
    expected_intake_receipt_hash:str; observed_intake_receipt_hash:str|None; expected_adapter_contract_receipt_hash:str; observed_adapter_contract_receipt_hash:str|None; expected_adapter_spec_hash:str; observed_adapter_spec_hash:str|None; expected_episode_trace_receipt_hash:str; observed_episode_trace_receipt_hash:str|None; comparison_points:tuple[ReplayComparisonPoint,...]; comparison_count:int; drift_count:int; verdict_class:str; chaos_replay_verdict_hash:str
    def __post_init__(self)->None: validate_chaos_replay_verdict(self)
    def _hash_payload(self)->dict[str,Any]: return _chaos_replay_verdict_payload(self.expected_intake_receipt_hash,self.observed_intake_receipt_hash,self.expected_adapter_contract_receipt_hash,self.observed_adapter_contract_receipt_hash,self.expected_adapter_spec_hash,self.observed_adapter_spec_hash,self.expected_episode_trace_receipt_hash,self.observed_episode_trace_receipt_hash,self.comparison_points,self.comparison_count,self.drift_count,self.verdict_class)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"chaos_replay_verdict_hash":self.chaos_replay_verdict_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class ChaosReplayVerdictReceipt:
    chaos_replay_verdict:ChaosReplayVerdict; chaos_replay_verdict_receipt_hash:str
    def __post_init__(self)->None: validate_chaos_replay_verdict_receipt(self)
    def _hash_payload(self)->dict[str,Any]: return _chaos_replay_verdict_receipt_payload(self.chaos_replay_verdict)
    def to_dict(self)->dict[str,Any]: return {"chaos_replay_verdict":self.chaos_replay_verdict.to_dict(),"chaos_replay_verdict_receipt_hash":self.chaos_replay_verdict_receipt_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

def validate_replay_comparison_point(point:ReplayComparisonPoint)->bool:
    if not isinstance(point,ReplayComparisonPoint): raise ValueError(_ERR_INVALID_INPUT)
    _validate_common_inputs(point.comparison_index,point.comparison_kind,point.comparison_label,point.expected_hash,point.observed_hash)
    if not isinstance(point.drifted,bool) or point.drift_class not in _ALLOWED_VERDICTS: raise ValueError(_ERR_INVALID_INPUT)
    d,dc=_derive_drift(point.comparison_kind,point.expected_hash,point.observed_hash)
    if point.drifted!=d or point.drift_class!=dc: raise ValueError(_ERR_DRIFT_CLASS_MISMATCH)
    _validate_hash_string(point.replay_comparison_point_hash)
    if point.replay_comparison_point_hash!=sha256_hex(point._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_chaos_replay_verdict(verdict:ChaosReplayVerdict)->bool:
    if not isinstance(verdict,ChaosReplayVerdict): raise ValueError(_ERR_INVALID_INPUT)
    for h in (verdict.expected_intake_receipt_hash,verdict.expected_adapter_contract_receipt_hash,verdict.expected_adapter_spec_hash,verdict.expected_episode_trace_receipt_hash): _validate_hash_string(h)
    for h in (verdict.observed_intake_receipt_hash,verdict.observed_adapter_contract_receipt_hash,verdict.observed_adapter_spec_hash,verdict.observed_episode_trace_receipt_hash):
        if h is not None: _validate_hash_string(h)
    if not isinstance(verdict.comparison_points,tuple) or len(verdict.comparison_points)<1 or len(verdict.comparison_points)>_MAX_COMPARISON_POINTS: raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(verdict.comparison_count,int) or isinstance(verdict.comparison_count,bool): raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(verdict.drift_count,int) or isinstance(verdict.drift_count,bool): raise ValueError(_ERR_INVALID_INPUT)
    if verdict.verdict_class not in _ALLOWED_VERDICTS: raise ValueError(_ERR_INVALID_VERDICT_CLASS)
    prev=None; idxs=set(); kl=set()
    for p in verdict.comparison_points:
        validate_replay_comparison_point(p); k=_comparison_point_order_key(p)
        if prev is not None and prev>k: raise ValueError(_ERR_COMPARISON_ORDER_MISMATCH)
        prev=k
        if p.comparison_index in idxs or (p.comparison_kind,p.comparison_label) in kl: raise ValueError(_ERR_DUPLICATE_COMPARISON_POINT)
        idxs.add(p.comparison_index); kl.add((p.comparison_kind,p.comparison_label))
    if verdict.comparison_count!=len(verdict.comparison_points): raise ValueError(_ERR_COMPARISON_COUNT_MISMATCH)
    if verdict.drift_count!=sum(1 for p in verdict.comparison_points if p.drifted): raise ValueError(_ERR_DRIFT_COUNT_MISMATCH)
    if verdict.verdict_class!=_derive_verdict_class(verdict.comparison_points): raise ValueError(_ERR_VERDICT_CLASS_MISMATCH)
    _validate_hash_string(verdict.chaos_replay_verdict_hash)
    if verdict.chaos_replay_verdict_hash!=sha256_hex(verdict._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_chaos_replay_verdict_receipt(receipt:ChaosReplayVerdictReceipt)->bool:
    if not isinstance(receipt,ChaosReplayVerdictReceipt): raise ValueError(_ERR_INVALID_INPUT)
    validate_chaos_replay_verdict(receipt.chaos_replay_verdict); _validate_hash_string(receipt.chaos_replay_verdict_receipt_hash)
    if receipt.chaos_replay_verdict_receipt_hash!=sha256_hex(receipt._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def build_replay_comparison_point(comparison_index:int,comparison_kind:str,comparison_label:str,expected_hash:str|None,observed_hash:str|None)->ReplayComparisonPoint:
    _validate_common_inputs(comparison_index,comparison_kind,comparison_label,expected_hash,observed_hash)
    drifted,drift_class=_derive_drift(comparison_kind,expected_hash,observed_hash)
    h=sha256_hex(_replay_comparison_point_payload(comparison_index,comparison_kind,comparison_label,expected_hash,observed_hash,drifted,drift_class))
    return ReplayComparisonPoint(comparison_index,comparison_kind,comparison_label,expected_hash,observed_hash,drifted,drift_class,h)

def _probe_key(r:StrategyProbeReceipt)->tuple[str,str]: return (r.strategy_probe_spec.probe_type,r.strategy_probe_spec.probe_label)

def build_chaos_replay_verdict(expected_adapter_contract_receipt:WorldAdapterContractReceipt,expected_adapter_spec:WorldAdapterSpec,expected_episode_trace_receipt:EpisodeTraceReceipt,observed_adapter_contract_receipt:WorldAdapterContractReceipt|None=None,observed_adapter_spec:WorldAdapterSpec|None=None,observed_episode_trace_receipt:EpisodeTraceReceipt|None=None,expected_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=(),observed_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=())->ChaosReplayVerdict:
    validate_world_adapter_contract_receipt(expected_adapter_contract_receipt); validate_world_adapter_spec(expected_adapter_spec)
    try: validate_spec_in_contract(expected_adapter_contract_receipt,expected_adapter_spec)
    except ValueError: raise ValueError(_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT)
    validate_episode_trace_receipt_with_adapter(expected_episode_trace_receipt,expected_adapter_contract_receipt,expected_adapter_spec)
    if len(expected_strategy_probe_receipts)>_MAX_STRATEGY_PROBE_RECEIPTS or len(observed_strategy_probe_receipts)>_MAX_STRATEGY_PROBE_RECEIPTS: raise ValueError(_ERR_INVALID_INPUT)
    for r in expected_strategy_probe_receipts: validate_strategy_probe_receipt_with_adapter(r,expected_adapter_contract_receipt,expected_adapter_spec,expected_episode_trace_receipt)
    if observed_adapter_contract_receipt is not None: validate_world_adapter_contract_receipt(observed_adapter_contract_receipt)
    if observed_adapter_spec is not None: validate_world_adapter_spec(observed_adapter_spec)
    if observed_adapter_contract_receipt is not None and observed_adapter_spec is not None:
        try: validate_spec_in_contract(observed_adapter_contract_receipt,observed_adapter_spec)
        except ValueError: raise ValueError(_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT)
    if observed_episode_trace_receipt is not None: validate_episode_trace_receipt(observed_episode_trace_receipt)
    if observed_episode_trace_receipt is not None and observed_adapter_contract_receipt is not None and observed_adapter_spec is not None:
        validate_episode_trace_receipt_with_adapter(observed_episode_trace_receipt,observed_adapter_contract_receipt,observed_adapter_spec)
    for r in observed_strategy_probe_receipts:
        validate_strategy_probe_receipt(r)
        if observed_adapter_contract_receipt is not None and observed_adapter_spec is not None and observed_episode_trace_receipt is not None:
            validate_strategy_probe_receipt_with_adapter(r,observed_adapter_contract_receipt,observed_adapter_spec,observed_episode_trace_receipt)
    points=[]; idx=0
    def add(kind,label,eh,oh):
        nonlocal idx; points.append(build_replay_comparison_point(idx,kind,label,eh,oh)); idx+=1
    add("INTAKE_RECEIPT","INTAKE",expected_adapter_contract_receipt.intake_receipt_hash,observed_adapter_contract_receipt.intake_receipt_hash if observed_adapter_contract_receipt else None)
    add("ADAPTER_CONTRACT","ADAPTER_CONTRACT",expected_adapter_contract_receipt.adapter_contract_receipt_hash,observed_adapter_contract_receipt.adapter_contract_receipt_hash if observed_adapter_contract_receipt else None)
    add("ADAPTER_SPEC","ADAPTER_SPEC",expected_adapter_spec.adapter_spec_hash,observed_adapter_spec.adapter_spec_hash if observed_adapter_spec else None)
    add("EPISODE_TRACE","EPISODE_TRACE",expected_episode_trace_receipt.episode_trace_receipt_hash,observed_episode_trace_receipt.episode_trace_receipt_hash if observed_episode_trace_receipt else None)
    es=expected_episode_trace_receipt.episode_trace.episode_steps; os=observed_episode_trace_receipt.episode_trace.episode_steps if observed_episode_trace_receipt else ()
    n=max(len(es),len(os))
    if 4+3*n+len(expected_strategy_probe_receipts)+len(observed_strategy_probe_receipts)>_MAX_COMPARISON_POINTS: raise ValueError(_ERR_INVALID_INPUT)
    for i in range(n):
        e=es[i] if i<len(es) else None; o=os[i] if i<len(os) else None
        add("OBSERVATION_SNAPSHOT",f"STEP_{i:06d}_OBSERVATION",e.observation_snapshot_receipt.observation_snapshot_receipt_hash if e else None,o.observation_snapshot_receipt.observation_snapshot_receipt_hash if o else None)
        add("ACTION_SELECTION",f"STEP_{i:06d}_ACTION",e.action_atom.action_atom_hash if e else None,o.action_atom.action_atom_hash if o else None)
        add("TERMINAL_FLAG",f"STEP_{i:06d}_TERMINAL",_terminal_flag_hash(e.terminal_flag) if e else None,_terminal_flag_hash(o.terminal_flag) if o else None)
    em={}; om={}
    for r in expected_strategy_probe_receipts:
        k=_probe_key(r)
        if k in em: raise ValueError(_ERR_PROBE_PAIRING_MISMATCH)
        em[k]=r
    for r in observed_strategy_probe_receipts:
        k=_probe_key(r)
        if k in om: raise ValueError(_ERR_PROBE_PAIRING_MISMATCH)
        om[k]=r
    for pt,pl in sorted(set(em.keys())|set(om.keys())):
        add("STRATEGY_PROBE",f"PROBE_{pt}_{pl}",em[(pt,pl)].strategy_probe_receipt_hash if (pt,pl) in em else None,om[(pt,pl)].strategy_probe_receipt_hash if (pt,pl) in om else None)
    t=tuple(points); c=len(t); d=sum(1 for p in t if p.drifted); vc=_derive_verdict_class(t); h=sha256_hex(_chaos_replay_verdict_payload(expected_adapter_contract_receipt.intake_receipt_hash,observed_adapter_contract_receipt.intake_receipt_hash if observed_adapter_contract_receipt else None,expected_adapter_contract_receipt.adapter_contract_receipt_hash,observed_adapter_contract_receipt.adapter_contract_receipt_hash if observed_adapter_contract_receipt else None,expected_adapter_spec.adapter_spec_hash,observed_adapter_spec.adapter_spec_hash if observed_adapter_spec else None,expected_episode_trace_receipt.episode_trace_receipt_hash,observed_episode_trace_receipt.episode_trace_receipt_hash if observed_episode_trace_receipt else None,t,c,d,vc))
    return ChaosReplayVerdict(expected_adapter_contract_receipt.intake_receipt_hash,observed_adapter_contract_receipt.intake_receipt_hash if observed_adapter_contract_receipt else None,expected_adapter_contract_receipt.adapter_contract_receipt_hash,observed_adapter_contract_receipt.adapter_contract_receipt_hash if observed_adapter_contract_receipt else None,expected_adapter_spec.adapter_spec_hash,observed_adapter_spec.adapter_spec_hash if observed_adapter_spec else None,expected_episode_trace_receipt.episode_trace_receipt_hash,observed_episode_trace_receipt.episode_trace_receipt_hash if observed_episode_trace_receipt else None,t,c,d,vc,h)

def build_chaos_replay_verdict_receipt(expected_adapter_contract_receipt:WorldAdapterContractReceipt,expected_adapter_spec:WorldAdapterSpec,expected_episode_trace_receipt:EpisodeTraceReceipt,observed_adapter_contract_receipt:WorldAdapterContractReceipt|None=None,observed_adapter_spec:WorldAdapterSpec|None=None,observed_episode_trace_receipt:EpisodeTraceReceipt|None=None,expected_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=(),observed_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=())->ChaosReplayVerdictReceipt:
    v=build_chaos_replay_verdict(expected_adapter_contract_receipt,expected_adapter_spec,expected_episode_trace_receipt,observed_adapter_contract_receipt,observed_adapter_spec,observed_episode_trace_receipt,expected_strategy_probe_receipts,observed_strategy_probe_receipts); h=sha256_hex(_chaos_replay_verdict_receipt_payload(v)); return ChaosReplayVerdictReceipt(v,h)

def validate_chaos_replay_verdict_with_artifacts(verdict:ChaosReplayVerdict,expected_adapter_contract_receipt:WorldAdapterContractReceipt,expected_adapter_spec:WorldAdapterSpec,expected_episode_trace_receipt:EpisodeTraceReceipt,observed_adapter_contract_receipt:WorldAdapterContractReceipt|None=None,observed_adapter_spec:WorldAdapterSpec|None=None,observed_episode_trace_receipt:EpisodeTraceReceipt|None=None,expected_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=(),observed_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=())->bool:
    validate_chaos_replay_verdict(verdict); rebuilt=build_chaos_replay_verdict(expected_adapter_contract_receipt,expected_adapter_spec,expected_episode_trace_receipt,observed_adapter_contract_receipt,observed_adapter_spec,observed_episode_trace_receipt,expected_strategy_probe_receipts,observed_strategy_probe_receipts)
    if verdict.chaos_replay_verdict_hash!=rebuilt.chaos_replay_verdict_hash: raise ValueError(_ERR_HASH_MISMATCH)
    if verdict.to_dict()!=rebuilt.to_dict(): raise ValueError(_ERR_REPLAY_ARTIFACT_MISMATCH)
    return True

def validate_chaos_replay_verdict_receipt_with_artifacts(receipt:ChaosReplayVerdictReceipt,expected_adapter_contract_receipt:WorldAdapterContractReceipt,expected_adapter_spec:WorldAdapterSpec,expected_episode_trace_receipt:EpisodeTraceReceipt,observed_adapter_contract_receipt:WorldAdapterContractReceipt|None=None,observed_adapter_spec:WorldAdapterSpec|None=None,observed_episode_trace_receipt:EpisodeTraceReceipt|None=None,expected_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=(),observed_strategy_probe_receipts:list[StrategyProbeReceipt]|tuple[StrategyProbeReceipt,...]=())->bool:
    validate_chaos_replay_verdict_receipt(receipt)
    rb=build_chaos_replay_verdict_receipt(expected_adapter_contract_receipt,expected_adapter_spec,expected_episode_trace_receipt,observed_adapter_contract_receipt,observed_adapter_spec,observed_episode_trace_receipt,expected_strategy_probe_receipts,observed_strategy_probe_receipts)
    if receipt.chaos_replay_verdict_receipt_hash!=rb.chaos_replay_verdict_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    if receipt.to_dict()!=rb.to_dict(): raise ValueError(_ERR_REPLAY_ARTIFACT_MISMATCH)
    return True
