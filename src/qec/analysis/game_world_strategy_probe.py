from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_adapter_contract import (
    ActionAtom,
    WorldAdapterContractReceipt,
    WorldAdapterSpec,
    validate_action_atom,
    validate_world_adapter_contract_receipt,
    validate_world_adapter_spec,
)
from .game_world_observation_snapshot import validate_spec_in_contract
from .game_world_episode_trace import EpisodeTraceReceipt, validate_episode_trace_receipt_with_adapter

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_PROBE_TYPE = "INVALID_PROBE_TYPE"
_ERR_INVALID_PROBE_PARAMETERS = "INVALID_PROBE_PARAMETERS"
_ERR_INVALID_PROBE_RESULT = "INVALID_PROBE_RESULT"
_ERR_PROBE_PARAMETER_TOO_LARGE = "PROBE_PARAMETER_TOO_LARGE"
_ERR_PROBE_RESULT_TOO_LARGE = "PROBE_RESULT_TOO_LARGE"
_ERR_PROBE_TRACE_MISMATCH = "PROBE_TRACE_MISMATCH"
_ERR_PROBE_ADAPTER_MISMATCH = "PROBE_ADAPTER_MISMATCH"
_ERR_PROBE_TYPE_MISMATCH = "PROBE_TYPE_MISMATCH"
_ERR_ACTION_NOT_IN_ALPHABET = "ACTION_NOT_IN_ALPHABET"
_ERR_UNKNOWN_ACTION_NOT_REJECTED = "UNKNOWN_ACTION_NOT_REJECTED"
_ERR_TRACE_TERMINAL_MISMATCH = "TRACE_TERMINAL_MISMATCH"
_ERR_SCORE_DELTA_MISMATCH = "SCORE_DELTA_MISMATCH"
_ERR_DIVERGENCE_RESULT_MISMATCH = "DIVERGENCE_RESULT_MISMATCH"
_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT = "ADAPTER_SPEC_NOT_IN_CONTRACT"

_MAX_PROBE_LABEL_LENGTH = 64
_MAX_PROBE_PARAMETER_BYTES = 4096
_MAX_RESULT_PAYLOAD_BYTES = 8192
_MAX_REPEAT_ACTION_STEPS = 1_000
_MAX_SCAN_ACTIONS = 1_000
_MAX_SCORE_DELTA_POINTS = 1_000
_MAX_DIVERGENCE_POINTS = 1_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ACTION_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED = frozenset({"NO_OP_BASELINE", "LEGAL_ACTION_SCAN", "REPEAT_ACTION_STABILITY", "TRACE_DIVERGENCE_SCAN", "TERMINAL_STATE_SCAN", "SCORE_DELTA_SCAN", "UNKNOWN_ACTION_REJECTION"})


def get_allowed_strategy_probe_types() -> frozenset[str]: return _ALLOWED

def _validate_hash_string(v: object) -> None:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)

def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)): return
    if isinstance(value, float): raise ValueError(_ERR_INVALID_INPUT)
    if isinstance(value, list):
        for x in value: _validate_json_safe_no_floats(x)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or k == "": raise ValueError(_ERR_INVALID_INPUT)
            _validate_json_safe_no_floats(v)
        return
    raise ValueError(_ERR_INVALID_INPUT)

def _strategy_probe_spec_payload(adapter_contract_receipt_hash:str,adapter_spec_hash:str,episode_trace_receipt_hash:str,probe_type:str,probe_label:str,canonical_probe_parameters:str,probe_parameters_hash:str)->dict[str,Any]:
    return {"adapter_contract_receipt_hash":adapter_contract_receipt_hash,"adapter_spec_hash":adapter_spec_hash,"episode_trace_receipt_hash":episode_trace_receipt_hash,"probe_type":probe_type,"probe_label":probe_label,"canonical_probe_parameters":canonical_probe_parameters,"probe_parameters_hash":probe_parameters_hash}

def _strategy_probe_result_payload(adapter_contract_receipt_hash:str,adapter_spec_hash:str,episode_trace_receipt_hash:str,strategy_probe_spec_hash:str,probe_type:str,canonical_result_payload:str,result_payload_hash:str)->dict[str,Any]:
    return {"adapter_contract_receipt_hash":adapter_contract_receipt_hash,"adapter_spec_hash":adapter_spec_hash,"episode_trace_receipt_hash":episode_trace_receipt_hash,"strategy_probe_spec_hash":strategy_probe_spec_hash,"probe_type":probe_type,"canonical_result_payload":canonical_result_payload,"result_payload_hash":result_payload_hash}

def _strategy_probe_receipt_payload(adapter_contract_receipt_hash:str,adapter_spec_hash:str,episode_trace_receipt_hash:str,strategy_probe_spec:"StrategyProbeSpec",strategy_probe_result:"StrategyProbeResult")->dict[str,Any]:
    return {"adapter_contract_receipt_hash":adapter_contract_receipt_hash,"adapter_spec_hash":adapter_spec_hash,"episode_trace_receipt_hash":episode_trace_receipt_hash,"strategy_probe_spec":strategy_probe_spec.to_dict(),"strategy_probe_result":strategy_probe_result.to_dict()}

def _parse_canonical(s:str, max_bytes:int, err:str)->object:
    if not isinstance(s,str): raise ValueError(_ERR_INVALID_INPUT)
    if len(s.encode("utf-8"))>max_bytes: raise ValueError(err)
    parsed=json.loads(s)
    if canonical_json(parsed)!=s: raise ValueError(_ERR_INVALID_INPUT)
    _validate_json_safe_no_floats(parsed)
    return parsed

def _validate_probe_type(pt:object)->None:
    if not isinstance(pt,str) or pt not in _ALLOWED: raise ValueError(_ERR_INVALID_PROBE_TYPE)

def _validate_probe_label(label:object)->None:
    if not isinstance(label,str) or len(label)<1 or len(label)>_MAX_PROBE_LABEL_LENGTH or _LABEL_RE.fullmatch(label) is None: raise ValueError(_ERR_INVALID_INPUT)

def _validate_probe_parameters(pt:str, params:object, spec:WorldAdapterSpec)->dict[str,Any]:
    if not isinstance(params,dict): raise ValueError(_ERR_INVALID_PROBE_PARAMETERS)
    _validate_json_safe_no_floats(params)
    if pt in {"NO_OP_BASELINE","LEGAL_ACTION_SCAN","TERMINAL_STATE_SCAN","SCORE_DELTA_SCAN"}:
        if params != {}: raise ValueError(_ERR_INVALID_PROBE_PARAMETERS)
    elif pt in {"REPEAT_ACTION_STABILITY","UNKNOWN_ACTION_REJECTION"}:
        if set(params.keys())!={"action_code"} or not isinstance(params["action_code"],str) or _ACTION_CODE_RE.fullmatch(params["action_code"]) is None: raise ValueError(_ERR_INVALID_PROBE_PARAMETERS)
        if pt=="REPEAT_ACTION_STABILITY" and params["action_code"] not in {a.action_code for a in spec.action_alphabet.actions}: raise ValueError(_ERR_ACTION_NOT_IN_ALPHABET)
    elif pt=="TRACE_DIVERGENCE_SCAN":
        if set(params.keys())!={"expected_episode_trace_hash"}: raise ValueError(_ERR_INVALID_PROBE_PARAMETERS)
        _validate_hash_string(params["expected_episode_trace_hash"])
    return params

def _derive_result_payload(spec: "StrategyProbeSpec",adapter_spec: WorldAdapterSpec,trace_receipt: EpisodeTraceReceipt)->dict[str,Any]:
    params=_parse_canonical(spec.canonical_probe_parameters,_MAX_PROBE_PARAMETER_BYTES,_ERR_PROBE_PARAMETER_TOO_LARGE)
    _validate_probe_parameters(spec.probe_type,params,adapter_spec)
    steps=trace_receipt.episode_trace.episode_steps
    if spec.probe_type=="NO_OP_BASELINE":
        no_ops=[s for s in steps if s.action_atom.action_code=="NO_OP"]
        return {"step_count":len(steps),"terminal_step_index":trace_receipt.episode_trace.terminal_step_index,"no_op_action_present":any(a.action_code=="NO_OP" for a in adapter_spec.action_alphabet.actions),"no_op_step_count":len(no_ops),"non_no_op_step_count":len(steps)-len(no_ops)}
    if spec.probe_type=="LEGAL_ACTION_SCAN":
        acts=adapter_spec.action_alphabet.actions
        if len(acts)>_MAX_SCAN_ACTIONS: raise ValueError(_ERR_INVALID_PROBE_RESULT)
        return {"action_count":len(acts),"action_codes":[a.action_code for a in acts],"action_atom_hashes":[a.action_atom_hash for a in acts]}
    if spec.probe_type=="REPEAT_ACTION_STABILITY":
        code=params["action_code"]; idx=[s.step_index for s in steps if s.action_atom.action_code==code]
        if len(idx)>_MAX_REPEAT_ACTION_STEPS: raise ValueError(_ERR_INVALID_PROBE_RESULT)
        return {"action_code":code,"repeated_step_indices":idx,"repeat_count":len(idx),"total_step_count":len(steps)}
    if spec.probe_type=="TRACE_DIVERGENCE_SCAN":
        expected=params["expected_episode_trace_hash"]; obs=trace_receipt.episode_trace.episode_trace_hash
        return {"expected_episode_trace_hash":expected,"observed_episode_trace_hash":obs,"diverged":expected!=obs,"divergence_class":"TRACE_HASH_DIVERGENCE" if expected!=obs else "TRACE_MATCH"}
    if spec.probe_type=="TERMINAL_STATE_SCAN":
        tsi=trace_receipt.episode_trace.terminal_step_index
        return {"terminal_present":tsi is not None,"terminal_step_index":tsi,"terminal_is_final":tsi==len(steps)-1 if tsi is not None else False,"step_count":len(steps)}
    if spec.probe_type=="SCORE_DELTA_SCAN":
        pts=[]
        for s in steps:
            o=s.observation_snapshot_receipt.observation_snapshot
            if o.observation_channel.channel_type=="SCORE_VALUE": pts.append({"step_index":s.step_index,"score_value":json.loads(o.canonical_observation_payload)})
        if len(pts)>_MAX_SCORE_DELTA_POINTS: raise ValueError(_ERR_INVALID_PROBE_RESULT)
        deltas=[pts[i+1]["score_value"]-pts[i]["score_value"] for i in range(len(pts)-1)]
        return {"score_points":pts,"score_delta_count":len(deltas),"deltas":deltas}
    code=params["action_code"]; known=code in {a.action_code for a in adapter_spec.action_alphabet.actions}
    return {"action_code":code,"rejected":not known,"reason":"ACTION_KNOWN" if known else "ACTION_NOT_IN_ALPHABET"}

@dataclass(frozen=True)
class StrategyProbeSpec:
    adapter_contract_receipt_hash:str; adapter_spec_hash:str; episode_trace_receipt_hash:str; probe_type:str; probe_label:str; canonical_probe_parameters:str; probe_parameters_hash:str; strategy_probe_spec_hash:str
    def __post_init__(self)->None: validate_strategy_probe_spec(self)
    def _hash_payload(self)->dict[str,Any]: return _strategy_probe_spec_payload(self.adapter_contract_receipt_hash,self.adapter_spec_hash,self.episode_trace_receipt_hash,self.probe_type,self.probe_label,self.canonical_probe_parameters,self.probe_parameters_hash)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"strategy_probe_spec_hash":self.strategy_probe_spec_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class StrategyProbeResult:
    adapter_contract_receipt_hash:str; adapter_spec_hash:str; episode_trace_receipt_hash:str; strategy_probe_spec_hash:str; probe_type:str; canonical_result_payload:str; result_payload_hash:str; strategy_probe_result_hash:str
    def __post_init__(self)->None: validate_strategy_probe_result(self)
    def _hash_payload(self)->dict[str,Any]: return _strategy_probe_result_payload(self.adapter_contract_receipt_hash,self.adapter_spec_hash,self.episode_trace_receipt_hash,self.strategy_probe_spec_hash,self.probe_type,self.canonical_result_payload,self.result_payload_hash)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"strategy_probe_result_hash":self.strategy_probe_result_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class StrategyProbeReceipt:
    adapter_contract_receipt_hash:str; adapter_spec_hash:str; episode_trace_receipt_hash:str; strategy_probe_spec:StrategyProbeSpec; strategy_probe_result:StrategyProbeResult; strategy_probe_receipt_hash:str
    def __post_init__(self)->None: validate_strategy_probe_receipt(self)
    def _hash_payload(self)->dict[str,Any]: return _strategy_probe_receipt_payload(self.adapter_contract_receipt_hash,self.adapter_spec_hash,self.episode_trace_receipt_hash,self.strategy_probe_spec,self.strategy_probe_result)
    def to_dict(self)->dict[str,Any]: return {**self._hash_payload(),"strategy_probe_receipt_hash":self.strategy_probe_receipt_hash}
    def to_canonical_json(self)->str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return canonical_bytes(self.to_dict())

# validators + builders omitted brevity? nope

def validate_strategy_probe_spec(spec: StrategyProbeSpec) -> bool:
    if not isinstance(spec,StrategyProbeSpec): raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(spec.adapter_contract_receipt_hash); _validate_hash_string(spec.adapter_spec_hash); _validate_hash_string(spec.episode_trace_receipt_hash)
    _validate_probe_type(spec.probe_type); _validate_probe_label(spec.probe_label)
    p=_parse_canonical(spec.canonical_probe_parameters,_MAX_PROBE_PARAMETER_BYTES,_ERR_PROBE_PARAMETER_TOO_LARGE)
    if sha256_hex(spec.canonical_probe_parameters)!=spec.probe_parameters_hash: _validate_hash_string(spec.probe_parameters_hash); raise ValueError(_ERR_HASH_MISMATCH)
    _validate_hash_string(spec.probe_parameters_hash); _validate_hash_string(spec.strategy_probe_spec_hash)
    if spec.strategy_probe_spec_hash!=sha256_hex(spec._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    if not isinstance(p,dict): raise ValueError(_ERR_INVALID_PROBE_PARAMETERS)
    return True

def validate_strategy_probe_result(result: StrategyProbeResult) -> bool:
    if not isinstance(result,StrategyProbeResult): raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(result.adapter_contract_receipt_hash); _validate_hash_string(result.adapter_spec_hash); _validate_hash_string(result.episode_trace_receipt_hash); _validate_hash_string(result.strategy_probe_spec_hash)
    _validate_probe_type(result.probe_type)
    _parse_canonical(result.canonical_result_payload,_MAX_RESULT_PAYLOAD_BYTES,_ERR_PROBE_RESULT_TOO_LARGE)
    _validate_hash_string(result.result_payload_hash); _validate_hash_string(result.strategy_probe_result_hash)
    if result.result_payload_hash!=sha256_hex(result.canonical_result_payload): raise ValueError(_ERR_HASH_MISMATCH)
    if result.strategy_probe_result_hash!=sha256_hex(result._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_strategy_probe_receipt(receipt: StrategyProbeReceipt)->bool:
    if not isinstance(receipt,StrategyProbeReceipt): raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(receipt.adapter_contract_receipt_hash); _validate_hash_string(receipt.adapter_spec_hash); _validate_hash_string(receipt.episode_trace_receipt_hash)
    validate_strategy_probe_spec(receipt.strategy_probe_spec); validate_strategy_probe_result(receipt.strategy_probe_result)
    if receipt.strategy_probe_spec.probe_type!=receipt.strategy_probe_result.probe_type: raise ValueError(_ERR_PROBE_TYPE_MISMATCH)
    _validate_hash_string(receipt.strategy_probe_receipt_hash)
    if receipt.strategy_probe_receipt_hash!=sha256_hex(receipt._hash_payload()): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def build_strategy_probe_spec(adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,probe_type:str,probe_label:str,probe_parameters:object|None=None)->StrategyProbeSpec:
    validate_world_adapter_contract_receipt(adapter_contract_receipt); validate_world_adapter_spec(adapter_spec); validate_spec_in_contract(adapter_contract_receipt,adapter_spec); validate_episode_trace_receipt_with_adapter(episode_trace_receipt,adapter_contract_receipt,adapter_spec)
    _validate_probe_type(probe_type); _validate_probe_label(probe_label)
    params=_validate_probe_parameters(probe_type, {} if probe_parameters is None else probe_parameters, adapter_spec)
    cps=canonical_json(params)
    if len(cps.encode())>_MAX_PROBE_PARAMETER_BYTES: raise ValueError(_ERR_PROBE_PARAMETER_TOO_LARGE)
    ph=sha256_hex(cps)
    sh=sha256_hex(_strategy_probe_spec_payload(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,probe_type,probe_label,cps,ph))
    return StrategyProbeSpec(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,probe_type,probe_label,cps,ph,sh)

def validate_strategy_probe_spec_with_adapter(spec:StrategyProbeSpec,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt)->bool:
    validate_world_adapter_contract_receipt(adapter_contract_receipt); validate_world_adapter_spec(adapter_spec)
    try: validate_spec_in_contract(adapter_contract_receipt,adapter_spec)
    except ValueError: raise ValueError(_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT)
    validate_episode_trace_receipt_with_adapter(episode_trace_receipt,adapter_contract_receipt,adapter_spec); validate_strategy_probe_spec(spec)
    if spec.adapter_contract_receipt_hash!=adapter_contract_receipt.adapter_contract_receipt_hash or spec.adapter_spec_hash!=adapter_spec.adapter_spec_hash: raise ValueError(_ERR_PROBE_ADAPTER_MISMATCH)
    if spec.episode_trace_receipt_hash!=episode_trace_receipt.episode_trace_receipt_hash: raise ValueError(_ERR_PROBE_TRACE_MISMATCH)
    _validate_probe_parameters(spec.probe_type,_parse_canonical(spec.canonical_probe_parameters,_MAX_PROBE_PARAMETER_BYTES,_ERR_PROBE_PARAMETER_TOO_LARGE),adapter_spec)
    return True

def build_strategy_probe_result(adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,strategy_probe_spec:StrategyProbeSpec)->StrategyProbeResult:
    validate_strategy_probe_spec_with_adapter(strategy_probe_spec,adapter_contract_receipt,adapter_spec,episode_trace_receipt)
    payload=_derive_result_payload(strategy_probe_spec,adapter_spec,episode_trace_receipt)
    cr=canonical_json(payload)
    if len(cr.encode())>_MAX_RESULT_PAYLOAD_BYTES: raise ValueError(_ERR_PROBE_RESULT_TOO_LARGE)
    rh=sha256_hex(cr)
    sh=sha256_hex(_strategy_probe_result_payload(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,strategy_probe_spec.strategy_probe_spec_hash,strategy_probe_spec.probe_type,cr,rh))
    return StrategyProbeResult(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,strategy_probe_spec.strategy_probe_spec_hash,strategy_probe_spec.probe_type,cr,rh,sh)

def validate_strategy_probe_result_with_adapter(result:StrategyProbeResult,strategy_probe_spec:StrategyProbeSpec,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt)->bool:
    validate_strategy_probe_spec_with_adapter(strategy_probe_spec,adapter_contract_receipt,adapter_spec,episode_trace_receipt); validate_strategy_probe_result(result)
    if result.probe_type!=strategy_probe_spec.probe_type: raise ValueError(_ERR_PROBE_TYPE_MISMATCH)
    if result.adapter_contract_receipt_hash!=adapter_contract_receipt.adapter_contract_receipt_hash or result.adapter_spec_hash!=adapter_spec.adapter_spec_hash: raise ValueError(_ERR_PROBE_ADAPTER_MISMATCH)
    if result.episode_trace_receipt_hash!=episode_trace_receipt.episode_trace_receipt_hash: raise ValueError(_ERR_PROBE_TRACE_MISMATCH)
    exp=canonical_json(_derive_result_payload(strategy_probe_spec,adapter_spec,episode_trace_receipt))
    if exp!=result.canonical_result_payload:
        if result.probe_type=="SCORE_DELTA_SCAN": raise ValueError(_ERR_SCORE_DELTA_MISMATCH)
        if result.probe_type=="TRACE_DIVERGENCE_SCAN": raise ValueError(_ERR_DIVERGENCE_RESULT_MISMATCH)
        raise ValueError(_ERR_INVALID_PROBE_RESULT)
    return True

def build_strategy_probe_receipt(adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt,probe_type:str,probe_label:str,probe_parameters:object|None=None)->StrategyProbeReceipt:
    s=build_strategy_probe_spec(adapter_contract_receipt,adapter_spec,episode_trace_receipt,probe_type,probe_label,probe_parameters)
    r=build_strategy_probe_result(adapter_contract_receipt,adapter_spec,episode_trace_receipt,s)
    h=sha256_hex(_strategy_probe_receipt_payload(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,s,r))
    return StrategyProbeReceipt(adapter_contract_receipt.adapter_contract_receipt_hash,adapter_spec.adapter_spec_hash,episode_trace_receipt.episode_trace_receipt_hash,s,r,h)

def validate_strategy_probe_receipt_with_adapter(receipt:StrategyProbeReceipt,adapter_contract_receipt:WorldAdapterContractReceipt,adapter_spec:WorldAdapterSpec,episode_trace_receipt:EpisodeTraceReceipt)->bool:
    validate_strategy_probe_receipt(receipt)
    validate_strategy_probe_spec_with_adapter(receipt.strategy_probe_spec,adapter_contract_receipt,adapter_spec,episode_trace_receipt)
    validate_strategy_probe_result_with_adapter(receipt.strategy_probe_result,receipt.strategy_probe_spec,adapter_contract_receipt,adapter_spec,episode_trace_receipt)
    if receipt.adapter_contract_receipt_hash!=adapter_contract_receipt.adapter_contract_receipt_hash or receipt.adapter_spec_hash!=adapter_spec.adapter_spec_hash: raise ValueError(_ERR_PROBE_ADAPTER_MISMATCH)
    if receipt.episode_trace_receipt_hash!=episode_trace_receipt.episode_trace_receipt_hash: raise ValueError(_ERR_PROBE_TRACE_MISMATCH)
    return True
