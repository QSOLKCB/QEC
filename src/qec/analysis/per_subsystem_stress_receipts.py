from __future__ import annotations
from dataclasses import dataclass, fields
import re
from typing import Any
from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .perturbation_matrix import EnergyMatrixReceipt, PerturbationMatrix, PerturbationMatrixEntry, validate_energy_matrix_receipt, validate_perturbation_matrix
from .semantic_stress_receipt import PerturbationStabilityProof, SemanticStressReceipt, validate_perturbation_stability_proof_with_receipts, validate_semantic_stress_receipt_with_matrix

_ERR_INVALID_INPUT="INVALID_INPUT"; _ERR_INVALID_HASH_FORMAT="INVALID_HASH_FORMAT"; _ERR_HASH_MISMATCH="HASH_MISMATCH"
_ERR_INVALID_SUBSYSTEM_TYPE="INVALID_SUBSYSTEM_TYPE"; _ERR_INVALID_SUBSYSTEM_LABEL="INVALID_SUBSYSTEM_LABEL"
_ERR_INVALID_SUBSYSTEM_STRESS_CLASS="INVALID_SUBSYSTEM_STRESS_CLASS"; _ERR_INVALID_SUBSYSTEM_STABILITY_CLASS="INVALID_SUBSYSTEM_STABILITY_CLASS"
_ERR_SUBSYSTEM_CLASSIFICATION_AMBIGUOUS="SUBSYSTEM_CLASSIFICATION_AMBIGUOUS"; _ERR_SUBSYSTEM_COUNT_MISMATCH="SUBSYSTEM_COUNT_MISMATCH"
_ERR_DUPLICATE_SUBSYSTEM_ENTRY="DUPLICATE_SUBSYSTEM_ENTRY"; _ERR_SUBSYSTEM_RECEIPT_MISMATCH="SUBSYSTEM_RECEIPT_MISMATCH"; _ERR_STABILITY_CLASS_MISMATCH="STABILITY_CLASS_MISMATCH"; _ERR_IMPACT_SCORE_MISMATCH="IMPACT_SCORE_MISMATCH"
_MAX_SUBSYSTEM_ENTRIES=10_000; _MAX_SUBSYSTEM_LABEL_LENGTH=96; _MAX_ABS_TOTAL_IMPACT_SCORE=1_000_000_000_000
_SHA256_RE=re.compile(r"^[0-9a-f]{64}$"); _LABEL_RE=re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_TYPES=frozenset({"LAYER","ROUTER","MASK","SHIFT","READOUT"})
_ALLOWED_STRESS=frozenset({"SUBSYSTEM_STRESS_STABLE","SUBSYSTEM_STRESS_CHANGED","SUBSYSTEM_STRESS_HEAVY","SUBSYSTEM_STRESS_EMPTY"})
_ALLOWED_STABILITY=frozenset({"SUBSYSTEM_STABILITY_STABLE","SUBSYSTEM_STABILITY_CHANGED","SUBSYSTEM_STABILITY_HEAVY","SUBSYSTEM_STABILITY_EMPTY"})

def get_allowed_stress_subsystem_types()->frozenset[str]: return _ALLOWED_TYPES
def get_allowed_subsystem_stress_classes()->frozenset[str]: return _ALLOWED_STRESS
def get_allowed_subsystem_stability_classes()->frozenset[str]: return _ALLOWED_STABILITY

def _vsha(v:object)->None:
    if not isinstance(v,str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)
def _vint(v:object,err:str,minv:int|None=None,maxv:int|None=None)->int:
    if not isinstance(v,int) or isinstance(v,bool): raise ValueError(err)
    if minv is not None and v<minv: raise ValueError(err)
    if maxv is not None and v>maxv: raise ValueError(err)
    return v

def _classify_entry_subsystem(entry:PerturbationMatrixEntry)->str|None:
    if not isinstance(entry,PerturbationMatrixEntry): raise ValueError(_ERR_INVALID_INPUT)
    t=entry.target_artifact_type; m=[]
    if t.startswith("Layer") or "Layered" in t or t.startswith("DecayLayer") or t=="LayerActivation": m.append("LAYER")
    if t.startswith("Router") or "Router" in t: m.append("ROUTER")
    if t.startswith("Mask") or "SearchMask" in t or "MaskCollision" in t: m.append("MASK")
    if t.startswith("Shift") or "Hilber" in t or "Hilbert" in t: m.append("SHIFT")
    if t.startswith("Readout") or "Readout" in t: m.append("READOUT")
    if len(m)>1: raise ValueError(_ERR_SUBSYSTEM_CLASSIFICATION_AMBIGUOUS)
    return m[0] if m else None

def _select_entries_for_subsystem(perturbation_matrix:PerturbationMatrix, subsystem_type:str)->tuple[PerturbationMatrixEntry,...]:
    if subsystem_type not in _ALLOWED_TYPES: raise ValueError(_ERR_INVALID_SUBSYSTEM_TYPE)
    return tuple(e for e in perturbation_matrix.entries if _classify_entry_subsystem(e)==subsystem_type)

def _derive_stress(c:int,ch:int,total:int)->str:
    if c==0: return "SUBSYSTEM_STRESS_EMPTY"
    if total==0 and ch==0: return "SUBSYSTEM_STRESS_STABLE"
    if total>0 and ch>0 and total<100: return "SUBSYSTEM_STRESS_CHANGED"
    if total>=100 and ch>0: return "SUBSYSTEM_STRESS_HEAVY"
    return "SUBSYSTEM_STRESS_STABLE"
def _derive_stability(s:str)->tuple[str,bool]:
    return {"SUBSYSTEM_STRESS_EMPTY":("SUBSYSTEM_STABILITY_EMPTY",True),"SUBSYSTEM_STRESS_STABLE":("SUBSYSTEM_STABILITY_STABLE",True),"SUBSYSTEM_STRESS_CHANGED":("SUBSYSTEM_STABILITY_CHANGED",False),"SUBSYSTEM_STRESS_HEAVY":("SUBSYSTEM_STABILITY_HEAVY",False)}[s]
def _subsystem_stress_receipt_payload(perturbation_matrix_hash:str,energy_matrix_receipt_hash:str,semantic_stress_receipt_hash:str,perturbation_stability_proof_hash:str,subsystem_type:str,subsystem_label:str,subsystem_entry_hashes:tuple[str,...],subsystem_entry_count:int,changed_entry_count:int,unchanged_entry_count:int,total_integer_impact_score:int,subsystem_stress_class:str,subsystem_stability_class:str,replay_stable:bool)->dict[str,Any]:
    return {"perturbation_matrix_hash":perturbation_matrix_hash,"energy_matrix_receipt_hash":energy_matrix_receipt_hash,"semantic_stress_receipt_hash":semantic_stress_receipt_hash,"perturbation_stability_proof_hash":perturbation_stability_proof_hash,"subsystem_type":subsystem_type,"subsystem_label":subsystem_label,"subsystem_entry_hashes":list(subsystem_entry_hashes),"subsystem_entry_count":subsystem_entry_count,"changed_entry_count":changed_entry_count,"unchanged_entry_count":unchanged_entry_count,"total_integer_impact_score":total_integer_impact_score,"subsystem_stress_class":subsystem_stress_class,"subsystem_stability_class":subsystem_stability_class,"replay_stable":replay_stable}

def _common_build(pm,em,sr,ps,stype,slabel):
    validate_perturbation_matrix(pm); validate_energy_matrix_receipt(em); validate_semantic_stress_receipt_with_matrix(sr,pm,em); validate_perturbation_stability_proof_with_receipts(ps,pm,em,sr)
    sel=_select_entries_for_subsystem(pm,stype); hs=tuple(e.perturbation_matrix_entry_hash for e in sel); ch=sum(1 for e in sel if e.changed); un=len(sel)-ch; total=sum(e.integer_impact_score for e in sel)
    sc=_derive_stress(len(sel),ch,total); stb,replay=_derive_stability(sc)
    p=_subsystem_stress_receipt_payload(pm.perturbation_matrix_hash,em.energy_matrix_receipt_hash,sr.semantic_stress_receipt_hash,ps.perturbation_stability_proof_hash,stype,slabel,hs,len(sel),ch,un,total,sc,stb,replay)
    return p,sha256_hex(p)

def _validate_core(r,etype,elabel,hfield):
    _vsha(getattr(r,hfield)); payload={f.name:getattr(r,f.name) for f in fields(r) if f.name!=hfield}
    for k in ("perturbation_matrix_hash","energy_matrix_receipt_hash","semantic_stress_receipt_hash","perturbation_stability_proof_hash"): _vsha(payload[k])
    if payload["subsystem_type"]!=etype: raise ValueError(_ERR_INVALID_SUBSYSTEM_TYPE)
    lab=payload["subsystem_label"]
    if not isinstance(lab,str) or not lab or len(lab)>_MAX_SUBSYSTEM_LABEL_LENGTH or _LABEL_RE.fullmatch(lab) is None or lab!=elabel: raise ValueError(_ERR_INVALID_SUBSYSTEM_LABEL)
    hs=payload["subsystem_entry_hashes"]
    if not isinstance(hs,tuple) or len(hs)>_MAX_SUBSYSTEM_ENTRIES: raise ValueError(_ERR_INVALID_INPUT)
    seen=set()
    for h in hs:
        _vsha(h)
        if h in seen: raise ValueError(_ERR_DUPLICATE_SUBSYSTEM_ENTRY)
        seen.add(h)
    c=_vint(payload["subsystem_entry_count"],_ERR_SUBSYSTEM_COUNT_MISMATCH,0); ch=_vint(payload["changed_entry_count"],_ERR_SUBSYSTEM_COUNT_MISMATCH,0); un=_vint(payload["unchanged_entry_count"],_ERR_SUBSYSTEM_COUNT_MISMATCH,0); total=_vint(payload["total_integer_impact_score"],_ERR_IMPACT_SCORE_MISMATCH)
    if c!=len(hs) or ch+un!=c: raise ValueError(_ERR_SUBSYSTEM_COUNT_MISMATCH)
    if abs(total)>_MAX_ABS_TOTAL_IMPACT_SCORE: raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    exp_s=_derive_stress(c,ch,total)
    if payload["subsystem_stress_class"] not in _ALLOWED_STRESS: raise ValueError(_ERR_INVALID_SUBSYSTEM_STRESS_CLASS)
    if payload["subsystem_stress_class"]!=exp_s: raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    exp_stb,exp_rep=_derive_stability(exp_s)
    if payload["subsystem_stability_class"] not in _ALLOWED_STABILITY: raise ValueError(_ERR_INVALID_SUBSYSTEM_STABILITY_CLASS)
    if payload["subsystem_stability_class"]!=exp_stb or payload["replay_stable"] is not exp_rep: raise ValueError(_ERR_STABILITY_CLASS_MISMATCH)
    base_payload=_subsystem_stress_receipt_payload(payload["perturbation_matrix_hash"],payload["energy_matrix_receipt_hash"],payload["semantic_stress_receipt_hash"],payload["perturbation_stability_proof_hash"],payload["subsystem_type"],payload["subsystem_label"],payload["subsystem_entry_hashes"],payload["subsystem_entry_count"],payload["changed_entry_count"],payload["unchanged_entry_count"],payload["total_integer_impact_score"],payload["subsystem_stress_class"],payload["subsystem_stability_class"],payload["replay_stable"])
    if sha256_hex(base_payload)!=getattr(r,hfield): raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_layer_activation_stability_receipt(r:"LayerActivationStabilityReceipt")->bool:
    return _validate_core(r,"LAYER","LAYER_ACTIVATION","layer_activation_stability_receipt_hash")

def validate_router_stress_receipt(r:"RouterStressReceipt")->bool:
    if not _validate_core(r,"ROUTER","ROUTER","layer_activation_stability_receipt_hash"): return False
    full_payload={f.name:getattr(r,f.name) for f in fields(r) if f.name!="router_stress_receipt_hash"}
    if sha256_hex(full_payload)!=r.router_stress_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_mask_stress_receipt(r:"MaskStressReceipt")->bool:
    if not _validate_core(r,"MASK","MASK","layer_activation_stability_receipt_hash"): return False
    full_payload={f.name:getattr(r,f.name) for f in fields(r) if f.name!="mask_stress_receipt_hash"}
    if sha256_hex(full_payload)!=r.mask_stress_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_shift_stress_receipt(r:"ShiftStressReceipt")->bool:
    if not _validate_core(r,"SHIFT","SHIFT","layer_activation_stability_receipt_hash"): return False
    full_payload={f.name:getattr(r,f.name) for f in fields(r) if f.name!="shift_stress_receipt_hash"}
    if sha256_hex(full_payload)!=r.shift_stress_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_readout_stress_receipt(r:"ReadoutStressReceipt")->bool:
    if not _validate_core(r,"READOUT","READOUT","layer_activation_stability_receipt_hash"): return False
    full_payload={f.name:getattr(r,f.name) for f in fields(r) if f.name!="readout_stress_receipt_hash"}
    if sha256_hex(full_payload)!=r.readout_stress_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True


@dataclass(frozen=True)
class LayerActivationStabilityReceipt:
    perturbation_matrix_hash:str; energy_matrix_receipt_hash:str; semantic_stress_receipt_hash:str; perturbation_stability_proof_hash:str; subsystem_type:str; subsystem_label:str; subsystem_entry_hashes:tuple[str,...]; subsystem_entry_count:int; changed_entry_count:int; unchanged_entry_count:int; total_integer_impact_score:int; subsystem_stress_class:str; subsystem_stability_class:str; replay_stable:bool; layer_activation_stability_receipt_hash:str
    def __post_init__(self): validate_layer_activation_stability_receipt(self)
    def to_dict(self): return {**_subsystem_stress_receipt_payload(self.perturbation_matrix_hash,self.energy_matrix_receipt_hash,self.semantic_stress_receipt_hash,self.perturbation_stability_proof_hash,self.subsystem_type,self.subsystem_label,self.subsystem_entry_hashes,self.subsystem_entry_count,self.changed_entry_count,self.unchanged_entry_count,self.total_integer_impact_score,self.subsystem_stress_class,self.subsystem_stability_class,self.replay_stable),"layer_activation_stability_receipt_hash":self.layer_activation_stability_receipt_hash}
    def to_canonical_json(self): return canonical_json(self.to_dict())
    def to_canonical_bytes(self): return canonical_bytes(self.to_dict())
# repeat classes
@dataclass(frozen=True)
class RouterStressReceipt(LayerActivationStabilityReceipt):
    router_stress_receipt_hash:str
    def __post_init__(self): validate_router_stress_receipt(self)
    def to_dict(self): return {**super().to_dict(),"router_stress_receipt_hash":self.router_stress_receipt_hash}
@dataclass(frozen=True)
class MaskStressReceipt(LayerActivationStabilityReceipt):
    mask_stress_receipt_hash:str
    def __post_init__(self): validate_mask_stress_receipt(self)
    def to_dict(self): return {**super().to_dict(),"mask_stress_receipt_hash":self.mask_stress_receipt_hash}
@dataclass(frozen=True)
class ShiftStressReceipt(LayerActivationStabilityReceipt):
    shift_stress_receipt_hash:str
    def __post_init__(self): validate_shift_stress_receipt(self)
    def to_dict(self): return {**super().to_dict(),"shift_stress_receipt_hash":self.shift_stress_receipt_hash}
@dataclass(frozen=True)
class ReadoutStressReceipt(LayerActivationStabilityReceipt):
    readout_stress_receipt_hash:str
    def __post_init__(self): validate_readout_stress_receipt(self)
    def to_dict(self): return {**super().to_dict(),"readout_stress_receipt_hash":self.readout_stress_receipt_hash}

def build_layer_activation_stability_receipt(pm:PerturbationMatrix,em:EnergyMatrixReceipt,sr:SemanticStressReceipt,ps:PerturbationStabilityProof)->LayerActivationStabilityReceipt:
    p,h=_common_build(pm,em,sr,ps,"LAYER","LAYER_ACTIVATION")
    p["subsystem_entry_hashes"]=tuple(p["subsystem_entry_hashes"])
    return LayerActivationStabilityReceipt(**p,layer_activation_stability_receipt_hash=h)

def build_router_stress_receipt(pm:PerturbationMatrix,em:EnergyMatrixReceipt,sr:SemanticStressReceipt,ps:PerturbationStabilityProof)->RouterStressReceipt:
    p,h=_common_build(pm,em,sr,ps,"ROUTER","ROUTER")
    p["subsystem_entry_hashes"]=tuple(p["subsystem_entry_hashes"])
    full={**p,"layer_activation_stability_receipt_hash":h,"router_stress_receipt_hash":sha256_hex({**p,"layer_activation_stability_receipt_hash":h})}
    return RouterStressReceipt(**full)

def build_mask_stress_receipt(pm:PerturbationMatrix,em:EnergyMatrixReceipt,sr:SemanticStressReceipt,ps:PerturbationStabilityProof)->MaskStressReceipt:
    p,h=_common_build(pm,em,sr,ps,"MASK","MASK")
    p["subsystem_entry_hashes"]=tuple(p["subsystem_entry_hashes"])
    full={**p,"layer_activation_stability_receipt_hash":h,"mask_stress_receipt_hash":sha256_hex({**p,"layer_activation_stability_receipt_hash":h})}
    return MaskStressReceipt(**full)

def build_shift_stress_receipt(pm:PerturbationMatrix,em:EnergyMatrixReceipt,sr:SemanticStressReceipt,ps:PerturbationStabilityProof)->ShiftStressReceipt:
    p,h=_common_build(pm,em,sr,ps,"SHIFT","SHIFT")
    p["subsystem_entry_hashes"]=tuple(p["subsystem_entry_hashes"])
    full={**p,"layer_activation_stability_receipt_hash":h,"shift_stress_receipt_hash":sha256_hex({**p,"layer_activation_stability_receipt_hash":h})}
    return ShiftStressReceipt(**full)

def build_readout_stress_receipt(pm:PerturbationMatrix,em:EnergyMatrixReceipt,sr:SemanticStressReceipt,ps:PerturbationStabilityProof)->ReadoutStressReceipt:
    p,h=_common_build(pm,em,sr,ps,"READOUT","READOUT")
    p["subsystem_entry_hashes"]=tuple(p["subsystem_entry_hashes"])
    full={**p,"layer_activation_stability_receipt_hash":h,"readout_stress_receipt_hash":sha256_hex({**p,"layer_activation_stability_receipt_hash":h})}
    return ReadoutStressReceipt(**full)
