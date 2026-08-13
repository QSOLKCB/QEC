# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping
from qec.sonify.canonical import canonical_sha256, require_int, require_nonempty_text, validate_sha256

QEC_VERSION="172.0.0"; CONTRACT_VERSION="172.0"
MATRIX_SCHEMA="qec.crossbar-matrix-manifest.v1"; INTERSECTION_ID_SCHEMA="qec.crossbar-intersection-id.v1"; VALIDATION_SCHEMA="qec.crossbar-matrix-validation.v1"
LINK_STATES: Final=("busy","idle","quarantined","unavailable"); _MAX=4096; _MAX_X=65536
_CB={"classical_software_model_only":True,"physical_crossbar_fidelity":False,"marker_authority_present":False,"route_search_present":False,"reservation_present":False,"connection_commit_present":False,"decoder_output_mutation_permitted":False,"payload_mutation_permitted":False,"browser_demo_is_canonical_evidence":False,"receipt_proves":"immutable_matrix_identity_and_declared_initial_link_state","receipt_does_not_prove":"end_to_end_route_continuity_or_physical_network_behavior"}
CLAIM_BOUNDARY: Final[Mapping[str,object]]=MappingProxyType(_CB)

def _obj(x, fields, label):
    if not isinstance(x,dict) or set(x)!=fields: raise ValueError(f"{label} must contain exactly the canonical fields")
    return x

def _list(x,label):
    if not isinstance(x,list): raise ValueError(f"{label} must be a list")
    return x

def _hashed(x):
    if not isinstance(x,dict) or x.get("schema")!=MATRIX_SCHEMA: raise ValueError("unexpected crossbar matrix manifest schema")
    h=validate_sha256(x.get("sha256"),"crossbar matrix manifest.sha256"); u=dict(x); u.pop("sha256",None)
    if canonical_sha256(u)!=h: raise ValueError("crossbar matrix manifest hash mismatch")
    return x

@dataclass(frozen=True)
class CrossbarLink:
    axis:str; link_id:str; ordinal:int; state:str="idle"
    def __post_init__(self):
        if self.axis not in ("horizontal","vertical"): raise ValueError("link.axis must be horizontal or vertical")
        require_nonempty_text(self.link_id,"link.link_id"); require_int(self.ordinal,"link.ordinal",minimum=0,maximum=_MAX-1)
        if self.state not in LINK_STATES: raise ValueError(f"link.state must be one of {LINK_STATES}")
    def as_dict(self): return {"axis":self.axis,"link_id":self.link_id,"ordinal":self.ordinal,"state":self.state}
    @classmethod
    def from_dict(cls,x):
        x=_obj(x,{"axis","link_id","ordinal","state"},"crossbar link"); return cls(x["axis"],x["link_id"],x["ordinal"],x["state"])

@dataclass(frozen=True)
class CrossbarIntersection:
    matrix_id:str; horizontal_link_id:str; horizontal_ordinal:int; vertical_link_id:str; vertical_ordinal:int
    def __post_init__(self):
        require_nonempty_text(self.matrix_id,"intersection.matrix_id"); require_nonempty_text(self.horizontal_link_id,"intersection.horizontal_link_id"); require_nonempty_text(self.vertical_link_id,"intersection.vertical_link_id")
        require_int(self.horizontal_ordinal,"intersection.horizontal_ordinal",minimum=0,maximum=_MAX-1); require_int(self.vertical_ordinal,"intersection.vertical_ordinal",minimum=0,maximum=_MAX-1)
    @property
    def intersection_id(self): return canonical_sha256({"schema":INTERSECTION_ID_SCHEMA,"contract_version":CONTRACT_VERSION,"matrix_id":self.matrix_id,"horizontal_link_id":self.horizontal_link_id,"horizontal_ordinal":self.horizontal_ordinal,"vertical_link_id":self.vertical_link_id,"vertical_ordinal":self.vertical_ordinal})
    def as_dict(self): return {"horizontal_link_id":self.horizontal_link_id,"horizontal_ordinal":self.horizontal_ordinal,"vertical_link_id":self.vertical_link_id,"vertical_ordinal":self.vertical_ordinal,"intersection_id":self.intersection_id}
    @classmethod
    def from_dict(cls,m,x):
        x=_obj(x,{"horizontal_link_id","horizontal_ordinal","vertical_link_id","vertical_ordinal","intersection_id"},"crossbar intersection"); r=cls(m,x["horizontal_link_id"],x["horizontal_ordinal"],x["vertical_link_id"],x["vertical_ordinal"])
        if r.intersection_id!=validate_sha256(x["intersection_id"],"intersection.intersection_id"): raise ValueError("crossbar intersection identity mismatch")
        return r

@dataclass(frozen=True)
class CrossbarMatrix:
    matrix_id:str; horizontal_links:tuple[CrossbarLink,...]; vertical_links:tuple[CrossbarLink,...]
    def __post_init__(self):
        object.__setattr__(self,"horizontal_links",tuple(self.horizontal_links)); object.__setattr__(self,"vertical_links",tuple(self.vertical_links))
        require_nonempty_text(self.matrix_id,"matrix.matrix_id")
        if not self.horizontal_links or not self.vertical_links: raise ValueError("crossbar matrix requires horizontal and vertical links")
        if max(len(self.horizontal_links),len(self.vertical_links))>_MAX or len(self.horizontal_links)*len(self.vertical_links)>_MAX_X: raise ValueError("crossbar matrix exceeds bounded size")
        self._axis(self.horizontal_links,"horizontal"); self._axis(self.vertical_links,"vertical")
        if {x.link_id for x in self.horizontal_links}&{x.link_id for x in self.vertical_links}: raise ValueError("crossbar link ids must be globally unique across axes")
    @staticmethod
    def _axis(xs,axis):
        if any(x.axis!=axis for x in xs): raise ValueError(f"{axis} link collection contains wrong-axis record")
        if tuple(x.ordinal for x in xs)!=tuple(range(len(xs))): raise ValueError(f"{axis} links must use contiguous canonical ordinal order")
        ids=tuple(x.link_id for x in xs)
        if len(ids)!=len(set(ids)): raise ValueError(f"{axis} link ids must be unique")
    @property
    def intersections(self): return tuple(CrossbarIntersection(self.matrix_id,h.link_id,h.ordinal,v.link_id,v.ordinal) for h in self.horizontal_links for v in self.vertical_links)
    def coordinate(self,hid,vid):
        require_nonempty_text(hid,"horizontal_link_id"); require_nonempty_text(vid,"vertical_link_id"); h=next((x for x in self.horizontal_links if x.link_id==hid),None); v=next((x for x in self.vertical_links if x.link_id==vid),None)
        if h is None or v is None: raise ValueError("unknown crossbar coordinate")
        return CrossbarIntersection(self.matrix_id,h.link_id,h.ordinal,v.link_id,v.ordinal)
    def as_dict(self):
        u={"schema":MATRIX_SCHEMA,"contract_version":CONTRACT_VERSION,"matrix_id":self.matrix_id,"link_state_vocabulary":list(LINK_STATES),"horizontal_links":[x.as_dict() for x in self.horizontal_links],"vertical_links":[x.as_dict() for x in self.vertical_links],"intersections":[x.as_dict() for x in self.intersections],"claim_boundary":dict(CLAIM_BOUNDARY)}; return {**u,"sha256":canonical_sha256(u)}
    def sha256(self): return self.as_dict()["sha256"]
    @classmethod
    def from_dict(cls,x):
        x=_hashed(x); fields={"schema","contract_version","matrix_id","link_state_vocabulary","horizontal_links","vertical_links","intersections","claim_boundary","sha256"}
        if set(x)!=fields: raise ValueError("crossbar matrix manifest fields are not canonical")
        if x["contract_version"]!=CONTRACT_VERSION: raise ValueError("unexpected crossbar matrix contract version")
        if x["link_state_vocabulary"]!=list(LINK_STATES): raise ValueError("crossbar link-state vocabulary is not canonical")
        cb=x["claim_boundary"]
        if not isinstance(cb,dict) or set(cb)!=set(CLAIM_BOUNDARY) or canonical_sha256(cb)!=canonical_sha256(dict(CLAIM_BOUNDARY)): raise ValueError("crossbar claim boundary mismatch")
        m=cls(x["matrix_id"],tuple(CrossbarLink.from_dict(v) for v in _list(x["horizontal_links"],"horizontal links")),tuple(CrossbarLink.from_dict(v) for v in _list(x["vertical_links"],"vertical links")))
        if tuple(CrossbarIntersection.from_dict(m.matrix_id,v) for v in _list(x["intersections"],"crossbar intersections"))!=m.intersections: raise ValueError("crossbar intersections must be complete canonical row-major closure")
        if m.sha256()!=validate_sha256(x["sha256"],"crossbar matrix manifest.sha256"): raise ValueError("crossbar matrix manifest is not exact canonical replay")
        return m

def validate_matrix_manifest(x):
    m=CrossbarMatrix.from_dict(x); u={"schema":VALIDATION_SCHEMA,"contract_version":CONTRACT_VERSION,"matrix_id":m.matrix_id,"crossbar_matrix_receipt_hash":m.sha256(),"horizontal_link_count":len(m.horizontal_links),"vertical_link_count":len(m.vertical_links),"intersection_count":len(m.intersections),"canonical_ordering_verified":True,"full_coordinate_coverage_verified":True,"intersection_identities_verified":True,"claim_boundary_verified":True,"all_passed":True}; return {**u,"sha256":canonical_sha256(u)}

def demo_matrix(matrix_id="crossbar-demo",*,horizontal_count=4,vertical_count=4,state_overrides=None):
    require_nonempty_text(matrix_id,"matrix_id"); require_int(horizontal_count,"horizontal_count",minimum=1,maximum=_MAX); require_int(vertical_count,"vertical_count",minimum=1,maximum=_MAX)
    if horizontal_count*vertical_count>_MAX_X: raise ValueError("requested demo matrix exceeds bounded intersection count")
    o=dict(state_overrides or {}); hs=tuple(f"H{i:03d}" for i in range(horizontal_count)); vs=tuple(f"V{i:03d}" for i in range(vertical_count)); unknown=set(o)-(set(hs)|set(vs))
    if unknown: raise ValueError(f"state override references unknown links: {sorted(unknown)}")
    for k,v in o.items():
        require_nonempty_text(k,"state override link id")
        if v not in LINK_STATES: raise ValueError(f"state override for {k} must be one of {LINK_STATES}")
    return CrossbarMatrix(matrix_id,tuple(CrossbarLink("horizontal",k,i,o.get(k,"idle")) for i,k in enumerate(hs)),tuple(CrossbarLink("vertical",k,i,o.get(k,"idle")) for i,k in enumerate(vs)))
