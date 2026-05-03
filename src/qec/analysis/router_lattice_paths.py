from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph, _validate_graph_internal_consistency
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe
_ALLOWED_TOKEN_TYPES={"NODE_ID","EDGE_ID","CONSTRAINT_TYPE","COORDINATE"}
_ALLOWED_EMPTY_REASONS={"NONE","NO_MATCH","AMBIGUOUS"}
_ALLOWED_RULE_KEYS={"allow_empty_result","require_connected_sequence","max_paths","path_ordering"}
def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({k:_deep_freeze(mapping[k]) for k in sorted(mapping)})
def _validate_graph(graph: SemanticLatticeGraph)->None:
    _validate_graph_internal_consistency(graph)
    if graph.graph_hash!=graph.stable_hash(): raise ValueError("INVALID_INPUT")
@dataclass(frozen=True)
class RouteToken:
    token_id:str; token_type:str; token_value:Any; token_index:int; token_hash:str
    def __post_init__(self)->None:
        if not self.token_id or self.token_type not in _ALLOWED_TOKEN_TYPES: raise ValueError("INVALID_INPUT")
        if not isinstance(self.token_index,int) or isinstance(self.token_index,bool) or self.token_index<0: raise ValueError("INVALID_INPUT")
        if self.token_type=="COORDINATE":
            if not isinstance(self.token_value,Mapping) or set(self.token_value.keys())!={"x","y","z"}: raise ValueError("INVALID_INPUT")
            for k in ("x","y","z"):
                v=self.token_value[k]
                if not isinstance(v,int) or isinstance(v,bool): raise ValueError("INVALID_INPUT")
        elif self.token_type in ("NODE_ID","EDGE_ID","CONSTRAINT_TYPE"):
            if not isinstance(self.token_value,str): raise ValueError("INVALID_INPUT")
        _ensure_json_safe(self.token_value)
        if self.token_hash and self.token_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict:
        p={"token_id":self.token_id,"token_type":self.token_type,"token_value":self.token_value,"token_index":self.token_index}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["token_hash"]=self.token_hash; return p
    def to_canonical_json(self)->str: return canonical_json(self._canonical_payload())
    def stable_hash(self)->str: return sha256_hex(self._canonical_payload())
@dataclass(frozen=True)
class RouterPathSpec:
    route_id:str; route_version:str; tokens:tuple[RouteToken,...]; resolution_rules:Mapping[str,Any]; router_path_spec_hash:str
    def __post_init__(self)->None:
        if not self.route_id or not self.route_version: raise ValueError("INVALID_INPUT")
        object.__setattr__(self,"tokens",tuple(self.tokens)); object.__setattr__(self,"resolution_rules",_freeze_mapping(dict(self.resolution_rules)))
        if len({t.token_id for t in self.tokens})!=len(self.tokens) or len({t.token_index for t in self.tokens})!=len(self.tokens): raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.tokens,key=lambda t:(t.token_index,t.token_type,t.token_id,t.token_hash)))!=self.tokens: raise ValueError("INVALID_INPUT")
        rules=dict(self.resolution_rules)
        if set(rules)!={"allow_empty_result","require_connected_sequence","max_paths","path_ordering"}: raise ValueError("INVALID_INPUT")
        if not isinstance(rules["allow_empty_result"],bool) or not isinstance(rules["require_connected_sequence"],bool): raise ValueError("INVALID_INPUT")
        if not isinstance(rules["max_paths"],int) or isinstance(rules["max_paths"],bool) or rules["max_paths"]<=0: raise ValueError("INVALID_INPUT")
        if rules["path_ordering"]!="CANONICAL": raise ValueError("INVALID_INPUT")
        if self.router_path_spec_hash and self.router_path_spec_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict: p={"route_id":self.route_id,"route_version":self.route_version,"tokens":[t.to_dict() for t in self.tokens],"resolution_rules":dict(self.resolution_rules)}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["router_path_spec_hash"]=self.router_path_spec_hash; return p
    def to_canonical_json(self)->str: return canonical_json(self._canonical_payload())
    def stable_hash(self)->str: return sha256_hex(self._canonical_payload())
@dataclass(frozen=True)
class SpecialPathIndex:
    index_id:str; index_version:str; graph_hash:str; indexed_node_ids:tuple[str,...]; indexed_edge_ids:tuple[str,...]; index_hash:str
    def __post_init__(self)->None:
        if not self.index_id or not self.index_version or not self.graph_hash: raise ValueError("INVALID_INPUT")
        object.__setattr__(self,"indexed_node_ids",tuple(self.indexed_node_ids)); object.__setattr__(self,"indexed_edge_ids",tuple(self.indexed_edge_ids))
        if len(set(self.indexed_node_ids))!=len(self.indexed_node_ids) or len(set(self.indexed_edge_ids))!=len(self.indexed_edge_ids): raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.indexed_node_ids))!=self.indexed_node_ids or tuple(sorted(self.indexed_edge_ids))!=self.indexed_edge_ids: raise ValueError("INVALID_INPUT")
        if self.index_hash and self.index_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict: p={"index_id":self.index_id,"index_version":self.index_version,"graph_hash":self.graph_hash,"indexed_node_ids":list(self.indexed_node_ids),"indexed_edge_ids":list(self.indexed_edge_ids)}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["index_hash"]=self.index_hash; return p
    def to_canonical_json(self)->str: return canonical_json(self._canonical_payload())
    def stable_hash(self)->str: return sha256_hex(self._canonical_payload())
@dataclass(frozen=True)
class ResolvedLatticePath:
    path_id:str; matched_token_hashes:tuple[str,...]; node_ids:tuple[str,...]; edge_ids:tuple[str,...]; path_hash:str
    def __post_init__(self)->None:
        if not self.path_id: raise ValueError("INVALID_INPUT")
        if self.path_hash and self.path_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict: p={"path_id":self.path_id,"matched_token_hashes":list(self.matched_token_hashes),"node_ids":list(self.node_ids),"edge_ids":list(self.edge_ids)}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["path_hash"]=self.path_hash; return p
    def to_canonical_json(self)->str:return canonical_json(self._canonical_payload())
    def stable_hash(self)->str:return sha256_hex(self._canonical_payload())
@dataclass(frozen=True)
class ResolvedLatticePathSet:
    graph_hash:str; router_path_spec_hash:str; special_path_index_hash:str; resolved_paths:tuple[ResolvedLatticePath,...]; resolved_path_hash:str; empty_result_reason:str
    def __post_init__(self)->None:
        object.__setattr__(self,"resolved_paths",tuple(self.resolved_paths))
        if self.empty_result_reason not in _ALLOWED_EMPTY_REASONS: raise ValueError("INVALID_INPUT")
        if self.resolved_paths and self.empty_result_reason!="NONE": raise ValueError("INVALID_INPUT")
        if not self.resolved_paths and self.empty_result_reason=="NONE": raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.resolved_paths,key=lambda p:(p.path_id,p.path_hash)))!=self.resolved_paths: raise ValueError("INVALID_INPUT")
        if len({p.path_id for p in self.resolved_paths})!=len(self.resolved_paths) or len({p.path_hash for p in self.resolved_paths})!=len(self.resolved_paths): raise ValueError("INVALID_INPUT")
        if self.resolved_path_hash and self.resolved_path_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict: p={"graph_hash":self.graph_hash,"router_path_spec_hash":self.router_path_spec_hash,"special_path_index_hash":self.special_path_index_hash,"resolved_paths":[x._canonical_payload() for x in self.resolved_paths],"empty_result_reason":self.empty_result_reason}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["resolved_path_hash"]=self.resolved_path_hash; return p
    def to_canonical_json(self)->str:return canonical_json(self._canonical_payload())
    def stable_hash(self)->str:return sha256_hex(self._canonical_payload())
@dataclass(frozen=True)
class RouterLatticePathReceipt:
    graph_hash:str; router_path_spec_hash:str; special_path_index_hash:str; resolved_path_hash:str; path_count:int; receipt_hash:str
    def __post_init__(self)->None:
        if not isinstance(self.path_count,int) or self.path_count<0: raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash!=self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict: p={"graph_hash":self.graph_hash,"router_path_spec_hash":self.router_path_spec_hash,"special_path_index_hash":self.special_path_index_hash,"resolved_path_hash":self.resolved_path_hash,"path_count":self.path_count}; _ensure_json_safe(p); return p
    def to_dict(self)->dict: p=dict(self._canonical_payload()); p["receipt_hash"]=self.receipt_hash; return p
    def to_canonical_json(self)->str:return canonical_json(self._canonical_payload())
    def stable_hash(self)->str:return sha256_hex(self._canonical_payload())
def build_router_path_spec(route_id:str,route_version:str,tokens:tuple[RouteToken,...],resolution_rules:Mapping[str,Any])->RouterPathSpec:
    for t in tokens:
        if t.token_hash!=t.stable_hash(): raise ValueError("INVALID_INPUT")
    st=tuple(sorted(tokens,key=lambda t:(t.token_index,t.token_type,t.token_id,t.token_hash)))
    s=RouterPathSpec(route_id,route_version,st,resolution_rules,"")
    return RouterPathSpec(**{**s.__dict__,"router_path_spec_hash":s.stable_hash()})
def build_special_path_index(graph:SemanticLatticeGraph,index_id:str,index_version:str,indexed_node_ids:tuple[str,...],indexed_edge_ids:tuple[str,...])->SpecialPathIndex:
    _validate_graph(graph)
    ns=tuple(sorted(indexed_node_ids)); es=tuple(sorted(indexed_edge_ids))
    node_ids={n.node_id for n in graph.nodes}; edge_ids={e.edge_id for e in graph.edges}
    if any(n not in node_ids for n in ns) or any(e not in edge_ids for e in es): raise ValueError("INVALID_INPUT")
    i=SpecialPathIndex(index_id,index_version,graph.stable_hash(),ns,es,"")
    return SpecialPathIndex(**{**i.__dict__,"index_hash":i.stable_hash()})
def resolve_router_lattice_paths(graph:SemanticLatticeGraph,router_path_spec:RouterPathSpec,special_path_index:SpecialPathIndex)->ResolvedLatticePathSet:
    _validate_graph(graph)
    if graph.stable_hash()!=special_path_index.graph_hash: raise ValueError("INVALID_INPUT")
    if router_path_spec.stable_hash()!=router_path_spec.router_path_spec_hash or special_path_index.stable_hash()!=special_path_index.index_hash: raise ValueError("INVALID_INPUT")
    node_by_id={n.node_id:n for n in graph.nodes if n.node_id in special_path_index.indexed_node_ids}; edge_by_id={e.edge_id:e for e in graph.edges if e.edge_id in special_path_index.indexed_edge_ids}
    matched_nodes=[]; matched_edges=[]; matched_hashes=[]
    for t in router_path_spec.tokens:
        if t.token_type=="NODE_ID":
            found=[node_id for node_id in node_by_id if node_id==t.token_value]
            if len(found)!=1: raise ValueError("INVALID_INPUT")
            matched_nodes.append(found[0]); matched_hashes.append(t.token_hash)
        elif t.token_type=="EDGE_ID":
            found=[edge_id for edge_id in edge_by_id if edge_id==t.token_value]
            if len(found)!=1: raise ValueError("INVALID_INPUT")
            matched_edges.append(found[0]); matched_hashes.append(t.token_hash)
        elif t.token_type=="COORDINATE":
            coord=(t.token_value["x"],t.token_value["y"],t.token_value["z"])
            found=[n.node_id for n in node_by_id.values() if n.coordinate==coord]
            if len(found)!=1: raise ValueError("INVALID_INPUT")
            matched_nodes.append(found[0]); matched_hashes.append(t.token_hash)
        else:
            cons=sorted([e.edge_id for e in edge_by_id.values() if e.constraint_type==t.token_value])
            if len(cons)==0: raise ValueError("INVALID_INPUT")
            if len(cons)>router_path_spec.resolution_rules["max_paths"]: raise ValueError("INVALID_INPUT")
            matched_edges.extend(cons); matched_hashes.append(t.token_hash)
    if router_path_spec.resolution_rules["require_connected_sequence"]:
        if len(matched_nodes)>1:
            for a,b in zip(matched_nodes,matched_nodes[1:]):
                direct=[e.edge_id for e in edge_by_id.values() if e.source_node_id==a and e.target_node_id==b]
                if len(direct)!=1: raise ValueError("INVALID_INPUT")
                if direct[0] not in matched_edges: matched_edges.append(direct[0])
        if matched_edges:
            edge_objs=[e for e in edge_by_id.values() if e.edge_id in matched_edges]
            edge_nodes=set()
            for e in edge_objs: edge_nodes.add(e.source_node_id); edge_nodes.add(e.target_node_id)
            for n in matched_nodes:
                if n not in edge_nodes: raise ValueError("INVALID_INPUT")
    if not matched_nodes and not matched_edges:
        if not router_path_spec.resolution_rules["allow_empty_result"]: raise ValueError("INVALID_INPUT")
        rps=ResolvedLatticePathSet(graph.stable_hash(),router_path_spec.stable_hash(),special_path_index.stable_hash(),tuple(),"","NO_MATCH")
        return ResolvedLatticePathSet(**{**rps.__dict__,"resolved_path_hash":rps.stable_hash()})
    path=ResolvedLatticePath("path-0",tuple(matched_hashes),tuple(matched_nodes),tuple(matched_edges),"")
    path=ResolvedLatticePath(**{**path.__dict__,"path_hash":path.stable_hash()})
    rps=ResolvedLatticePathSet(graph.stable_hash(),router_path_spec.stable_hash(),special_path_index.stable_hash(),(path,),"","NONE")
    return ResolvedLatticePathSet(**{**rps.__dict__,"resolved_path_hash":rps.stable_hash()})
def build_router_lattice_path_receipt(graph:SemanticLatticeGraph,router_path_spec:RouterPathSpec,special_path_index:SpecialPathIndex,resolved_path_set:ResolvedLatticePathSet)->RouterLatticePathReceipt:
    _validate_graph(graph)
    if resolved_path_set.graph_hash!=graph.stable_hash(): raise ValueError("INVALID_INPUT")
    if resolved_path_set.router_path_spec_hash!=router_path_spec.stable_hash(): raise ValueError("INVALID_INPUT")
    if resolved_path_set.special_path_index_hash!=special_path_index.stable_hash(): raise ValueError("INVALID_INPUT")
    if resolved_path_set.resolved_path_hash!=resolved_path_set.stable_hash(): raise ValueError("INVALID_INPUT")
    r=RouterLatticePathReceipt(graph.stable_hash(),router_path_spec.stable_hash(),special_path_index.stable_hash(),resolved_path_set.resolved_path_hash,len(resolved_path_set.resolved_paths),"")
    return RouterLatticePathReceipt(**{**r.__dict__,"receipt_hash":r.stable_hash()})
def validate_router_lattice_path_receipt(receipt:RouterLatticePathReceipt,graph:SemanticLatticeGraph,router_path_spec:RouterPathSpec,special_path_index:SpecialPathIndex,resolved_path_set:ResolvedLatticePathSet)->None:
    _validate_graph(graph)
    if receipt.graph_hash!=graph.stable_hash() or receipt.router_path_spec_hash!=router_path_spec.stable_hash() or receipt.special_path_index_hash!=special_path_index.stable_hash(): raise ValueError("INVALID_INPUT")
    if resolved_path_set.resolved_path_hash!=resolved_path_set.stable_hash(): raise ValueError("INVALID_INPUT")
    if receipt.resolved_path_hash!=resolved_path_set.resolved_path_hash or receipt.path_count!=len(resolved_path_set.resolved_paths): raise ValueError("INVALID_INPUT")
    if resolved_path_set.graph_hash!=graph.stable_hash() or resolved_path_set.router_path_spec_hash!=router_path_spec.stable_hash() or resolved_path_set.special_path_index_hash!=special_path_index.stable_hash(): raise ValueError("INVALID_INPUT")
    if receipt.receipt_hash!=RouterLatticePathReceipt(**{**receipt.__dict__,"receipt_hash":""}).stable_hash(): raise ValueError("INVALID_INPUT")
for _n in ("apply","execute","run","traverse","search","shortest_path","readout","project","render"):
    for _cls in (RouterPathSpec,SpecialPathIndex,ResolvedLatticePathSet,RouterLatticePathReceipt):
        if hasattr(_cls,_n): raise RuntimeError("INVALID_INPUT")
