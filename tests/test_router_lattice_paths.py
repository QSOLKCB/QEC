import pytest
from qec.analysis.atomic_semantic_lattice_contract import AtomicLatticeBounds, SemanticLatticeNode, ConstraintEdgeReceipt, build_semantic_lattice_graph
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.router_lattice_paths import (
    RouteToken,
    RouterPathSpec,
    SpecialPathIndex,
    ResolvedLatticePath,
    ResolvedLatticePathSet,
    RouterLatticePathReceipt,
    build_router_path_spec,
    build_special_path_index,
    resolve_router_lattice_paths,
    build_router_lattice_path_receipt,
    validate_router_lattice_path_receipt,
)

def _bounds():
    b=AtomicLatticeBounds("b","153.1",5,5,5,"")
    return AtomicLatticeBounds(**{**b.__dict__,"bounds_hash":b.stable_hash()})
def _node(i,c):
    n=SemanticLatticeNode(i,"semantic",c,"cr","sp",{},"")
    return SemanticLatticeNode(**{**n.__dict__,"node_hash":n.stable_hash()})
def _edge(e,s,t,ct="adjacent"):
    p={"k":"v","edge":e}; h=sha256_hex(p); x=ConstraintEdgeReceipt(e,s,t,ct,p,h,"",""); x=ConstraintEdgeReceipt(**{**x.__dict__,"edge_hash":x._edge_stable_hash()}); return ConstraintEdgeReceipt(**{**x.__dict__,"receipt_hash":x.stable_hash()})
def _graph():
    n1,n2,n3=_node("n1",(0,0,0)),_node("n2",(1,0,0)),_node("n3",(2,0,0))
    e1,e2=_edge("e1","n1","n2","adjacent"),_edge("e2","n2","n3","soft")
    return build_semantic_lattice_graph("l","153.1",_bounds(),(n1,n2,n3),(e1,e2))
def _tok(i,t,v,ix):
    r=RouteToken(i,t,v,ix,"")
    return RouteToken(**{**r.__dict__,"token_hash":r.stable_hash()})
def _spec(tokens,allow=True,connected=False,max_paths=5):
    return build_router_path_spec("r","153.1",tokens,{"allow_empty_result":allow,"require_connected_sequence":connected,"max_paths":max_paths,"path_ordering":"CANONICAL"})
def test_same_router_inputs_different_order_produce_same_resolved_path_hash():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n3","n1","n2"),("e2","e1"))
    s1=_spec((_tok("a","NODE_ID","n1",0),_tok("b","EDGE_ID","e1",1)))
    s2=_spec((_tok("b","EDGE_ID","e1",1),_tok("a","NODE_ID","n1",0)))
    assert resolve_router_lattice_paths(g,s1,idx).resolved_path_hash==resolve_router_lattice_paths(g,s2,idx).resolved_path_hash
def test_router_resolution_rejects_disconnected_sequence():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n3"),tuple())
    s=_spec((_tok("a","NODE_ID","n1",0),_tok("b","NODE_ID","n3",1)),connected=True)
    with pytest.raises(ValueError,match="INVALID_INPUT"): resolve_router_lattice_paths(g,s,idx)
def test_router_resolution_rejects_ambiguous_direct_edge():
    g=_graph(); e3=_edge("e3","n1","n2","adjacent")
    g2=build_semantic_lattice_graph("l","153.1",_bounds(),g.nodes,(g.edges[0],e3,g.edges[1]))
    idx=build_special_path_index(g2,"i","1",("n1","n2"),("e1","e3"))
    s=_spec((_tok("a","NODE_ID","n1",0),_tok("b","NODE_ID","n2",1)),connected=True)
    with pytest.raises(ValueError,match="INVALID_INPUT"): resolve_router_lattice_paths(g2,s,idx)
def test_router_lattice_receipt_rejects_mismatched_graph_spec_index_or_paths():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2")); s=_spec((_tok("a","NODE_ID","n1",0),))
    rps=resolve_router_lattice_paths(g,s,idx); rec=build_router_lattice_path_receipt(g,s,idx,rps)
    bad=RouterLatticePathReceipt(**{**rec.__dict__,"path_count":9,"receipt_hash":""}); bad=RouterLatticePathReceipt(**{**bad.__dict__,"receipt_hash":bad.stable_hash()})
    with pytest.raises(ValueError,match="INVALID_INPUT"): validate_router_lattice_path_receipt(bad,g,s,idx,rps)
def test_v153_1_does_not_implement_readout_search_mask_or_hilbert_behavior():
    for cls in (RouterPathSpec,SpecialPathIndex,ResolvedLatticePathSet,RouterLatticePathReceipt):
        for n in ("readout","search","mask","hilber","hilbert","project","traverse","shortest_path"):
            assert not hasattr(cls,n)


def test_coordinate_token_without_match_fails():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2"))
    s=_spec((_tok("c","COORDINATE",{"x":4,"y":4,"z":4},0),))
    with pytest.raises(ValueError,match="INVALID_INPUT"): resolve_router_lattice_paths(g,s,idx)

def test_resolved_path_set_canonical_payload_excludes_path_hash():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2")); s=_spec((_tok("a","NODE_ID","n1",0),))
    rps=resolve_router_lattice_paths(g,s,idx)
    assert "path_hash" not in rps.to_canonical_json()

def test_parent_payload_does_not_include_child_hash_fields():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2")); s=_spec((_tok("a","NODE_ID","n1",0),))
    rps=resolve_router_lattice_paths(g,s,idx)
    assert "path_hash" not in rps._canonical_payload()["resolved_paths"][0]

def test_same_tokens_different_order_produce_same_result():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2"))
    t1=_tok("a","NODE_ID","n1",0); t2=_tok("b","EDGE_ID","e1",1); t3=_tok("c","CONSTRAINT_TYPE","soft",2)
    r1=resolve_router_lattice_paths(g,_spec((t1,t2,t3)),idx)
    r2=resolve_router_lattice_paths(g,_spec((t3,t2,t1)),idx)
    assert r1.resolved_path_hash==r2.resolved_path_hash

def test_ambiguous_token_resolution_rejected():
    g=_graph(); e3=_edge("e3","n2","n3","adjacent")
    g2=build_semantic_lattice_graph("l","153.1",_bounds(),g.nodes,(g.edges[0],e3,g.edges[1]))
    idx=build_special_path_index(g2,"i","1",("n1","n2","n3"),("e1","e2","e3"))
    s=_spec((_tok("x","CONSTRAINT_TYPE","adjacent",0),),max_paths=1)
    with pytest.raises(ValueError,match="INVALID_INPUT"): resolve_router_lattice_paths(g2,s,idx)

def test_non_string_node_id_token_value_rejected():
    with pytest.raises(ValueError,match="INVALID_INPUT"): RouteToken("t","NODE_ID",123,0,"")

def test_non_string_edge_id_token_value_rejected():
    with pytest.raises(ValueError,match="INVALID_INPUT"): RouteToken("t","EDGE_ID",{"x":1},0,"")

def test_non_string_constraint_type_token_value_rejected():
    with pytest.raises(ValueError,match="INVALID_INPUT"): RouteToken("t","CONSTRAINT_TYPE",["a","b"],0,"")

def test_unmatched_constraint_type_token_rejected():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2"))
    s=_spec((_tok("x","CONSTRAINT_TYPE","nonexistent",0),))
    with pytest.raises(ValueError,match="INVALID_INPUT"): resolve_router_lattice_paths(g,s,idx)

def test_forged_path_set_hash_rejected_in_receipt_validation():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2")); s=_spec((_tok("a","NODE_ID","n1",0),))
    rps=resolve_router_lattice_paths(g,s,idx); rec=build_router_lattice_path_receipt(g,s,idx,rps)
    # Create a forged path set with empty hash, which passes dataclass validation but should fail receipt validation
    forged_rps=ResolvedLatticePathSet(rps.graph_hash,rps.router_path_spec_hash,rps.special_path_index_hash,rps.resolved_paths,"",rps.empty_result_reason)
    forged_rec=RouterLatticePathReceipt(rec.graph_hash,rec.router_path_spec_hash,rec.special_path_index_hash,"",rec.path_count,"")
    forged_rec=RouterLatticePathReceipt(**{**forged_rec.__dict__,"receipt_hash":forged_rec.stable_hash()})
    # The validation should reject because resolved_path_hash does not equal stable_hash()
    with pytest.raises(ValueError,match="INVALID_INPUT"): validate_router_lattice_path_receipt(forged_rec,g,s,idx,forged_rps)

def test_empty_result_with_none_reason_rejected():
    with pytest.raises(ValueError,match="INVALID_INPUT"): ResolvedLatticePathSet("gh","rph","sph",tuple(),"","NONE")

def test_stale_token_hash_rejected_in_dataclass():
    # RouteToken constructor validates hash when non-empty
    with pytest.raises(ValueError,match="INVALID_INPUT"): RouteToken("t","NODE_ID","n1",0,"stale_hash_not_matching")

def test_build_receipt_validates_path_set_belongs_to_graph():
    g=_graph(); idx=build_special_path_index(g,"i","1",("n1","n2","n3"),("e1","e2")); s=_spec((_tok("a","NODE_ID","n1",0),))
    rps=resolve_router_lattice_paths(g,s,idx)
    forged_rps=ResolvedLatticePathSet("wrong_graph_hash",rps.router_path_spec_hash,rps.special_path_index_hash,rps.resolved_paths,"","NONE")
    forged_rps=ResolvedLatticePathSet(**{**forged_rps.__dict__,"resolved_path_hash":forged_rps.stable_hash()})
    with pytest.raises(ValueError,match="INVALID_INPUT"): build_router_lattice_path_receipt(g,s,idx,forged_rps)
