import json
import pytest

from qec.analysis.atomic_semantic_lattice_contract import AtomicLatticeBounds, ConstraintEdgeReceipt, SemanticLatticeGraph, SemanticLatticeNode, build_semantic_lattice_graph
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.layer_spec_contract import LayerInvariantSet, LayerSpec
from qec.analysis.layered_lattice_projection_receipts import (
    LayeredEdgeBinding,
    LayeredLatticeProjectionReceipt,
    LayeredLatticeProjectionSpec,
    LayeredNodeBinding,
    MAX_LAYERED_EDGE_BINDINGS,
    MAX_LAYERED_NODE_BINDINGS,
    build_layered_lattice_projection_receipt,
    build_layered_lattice_projection_spec,
    build_layered_topology_integrity_receipt,
    validate_layered_lattice_projection_receipt,
)
from qec.analysis.layered_state_receipt import BaseStateReference, build_layered_receipt


def _bounds():
    b = AtomicLatticeBounds("b", "1", 2, 2, 2, "")
    return AtomicLatticeBounds(**{**b.__dict__, "bounds_hash": b.stable_hash()})


def _node(n, c):
    x = SemanticLatticeNode(n, "t", c, "r", "s", {}, "")
    return SemanticLatticeNode(**{**x.__dict__, "node_hash": x.stable_hash()})


def _edge(i, s, t):
    e = ConstraintEdgeReceipt(i, s, t, "k", {}, sha256_hex({}), "", "")
    e = ConstraintEdgeReceipt(**{**e.__dict__, "edge_hash": e._edge_stable_hash()})
    return ConstraintEdgeReceipt(**{**e.__dict__, "receipt_hash": e.stable_hash()})

def _graph(): return build_semantic_lattice_graph("l","153.3",_bounds(),(_node("n1",(0,0,0)),_node("n2",(1,0,0))),(_edge("e1","n1","n2"),))

def _layered():
    lr = build_layered_receipt(BaseStateReference("bh","base",{}), LayerSpec("L","1",LayerInvariantSet(("inv",)),{}, {}, ()), {"x":1})
    return lr

def _spec(lr):
    return build_layered_lattice_projection_spec("p","153.3",({"binding_id":"nb1","layer_id":"L","node_id":"n1","base_hash":lr.base_hash,"layered_hash":lr.layered_hash,"layer_spec_hash":lr.layer_spec_hash,"layer_payload_hash":lr.layer_payload_hash,"binding_role":"BASE_STATE","binding_metadata":{"k":[1,2]},"binding_hash":""},),({"binding_id":"eb1","layer_id":"L","edge_id":"e1","base_hash":lr.base_hash,"layered_hash":lr.layered_hash,"layer_spec_hash":lr.layer_spec_hash,"layer_payload_hash":lr.layer_payload_hash,"binding_role":"LAYER_CONSTRAINT","binding_metadata":{},"binding_hash":""},))


def test_determinism_and_json_stability():
    g=_graph(); lr=_layered(); s1=_spec(lr)
    s2=build_layered_lattice_projection_spec("p","153.3",tuple(reversed(s1.node_bindings)),tuple(reversed(s1.edge_bindings)))
    assert s1.spec_hash==s2.spec_hash
    t1=build_layered_topology_integrity_receipt(g,lr,s1); t2=build_layered_topology_integrity_receipt(g,lr,s2)
    assert t1.receipt_hash==t2.receipt_hash
    r1=build_layered_lattice_projection_receipt(g,lr,s1); r2=build_layered_lattice_projection_receipt(g,lr,s2)
    assert r1.receipt_hash==r2.receipt_hash
    assert json.dumps(r1.to_dict(), sort_keys=True)


def test_failures_and_hash_integrity():
    g=_graph(); lr=_layered(); s=_spec(lr)
    with pytest.raises(ValueError):
        bad=LayeredNodeBinding(**{**s.node_bindings[0].__dict__,"base_hash":"x"})
        build_layered_lattice_projection_spec("p","153.3",(bad,),s.edge_bindings)
    rec=build_layered_lattice_projection_receipt(g,lr,s)
    validate_layered_lattice_projection_receipt(rec,g,lr,s)
    with pytest.raises(ValueError):
        validate_layered_lattice_projection_receipt(LayeredLatticeProjectionReceipt(**{**rec.__dict__,"receipt_hash":"0"*64}),g,lr,s)


def test_graph_binding_and_bounds_and_empty_rejected():
    g=_graph(); lr=_layered()
    with pytest.raises(ValueError):
        build_layered_lattice_projection_spec("p","153.3",tuple(),tuple())
    with pytest.raises(ValueError):
        build_layered_lattice_projection_spec("p","153.3",tuple({"binding_id":f"n{i}","layer_id":"L","node_id":"n1","base_hash":lr.base_hash,"layered_hash":lr.layered_hash,"layer_spec_hash":lr.layer_spec_hash,"layer_payload_hash":lr.layer_payload_hash,"binding_role":"BASE_STATE","binding_metadata":{},"binding_hash":""} for i in range(MAX_LAYERED_NODE_BINDINGS+1)),tuple())
    s=_spec(lr)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SemanticLatticeGraph(g.lattice_id,g.lattice_version,g.bounds_hash,g.node_hashes,g.edge_receipt_hashes,g.nodes,(_edge("dangling","n1","missing"),),g.graph_hash)


def test_scope_guards_exports_and_immutability():
    import qec.analysis.layered_lattice_projection_receipts as m
    for name in ("QAMCompatibilityProfile","QAMSpecReceipt","QAMCompatibilityValidationReceipt","SearchMask64","MaskReductionReceipt","MaskCollisionReceipt","HilberShiftSpec","HilbertShiftSpec","ShiftProjectionReceipt","ReadoutShell","ReadoutShellStack","ReadoutCombinationMatrix","MarkovBasisReceipt","LatticeDriftReceipt"):
        assert not hasattr(m,name)
    for cls in (m.LayeredNodeBinding,m.LayeredEdgeBinding,m.LayeredLatticeProjectionSpec,m.LayeredTopologyIntegrityReceipt,m.LayeredLatticeProjectionReceipt):
        for attr in ("apply","execute","run","traverse","pathfind","resolve","readout","search","mask","hilber","hilbert","markov"):
            assert not hasattr(cls,attr)
    s=_spec(_layered())
    d=s.to_dict(); d["node_bindings"][0]["binding_metadata"]["k"]=[1,2,9]
    assert s.node_bindings[0].binding_metadata["k"]==(1,2)
