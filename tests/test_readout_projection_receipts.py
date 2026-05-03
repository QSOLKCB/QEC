import pytest
import json
import qec.analysis.readout_projection_receipts as readout_projection_receipts

from qec.analysis.atomic_semantic_lattice_contract import AtomicLatticeBounds, ConstraintEdgeReceipt, SemanticLatticeNode, build_semantic_lattice_graph
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.readout_projection_receipts import (
    MAX_READOUT_FIELDS,
    ReadoutFieldSpec,
    ReadoutProjectionReceipt,
    ReadoutProjectionSet,
    build_readout_projection_receipt,
    build_readout_projection_spec,
    project_readout_fields,
    validate_readout_projection_receipt,
)
from qec.analysis.router_lattice_paths import build_router_path_spec, build_special_path_index, resolve_router_lattice_paths, RouteToken


def _bounds():
    b = AtomicLatticeBounds("b", "153.2", 5, 5, 5, "")
    return AtomicLatticeBounds(**{**b.__dict__, "bounds_hash": b.stable_hash()})

def _node(i, c, m=None):
    n = SemanticLatticeNode(i, "semantic", c, "cr", "sp", m or {"k": {"v": i}}, "")
    return SemanticLatticeNode(**{**n.__dict__, "node_hash": n.stable_hash()})

def _edge(e, s, t):
    p = {"payload": {"pv": e}}
    x = ConstraintEdgeReceipt(e, s, t, "adjacent", p, sha256_hex(p), "", "")
    x = ConstraintEdgeReceipt(**{**x.__dict__, "edge_hash": x._edge_stable_hash()})
    return ConstraintEdgeReceipt(**{**x.__dict__, "receipt_hash": x.stable_hash()})

def _fixtures():
    g = build_semantic_lattice_graph("l", "153.2", _bounds(), (_node("n1", (0,0,0)), _node("n2", (1,0,0))), (_edge("e1","n1","n2"),))
    tok1 = RouteToken("t1","NODE_ID","n1",0,"")
    tok2 = RouteToken("t2","EDGE_ID","e1",1,"")
    tok1 = RouteToken(**{**tok1.__dict__, "token_hash": tok1.stable_hash()})
    tok2 = RouteToken(**{**tok2.__dict__, "token_hash": tok2.stable_hash()})
    spec = build_router_path_spec("r","153.2",(tok1,tok2),{"allow_empty_result":False,"require_connected_sequence":False,"max_paths":4,"path_ordering":"CANONICAL"})
    idx = build_special_path_index(g,"i","1",("n1","n2"),("e1",))
    rps = resolve_router_lattice_paths(g,spec,idx)
    return g, spec, idx, rps


def test_v153_2_determinism_and_order_permutation():
    g, ps, idx, rps = _fixtures()
    f1 = ReadoutFieldSpec("a","NODE","path-0","n1","IDENTITY_HASH",tuple())
    f2 = ReadoutFieldSpec("b","EDGE","path-0","e1","CONSTRAINT_PAYLOAD_VALUE",("payload","pv"))
    s1 = build_readout_projection_spec("p", (f1, f2))
    s2 = build_readout_projection_spec("p", (f2, f1))
    assert s1.readout_projection_spec_hash == s2.readout_projection_spec_hash
    r1 = build_readout_projection_receipt(g, ps, idx, rps, s1)
    r2 = build_readout_projection_receipt(g, ps, idx, rps, s2)
    assert r1.receipt_hash == r2.receipt_hash


def test_modes_and_source_binding_and_failures():
    g, ps, idx, rps = _fixtures()
    spec = build_readout_projection_spec("p", (
        {"field_id":"nidh","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"IDENTITY_HASH","key_path":()},
        {"field_id":"nmv","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"METADATA_VALUE","key_path":("k","v")},
        {"field_id":"nco","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"COORDINATE","key_path":()},
        {"field_id":"eidh","source_type":"EDGE","path_id":"path-0","source_id":"e1","projection_mode":"IDENTITY_HASH","key_path":()},
        {"field_id":"ecv","source_type":"EDGE","path_id":"path-0","source_id":"e1","projection_mode":"CONSTRAINT_PAYLOAD_VALUE","key_path":("payload","pv")},
        {"field_id":"pidh","source_type":"PATH","path_id":"path-0","source_id":"path-0","projection_mode":"PATH_IDENTITY","key_path":()},
    ))
    pset = project_readout_fields(g, rps, spec, ps, idx)
    assert isinstance(pset, ReadoutProjectionSet)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ReadoutFieldSpec("x","NODE","path-0","n1","METADATA_VALUE",tuple())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ReadoutFieldSpec("x","EDGE","path-0","e1","COORDINATE",tuple())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ReadoutFieldSpec("x","PATH","path-0","path-0","CONSTRAINT_PAYLOAD_VALUE",("a",))


def test_receipt_integrity_and_tamper_rejection_and_self_hash_exclusion():
    g, ps, idx, rps = _fixtures()
    spec = build_readout_projection_spec("p", ({"field_id":"a","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"IDENTITY_HASH","key_path":()},))
    rec = build_readout_projection_receipt(g, ps, idx, rps, spec)
    validate_readout_projection_receipt(rec, g, ps, idx, rps, spec)
    assert "receipt_hash" not in ReadoutProjectionReceipt(**{**rec.__dict__, "receipt_hash": ""})._canonical_payload()
    pset = project_readout_fields(g, rps, spec, ps, idx)
    assert "projection_hash" not in ReadoutProjectionSet(**{**pset.__dict__, "projection_hash": ""})._canonical_payload()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_readout_projection_receipt(ReadoutProjectionReceipt(**{**rec.__dict__, "receipt_hash": "0" * 64}), g, ps, idx, rps, spec)


def test_bounds_duplicates_non_json_safe_and_max_fields():
    g, ps, idx, rps = _fixtures()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_projection_spec("p", ({"field_id":"a","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"IDENTITY_HASH","key_path":()}, {"field_id":"a","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"IDENTITY_HASH","key_path":()}))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_projection_spec("p", tuple({"field_id":str(i),"source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"IDENTITY_HASH","key_path":()} for i in range(MAX_READOUT_FIELDS + 1)))
    bad = build_readout_projection_spec("p", ({"field_id":"a","source_type":"NODE","path_id":"path-0","source_id":"n1","projection_mode":"METADATA_VALUE","key_path":("missing",)},))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        project_readout_fields(g, rps, bad, ps, idx)


def test_projected_value_deep_immutability_and_to_dict_json_safety():
    field = readout_projection_receipts.ProjectedReadoutField(
        "f", "PATH", "path-0", "path-0", "PATH_IDENTITY",
        {"outer": {"inner": [1, {"x": 2}]}},
        sha256_hex({"outer": {"inner": [1, {"x": 2}]}}),
        "",
    )
    field = readout_projection_receipts.ProjectedReadoutField(**{**field.__dict__, "projected_field_hash": field.stable_hash()})
    with pytest.raises(TypeError):
        field.projected_value["outer"] = {}
    with pytest.raises(TypeError):
        field.projected_value["outer"]["inner"][1]["x"] = 9
    d = field.to_dict()
    d["projected_value"]["outer"]["inner"][1]["x"] = 9
    assert field.to_dict()["projected_value"]["outer"]["inner"][1]["x"] == 2
    json.dumps(field.to_dict(), sort_keys=True)


def test_v153_2_does_not_implement_readout_execution_or_search_mask():
    forbidden = (
        "apply", "execute", "run", "readout", "project", "traverse", "search", "mask", "hilber", "hilbert",
    )
    for cls in (
        readout_projection_receipts.ReadoutFieldSpec,
        readout_projection_receipts.ReadoutProjectionSpec,
        readout_projection_receipts.ProjectedReadoutField,
        readout_projection_receipts.ReadoutProjectionSet,
        readout_projection_receipts.ReadoutProjectionReceipt,
    ):
        for name in forbidden:
            assert not hasattr(cls, name)
    future_scope_names = (
        "SearchMask64", "MaskReductionReceipt", "MaskCollisionReceipt", "HilberShiftSpec", "HilbertShiftSpec",
        "ShiftProjectionReceipt", "ReadoutShell", "ReadoutShellStack", "ReadoutCombinationMatrix", "MarkovBasisReceipt",
    )
    for name in future_scope_names:
        assert not hasattr(readout_projection_receipts, name)
        assert name not in readout_projection_receipts.__all__
