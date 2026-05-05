import pytest

from qec.analysis.atomic_semantic_lattice_contract import AtomicLatticeBounds, ConstraintEdgeReceipt, SemanticLatticeNode, build_semantic_lattice_graph
from qec.analysis.canonical_hashing import sha256_hex
import qec.analysis.router_lattice_paths as rlp
import qec.analysis.readout_projection_receipts as rpr
from qec.analysis.subgraph_invariant_pattern import build_subgraph_invariant_pattern
from qec.analysis.scale_preserving_router_readout_receipts import (
    build_router_scale_receipt,
    build_readout_scale_projection_receipt,
    validate_router_scale_receipt,
    validate_readout_scale_projection_receipt,
    assert_router_scale_preserved,
    assert_readout_scale_preserved,
)


def _fx():
    b = AtomicLatticeBounds("b", "154.3", 5, 5, 5, "")
    b = AtomicLatticeBounds(**{**b.__dict__, "bounds_hash": b.stable_hash()})
    n1 = SemanticLatticeNode("n1", "A", (0, 0, 0), "cr", "sp", {"k": {"v": "n1"}}, "")
    n2 = SemanticLatticeNode("n2", "B", (1, 0, 0), "cr", "sp", {"k": {"v": "n2"}}, "")
    n1 = SemanticLatticeNode(**{**n1.__dict__, "node_hash": n1.stable_hash()})
    n2 = SemanticLatticeNode(**{**n2.__dict__, "node_hash": n2.stable_hash()})
    payload = {"payload": {"pv": "e1"}}
    e1 = ConstraintEdgeReceipt("e1", "n1", "n2", "adjacent", payload, sha256_hex(payload), "", "")
    e1 = ConstraintEdgeReceipt(**{**e1.__dict__, "edge_hash": e1._edge_stable_hash()})
    e1 = ConstraintEdgeReceipt(**{**e1.__dict__, "receipt_hash": e1.stable_hash()})
    g = build_semantic_lattice_graph("l", "154.3", b, (n1, n2), (e1,))
    t1 = rlp.RouteToken("t1", "NODE_ID", "n1", 0, "")
    t2 = rlp.RouteToken("t2", "EDGE_ID", "e1", 1, "")
    t1 = rlp.RouteToken(**{**t1.__dict__, "token_hash": t1.stable_hash()})
    t2 = rlp.RouteToken(**{**t2.__dict__, "token_hash": t2.stable_hash()})
    spec = rlp.build_router_path_spec("r", "154.3", (t1, t2), {"allow_empty_result": False, "require_connected_sequence": False, "max_paths": 4, "path_ordering": "CANONICAL"})
    idx = rlp.build_special_path_index(g, "i", "1", ("n1", "n2"), ("e1",))
    rps = rlp.resolve_router_lattice_paths(g, spec, idx)
    router_receipt = rlp.build_router_lattice_path_receipt(g, spec, idx, rps)
    fs = rpr.build_readout_projection_spec("p", ({"field_id": "a", "source_type": "NODE", "path_id": "path-0", "source_id": "n1", "projection_mode": "IDENTITY_HASH", "key_path": ()},))
    readout_receipt = rpr.build_readout_projection_receipt(g, spec, idx, rps, fs)
    pattern = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", e1.constraint_payload_hash)])
    return router_receipt, readout_receipt, pattern


def test_router_scale_receipt_determinism():
    rr, _, p = _fx()
    a = build_router_scale_receipt(rr, p, 1)
    b = build_router_scale_receipt(rr, p, 1)
    assert a.router_scale_receipt_hash == b.router_scale_receipt_hash


def test_readout_scale_projection_determinism():
    _, rd, p = _fx()
    a = build_readout_scale_projection_receipt(rd, p, 2)
    b = build_readout_scale_projection_receipt(rd, p, 2)
    assert a.readout_scale_projection_receipt_hash == b.readout_scale_projection_receipt_hash


def test_router_scale_receipt_invalid_scale():
    rr, _, p = _fx()
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        build_router_scale_receipt(rr, p, 9)


def test_readout_scale_projection_invalid_scale():
    _, rd, p = _fx()
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        build_readout_scale_projection_receipt(rd, p, -1)


def test_bool_scale_index_rejected():
    rr, rd, p = _fx()
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        build_router_scale_receipt(rr, p, True)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        build_readout_scale_projection_receipt(rd, p, False)


def test_router_unknown_receipt_rejected():
    rr, _, p = _fx()
    with pytest.raises(ValueError, match="UNKNOWN_ROUTER_RECEIPT"):
        build_router_scale_receipt("x", p, 0)
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(rr, "receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="UNKNOWN_ROUTER_RECEIPT"):
        build_router_scale_receipt(rr, p, 0)


def test_readout_unknown_receipt_rejected():
    _, rd, p = _fx()
    with pytest.raises(ValueError, match="UNKNOWN_READOUT_RECEIPT"):
        build_readout_scale_projection_receipt({}, p, 0)
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(rd, "receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="UNKNOWN_READOUT_RECEIPT"):
        build_readout_scale_projection_receipt(rd, p, 0)


def test_router_scale_receipt_hash_tamper_detection():
    rr, _, p = _fx()
    r = build_router_scale_receipt(rr, p, 0)
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(r, "router_scale_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_router_scale_receipt(r)


def test_readout_scale_projection_hash_tamper_detection():
    _, rd, p = _fx()
    r = build_readout_scale_projection_receipt(rd, p, 0)
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(r, "readout_scale_projection_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_readout_scale_projection_receipt(r)


def test_scale_preserved_is_deterministic():
    """v154.3 invariant: scale_preserved is always True and deterministic."""
    rr, rd, p = _fx()

    r1 = build_router_scale_receipt(rr, p, 0)
    r2 = build_router_scale_receipt(rr, p, 0)

    assert r1.scale_preserved is True
    assert r2.scale_preserved is True
    assert r1.router_scale_receipt_hash == r2.router_scale_receipt_hash


def test_readout_scale_preserved_is_deterministic():
    """v154.3 invariant: scale_preserved is always True and deterministic for readout."""
    _, rd, p = _fx()

    r = build_readout_scale_projection_receipt(rd, p, 1)

    assert r.scale_preserved is True


def test_scale_preserved_true_passes_assertion():
    """Verify that scale_preserved=True receipts pass assertion helpers."""
    rr, rd, p = _fx()
    router = build_router_scale_receipt(rr, p, 0)
    assert router.scale_preserved is True
    assert assert_router_scale_preserved(router) is True
    
    readout = build_readout_scale_projection_receipt(rd, p, 1)
    assert readout.scale_preserved is True
    assert assert_readout_scale_preserved(readout) is True


def test_no_router_reexecution_or_readout_reprojection(monkeypatch):
    rr, rd, p = _fx()
    monkeypatch.setattr(rlp, "resolve_router_lattice_paths", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    monkeypatch.setattr(rpr, "project_readout_fields", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    build_router_scale_receipt(rr, p, 0)
    build_readout_scale_projection_receipt(rd, p, 0)


def test_no_v154_4_or_v155_scope_leak():
    import qec.analysis.scale_preserving_router_readout_receipts as m

    for name in (
        "GovernanceCompressionEntry", "GovernanceCompressionReceipt", "SemanticCompressionEntry", "SemanticCompressionReceipt",
        "DecayCheckpoint", "DecayCheckpointSet", "DigitalDecaySignature",
    ):
        assert not hasattr(m, name)


def test_source_receipts_not_mutated():
    rr, rd, p = _fx()
    rr_before, rd_before, p_before = dict(rr.__dict__), dict(rd.__dict__), dict(p.__dict__)
    build_router_scale_receipt(rr, p, 2)
    build_readout_scale_projection_receipt(rd, p, 2)
    assert rr.__dict__ == rr_before
    assert rd.__dict__ == rd_before
    assert p.__dict__ == p_before


def test_cross_run_hash_stability():
    rr, rd, p = _fx()
    hashes = []
    for _ in range(3):
        hashes.append((build_router_scale_receipt(rr, p, 1).router_scale_receipt_hash, build_readout_scale_projection_receipt(rd, p, 1).readout_scale_projection_receipt_hash))
    assert len(set(hashes)) == 1


def test_tampered_pattern_hash_rejected():
    rr, rd, p = _fx()
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(p, "pattern_hash", "a" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        build_router_scale_receipt(rr, p, 0)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        build_readout_scale_projection_receipt(rd, p, 0)


def test_malformed_pattern_hash_rejected():
    rr, rd, p = _fx()
    # Intentional tamper-path test: bypasses frozen dataclass guards to verify builder/validator recomputation.
    object.__setattr__(p, "pattern_hash", "not-a-hash")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_router_scale_receipt(rr, p, 0)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_scale_projection_receipt(rd, p, 0)
