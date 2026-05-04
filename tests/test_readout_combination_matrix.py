import json

import pytest

from qec.analysis.readout_combination_matrix import (
    MarkovBasisReceipt,
    ReadoutCombinationMatrix,
    ReadoutMatrixReceipt,
    ReadoutTransitionReceipt,
    build_markov_basis_receipt,
    build_readout_combination_matrix,
    build_readout_matrix_receipt,
    build_readout_transition_receipt,
    validate_markov_basis_receipt,
    validate_readout_matrix_receipt,
    validate_readout_transition_receipt,
)


def _h(ch: str) -> str:
    return ch * 64


def _matrix():
    return build_readout_combination_matrix(
        "m1", "READOUT_SHELL_SET", "CORE_KERNEL_SET", ("r1", "r2"), (_h("1"), _h("2")), ("c1", "c2"), (_h("a"), _h("b"))
    )


def test_determinism_and_ordering():
    m1 = _matrix()
    m2 = _matrix()
    assert m1.matrix_hash == m2.matrix_hash
    r1 = build_readout_matrix_receipt("mr", m1)
    r2 = build_readout_matrix_receipt("mr", m2)
    assert r1.receipt_hash == r2.receipt_hash
    b1 = build_markov_basis_receipt("mb", m1, r1)
    b2 = build_markov_basis_receipt("mb", m2, r2)
    assert b1.receipt_hash == b2.receipt_hash
    t1 = build_readout_transition_receipt("tr", b1, 0)
    t2 = build_readout_transition_receipt("tr", b2, 0)
    assert t1.receipt_hash == t2.receipt_hash
    assert json.dumps(m1.to_dict(), sort_keys=True) == json.dumps(m2.to_dict(), sort_keys=True)


def test_order_identity_bearing_and_row_major():
    m1 = _matrix()
    m3 = build_readout_combination_matrix("m1", "READOUT_SHELL_SET", "CORE_KERNEL_SET", ("r2", "r1"), (_h("2"), _h("1")), ("c1", "c2"), (_h("a"), _h("b")))
    m4 = build_readout_combination_matrix("m1", "READOUT_SHELL_SET", "CORE_KERNEL_SET", ("r1", "r2"), (_h("1"), _h("2")), ("c2", "c1"), (_h("b"), _h("a")))
    assert m1.matrix_hash != m3.matrix_hash
    assert m1.matrix_hash != m4.matrix_hash
    assert [c["cell_id"] for c in m1.cells] == ["r1::c1", "r1::c2", "r2::c1", "r2::c2"]


def test_invalid_inputs_and_tamper_detection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_combination_matrix("m", "BAD", "CORE_KERNEL_SET", ("r",), (_h("1"),), ("c",), (_h("a"),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_combination_matrix("m", "READOUT_SHELL_SET", "CORE_KERNEL_SET", (), (), ("c",), (_h("a"),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_combination_matrix("m", "READOUT_SHELL_SET", "CORE_KERNEL_SET", ("r",), (_h("1"),), (), ())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_combination_matrix("m", "READOUT_SHELL_SET", "CORE_KERNEL_SET", ("r", "r"), (_h("1"), _h("2")), ("c",), (_h("a"),))

    m = _matrix()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ReadoutCombinationMatrix(**{**m.to_dict(), "cells": tuple({**m.cells[0], "cell_id": "bad"} if i == 0 else m.cells[i] for i in range(len(m.cells)))})

    r = build_readout_matrix_receipt("mr", m)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_readout_matrix_receipt(ReadoutMatrixReceipt(**{**r.to_dict(), "row_order_hash": _h("f")}), m)

    b = build_markov_basis_receipt("mb", m, r)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_markov_basis_receipt(MarkovBasisReceipt(**{**b.to_dict(), "receipt_hash": _h("e")}), m, r)


def test_transition_and_immutability_and_scope():
    m = _matrix()
    r = build_readout_matrix_receipt("mr", m)
    b = build_markov_basis_receipt("mb", m, r)
    tr = build_readout_transition_receipt("tid", b, 0)
    assert tr.transition_valid
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_transition_receipt("tid", b, True)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_transition_receipt("tid", b, -1)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_readout_transition_receipt("tid", b, b.transition_count)
    with pytest.raises(TypeError):
        b.state_identity_hashes[0] = _h("3")
    d = m.to_dict()
    d["ordered_row_ids"][0] = "x"
    assert m.ordered_row_ids[0] == "r1"
    json.dumps(tr.to_dict(), sort_keys=True)
    validate_readout_transition_receipt(tr, b)
    forbidden = {"LatticeDriftReceipt", "RouterReplayReceipt", "ReadoutReplayReceipt", "MaskReplayReceipt", "ShiftReplayReceipt", "LatticeReplayProofReceipt", "DecayCheckpoint", "DigitalDecaySignature"}
    module_exports = set(__import__("qec.analysis.readout_combination_matrix", fromlist=["*"]).__dict__.keys())
    assert forbidden.isdisjoint(module_exports)


def test_v153_8_scope_guard_uses_runtime_attribute_check(monkeypatch):
    import qec.analysis.readout_combination_matrix as m

    monkeypatch.setattr(m.ReadoutTransitionReceipt, "execute", lambda self: None, raising=False)
    with pytest.raises(RuntimeError, match="INVALID_STATE"):
        m._assert_no_v153_8_forbidden_scope()


def test_v153_8_scope_guard_allows_current_classes():
    import qec.analysis.readout_combination_matrix as m

    m._assert_no_v153_8_forbidden_scope()
