from __future__ import annotations
import json
import pytest
import qec.analysis.lattice_drift_replay_alignment as m


def test_aliases_removed():
    assert m.ReadoutReplayReceipt is not m.RouterReplayReceipt
    assert m.MaskReplayReceipt is not m.RouterReplayReceipt
    assert m.ShiftReplayReceipt is not m.RouterReplayReceipt
    assert m.KernelReplayReceipt is not m.RouterReplayReceipt
    assert m.ReadoutMatrixReplayReceipt is not m.RouterReplayReceipt
    assert m.build_readout_replay_receipt is not m.build_router_replay_receipt


def test_drift_statuses_and_tamper():
    h1, h2 = "a" * 64, "b" * 64
    match = m.build_lattice_drift_receipt("d1", "ROUTER_PATH", "a", h1, h1, h1)
    assert (match.replay_status, match.drift_reason, match.drift_detected) == ("REPLAY_MATCH", "HASH_MATCH", False)
    drift = m.build_lattice_drift_receipt("d2", "ROUTER_PATH", "a", h1, h2, h1)
    assert drift.replay_status == "REPLAY_DRIFT"
    missing = m.build_lattice_drift_receipt("d3", "ROUTER_PATH", "a", h1, h2, h1, missing=True)
    assert missing.replay_status == "REPLAY_MISSING"
    incon = m.build_lattice_drift_receipt("d4", "ROUTER_PATH", "a", h1, h2, h1, inconsistent=True)
    assert incon.replay_status == "REPLAY_INCONSISTENT"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        m.LatticeDriftReceipt(**{**match.__dict__, "drift_reason": "HASH_MISMATCH"})


def test_scope_guard_and_no_v154_exports():
    for name in ["SubgraphInvariantPattern", "MultiScaleInvariantReceipt", "SierpinskiCompressionReceipt", "DecayCheckpoint", "DigitalDecaySignature", "PerturbationContract", "PerturbationMatrix", "SubstrateContract", "RecursiveProofReceipt"]:
        assert not hasattr(m, name)
    setattr(m.LatticeDriftReceipt, "execute", lambda self: None)
    try:
        with pytest.raises(RuntimeError, match="INVALID_STATE"):
            m._assert_no_v153_9_forbidden_scope()
    finally:
        delattr(m.LatticeDriftReceipt, "execute")


def test_json_safe_dict_roundtrip():
    h1 = "a" * 64
    receipt = m.build_lattice_drift_receipt("d", "READOUT_MATRIX", "z", h1, h1, h1)
    d = receipt.to_dict()
    json.dumps(d, sort_keys=True)
    d["drift_id"] = "x"
    assert receipt.drift_id == "d"
