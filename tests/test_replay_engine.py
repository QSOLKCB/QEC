"""
v81.0.0 — Tests for the Deterministic Replay & Verification Engine.

Covers:
    - Determinism (identical runs match)
    - Modified history detection
    - Hash chain consistency
    - Serialization stability
    - Float precision consistency
    - Empty / single-step history edge cases
    - Replay API contract
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from qec.controller.replay_engine import (
    build_hash_chain,
    compare_histories,
    replay_fsm,
    serialize_state,
    verify_run,
)
from qec.controller.qec_fsm import QECFSM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_INPUT = {
    "rms_energy": 0.012,
    "spectral_centroid_hz": 480.0,
    "spectral_spread_hz": 210.0,
    "zero_crossing_rate": 0.048,
    "fft_top_peaks": [
        {"frequency_hz": 100.0, "magnitude": 0.5},
        {"frequency_hz": 200.0, "magnitude": 0.3},
    ],
}

_SAMPLE_CONFIG = {
    "stability_threshold": 0.5,
    "boundary_crossing_threshold": 2,
    "max_reject_cycles": 3,
    "epsilon": 1e-3,
    "n_perturbations": 9,
    "drift_threshold": 1e-4,
}


def _run_fsm():
    """Run a fresh FSM and return (result, input_copy, config_copy)."""
    inp = copy.deepcopy(_SAMPLE_INPUT)
    cfg = copy.deepcopy(_SAMPLE_CONFIG)
    fsm = QECFSM(config=cfg)
    result = fsm.run(inp, max_steps=20)
    return result, inp, cfg


# ---------------------------------------------------------------------------
# 1. Determinism — identical runs produce identical histories
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_produce_identical_history(self):
        r1, _, _ = _run_fsm()
        r2, _, _ = _run_fsm()
        assert r1["history"] == r2["history"]
        assert r1["final_state"] == r2["final_state"]
        assert r1["steps"] == r2["steps"]

    def test_replay_matches_original(self):
        r1, inp, cfg = _run_fsm()
        r2 = replay_fsm(inp, cfg, max_steps=20)
        cmp = compare_histories(r1["history"], r2["history"])
        assert cmp["match"] is True
        assert cmp["mismatch_index"] is None


# ---------------------------------------------------------------------------
# 2. Modified history detection
# ---------------------------------------------------------------------------

class TestModifiedHistory:
    def test_tampered_state_detected(self):
        r1, _, _ = _run_fsm()
        h = copy.deepcopy(r1["history"])
        # Tamper with a decision field.
        h[0]["decision"] = "TAMPERED"
        r2, _, _ = _run_fsm()
        cmp = compare_histories(h, r2["history"])
        assert cmp["match"] is False
        assert cmp["mismatch_index"] == 0

    def test_tampered_stability_score_detected(self):
        r1, _, _ = _run_fsm()
        h = copy.deepcopy(r1["history"])
        # Find an entry with a stability score and tamper it.
        for i, entry in enumerate(h):
            if entry.get("stability_score") is not None:
                h[i]["stability_score"] = 999.0
                break
        r2, _, _ = _run_fsm()
        cmp = compare_histories(h, r2["history"])
        assert cmp["match"] is False

    def test_extra_entry_detected(self):
        r1, _, _ = _run_fsm()
        h = copy.deepcopy(r1["history"])
        h.append({"from_state": "FAKE", "to_state": "FAKE"})
        r2, _, _ = _run_fsm()
        cmp = compare_histories(h, r2["history"])
        assert cmp["match"] is False

    def test_missing_entry_detected(self):
        r1, _, _ = _run_fsm()
        h = copy.deepcopy(r1["history"])
        if len(h) > 1:
            h.pop()
        r2, _, _ = _run_fsm()
        cmp = compare_histories(h, r2["history"])
        assert cmp["match"] is False


# ---------------------------------------------------------------------------
# 3. Hash chain consistency
# ---------------------------------------------------------------------------

class TestHashChain:
    def test_identical_histories_same_hash(self):
        r1, _, _ = _run_fsm()
        r2, _, _ = _run_fsm()
        c1 = build_hash_chain(r1["history"])
        c2 = build_hash_chain(r2["history"])
        assert c1["final_hash"] == c2["final_hash"]
        assert c1["step_hashes"] == c2["step_hashes"]

    def test_tampered_history_different_hash(self):
        r1, _, _ = _run_fsm()
        h = copy.deepcopy(r1["history"])
        h[0]["decision"] = "TAMPERED"
        c_orig = build_hash_chain(r1["history"])
        c_tampered = build_hash_chain(h)
        assert c_orig["final_hash"] != c_tampered["final_hash"]

    def test_hash_chain_length_matches_history(self):
        r1, _, _ = _run_fsm()
        chain = build_hash_chain(r1["history"])
        assert len(chain["step_hashes"]) == len(r1["history"])

    def test_hash_chain_is_chained(self):
        """Verify each hash depends on the previous one."""
        r1, _, _ = _run_fsm()
        chain = build_hash_chain(r1["history"])
        prev = hashlib.sha256(b"genesis").hexdigest()
        for i, entry in enumerate(r1["history"]):
            payload = prev + serialize_state(entry)
            expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            assert chain["step_hashes"][i] == expected
            prev = expected


# ---------------------------------------------------------------------------
# 4. Serialization stability
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_serialization_deterministic(self):
        entry = {
            "from_state": "INIT",
            "to_state": "ANALYZE",
            "stability_score": None,
            "epsilon": 0.001,
            "decision": "CONTINUE",
        }
        s1 = serialize_state(entry)
        s2 = serialize_state(entry)
        assert s1 == s2

    def test_serialization_key_order_independent(self):
        e1 = {"b": 2, "a": 1}
        e2 = {"a": 1, "b": 2}
        assert serialize_state(e1) == serialize_state(e2)

    def test_float_precision_stable(self):
        e1 = {"value": 0.1 + 0.2}
        e2 = {"value": 0.30000000000000004}
        assert serialize_state(e1) == serialize_state(e2)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_history(self):
        cmp = compare_histories([], [])
        assert cmp["match"] is True
        assert cmp["mismatch_index"] is None

    def test_empty_history_hash_chain(self):
        chain = build_hash_chain([])
        assert chain["final_hash"] == hashlib.sha256(b"genesis").hexdigest()
        assert chain["step_hashes"] == []

    def test_single_step_history(self):
        entry = {
            "from_state": "INIT",
            "to_state": "ANALYZE",
            "stability_score": None,
            "phase": None,
            "epsilon": 0.001,
            "reject_cycle": 0,
            "decision": "CONTINUE",
        }
        cmp = compare_histories([entry], [copy.deepcopy(entry)])
        assert cmp["match"] is True

    def test_single_step_hash_chain(self):
        entry = {"from_state": "INIT", "to_state": "ANALYZE"}
        chain = build_hash_chain([entry])
        assert len(chain["step_hashes"]) == 1
        assert chain["final_hash"] == chain["step_hashes"][0]


# ---------------------------------------------------------------------------
# 6. Full verify_run API
# ---------------------------------------------------------------------------

class TestVerifyRun:
    def test_verify_run_matches(self):
        result, inp, cfg = _run_fsm()
        vr = verify_run(inp, result["history"], cfg, max_steps=20)
        assert vr["match"] is True
        assert isinstance(vr["final_hash"], str)
        assert len(vr["final_hash"]) == 64  # SHA-256 hex
        assert vr["steps"] == result["steps"]
        assert vr["mismatch_index"] is None

    def test_verify_run_detects_tamper(self):
        result, inp, cfg = _run_fsm()
        h = copy.deepcopy(result["history"])
        h[0]["decision"] = "TAMPERED"
        vr = verify_run(inp, h, cfg, max_steps=20)
        assert vr["match"] is False
        assert vr["mismatch_index"] == 0
