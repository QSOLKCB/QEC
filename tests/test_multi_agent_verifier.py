"""
v81.2.0 — Tests for the Deterministic Multi-Agent Verifier.

Covers:
    - Identical runs reach consensus
    - Agreement ratio is 1.0 for deterministic runs
    - Determinism (running twice yields same result)
    - Correct n_agents count propagated
"""

from __future__ import annotations

import copy

from qec.controller.multi_agent_verifier import (
    compute_consensus,
    run_agents,
    verify_multi_agent,
)
from qec.controller.qec_fsm import QECFSM

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
    inp = copy.deepcopy(_SAMPLE_INPUT)
    cfg = copy.deepcopy(_SAMPLE_CONFIG)
    fsm = QECFSM(config=cfg)
    result = fsm.run(inp, max_steps=20)
    return result, inp, cfg


class TestMultiAgentVerifier:
    def test_identical_runs_reach_consensus(self):
        result, inp, cfg = _run_fsm()
        report = verify_multi_agent(inp, result["history"], cfg, n_agents=3)
        assert report["consensus"] is True

    def test_agreement_ratio_is_one(self):
        result, inp, cfg = _run_fsm()
        report = verify_multi_agent(inp, result["history"], cfg, n_agents=4)
        assert report["agreement_ratio"] == 1.0

    def test_determinism_across_invocations(self):
        result, inp, cfg = _run_fsm()
        r1 = verify_multi_agent(inp, result["history"], cfg, n_agents=3)
        r2 = verify_multi_agent(inp, result["history"], cfg, n_agents=3)
        assert r1["consensus_hash"] == r2["consensus_hash"]
        assert r1["consensus"] == r2["consensus"]

    def test_n_agents_propagated(self):
        result, inp, cfg = _run_fsm()
        report = verify_multi_agent(inp, result["history"], cfg, n_agents=5)
        assert report["n_agents"] == 5
        assert len(report["results"]) == 5

    def test_compute_consensus_unanimous(self):
        fake_results = [
            {"final_hash": "abc123"},
            {"final_hash": "abc123"},
            {"final_hash": "abc123"},
        ]
        c = compute_consensus(fake_results)
        assert c["consensus"] is True
        assert c["agreement_ratio"] == 1.0
        assert c["consensus_hash"] == "abc123"
