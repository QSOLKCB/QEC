"""Tests for v82.7.0 — Hybrid Co-Design Engine."""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest

from qec.experiments.hybrid_codesign import (
    run_hybrid_codesign,
    run_hybrid_pair,
    score_hybrid_pair,
    summarize_hybrid_pair,
)


# ---------------------------------------------------------------------------
# Deterministic fake pipelines
# ---------------------------------------------------------------------------

def _make_fake_result(
    *,
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "convergent",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: list | None = None,
) -> dict:
    if strong_invariants is None:
        strong_invariants = ["energy", "centroid"]
    return {
        "features": {
            "energy": 1.0,
            "spread": 0.5,
            "zcr": 0.2,
            "centroid": 3.0,
            "gradient_energy": 0.1,
            "curvature": 0.05,
        },
        "probe": {
            "final_state": "ACCEPT",
            "steps": 5,
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "epsilon": 1e-3,
                "reject_cycle": 0,
                "decision": "ACCEPT",
                "thresholds": None,
                "reason": None,
            }],
        },
        "invariants": {
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "invariants": {
                    "strong_invariants": strong_invariants,
                    "weak_invariants": [],
                    "non_invariants": [],
                },
            }],
            "final_state": "ACCEPT",
        },
        "trajectory": {
            "n_points": 5,
            "mean_velocity": 0.01,
            "max_velocity": 0.05,
            "oscillation_score": 0.0,
            "convergence_rate": 0.001,
            "classification": classification,
        },
        "verification": {"match": True, "final_hash": "abc123", "steps": 5},
        "proof": {"payload": {}, "signature": "sig", "verified": verified},
        "consensus": {
            "n_agents": 3,
            "consensus": consensus,
            "agreement_ratio": 1.0,
            "consensus_hash": "abc123",
        },
    }


def _fake_uff(theta, *, v_circ_fn=None, **kwargs):
    """Deterministic UFF mock keyed on theta[0]."""
    s = float(theta[0]) * 0.1
    return _make_fake_result(stability_score=s, phase="stable_region")


def _fake_pipeline(seq: int) -> dict:
    """Deterministic pipeline keyed on integer index."""
    if seq % 3 == 0:
        return _make_fake_result(
            stability_score=0.05, phase="stable_region",
            classification="convergent",
            strong_invariants=["energy", "centroid", "spread"],
        )
    elif seq % 3 == 1:
        return _make_fake_result(
            stability_score=0.8, phase="near_boundary",
            classification="oscillating",
            strong_invariants=["energy"],
        )
    else:
        return _make_fake_result(
            stability_score=2.5, phase="chaotic_transition",
            classification="divergent",
            strong_invariants=[],
        )


def _fake_pipeline_invalid(seq: int) -> dict:
    """Pipeline that returns invalid (consensus=False) results."""
    return _make_fake_result(consensus=False, verified=False)



@pytest.fixture()
def _authenticated_hybrid(monkeypatch):
    """Verifier-owned test trust: exercise real signatures, not stored flags."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption,
    )
    from qec.controller.execution_proof import (
        create_execution_proof, experiment_result_binding,
    )
    from qec.controller.replay_engine import build_hash_chain

    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = private.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    trusted_signers = {"uff-test": public_pem, "sequence-test": public_pem}

    def authenticated_result(result, signer_id, domain, identity):
        history = result["probe"]["history"]
        result["verification"] = {
            "final_hash": build_hash_chain(history)["final_hash"],
            "steps": len(history),
            "match": True,
            "mismatch_index": None,
        }
        result["proof"] = create_execution_proof(
            result["verification"], signer_id, private_pem, public_pem,
            trusted_signers=trusted_signers,
            metadata={"experiment_binding_v1": experiment_result_binding(
                result, domain=domain, identity=identity,
            )},
        )
        assert result["proof"]["verified"] is True
        return result

    def signed_uff(theta, **kwargs):
        return authenticated_result(_fake_uff(theta), "uff-test", "uff", theta)

    def signed_pipeline(seq):
        return authenticated_result(
            _fake_pipeline(seq), "sequence-test", "sequence", seq,
        )

    monkeypatch.setattr(
        "qec.experiments.uff_bridge.run_uff_experiment", signed_uff
    )
    return trusted_signers, signed_pipeline

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunHybridPair:
    def test_returns_both_results(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )
        result = run_hybrid_pair(
            [1.0, 2.0, 0.5], 0, pipeline_fn=_fake_pipeline,
        )
        assert "theta_result" in result
        assert "seq_result" in result

    def test_requires_pipeline_fn(self):
        with pytest.raises(ValueError, match="pipeline_fn"):
            run_hybrid_pair([1.0, 2.0, 0.5], 0)


class TestSummarizeHybridPair:
    def test_required_keys(self):
        theta = [1.0, 2.0, 0.5]
        theta_result = _make_fake_result(stability_score=0.1)
        seq_result = _make_fake_result(stability_score=0.1)

        summary = summarize_hybrid_pair(theta, theta_result, 0, seq_result)

        assert "theta_summary" in summary
        assert "sequence_summary" in summary
        assert "theta_embedding" in summary
        assert "sequence_embedding" in summary
        assert "distance" in summary

    def test_embedding_length(self):
        theta = [1.0, 2.0, 0.5]
        result = _make_fake_result()
        summary = summarize_hybrid_pair(theta, result, 0, result)

        assert len(summary["theta_embedding"]) == 4
        assert len(summary["sequence_embedding"]) == 4

    def test_distance_non_negative(self):
        theta = [1.0, 2.0, 0.5]
        result = _make_fake_result()
        summary = summarize_hybrid_pair(theta, result, 0, result)
        assert summary["distance"] >= 0.0



def test_signed_proofs_cannot_be_transplanted_to_other_experiment_results(
    _authenticated_hybrid,
):
    """Real Ed25519 receipts do not authorize unrelated scored data."""
    trusted_signers, signed_pipeline = _authenticated_hybrid
    theta = [1.0, 2.0, 0.5]
    raw = run_hybrid_pair(
        theta, 0, pipeline_fn=signed_pipeline, trusted_signers=trusted_signers
    )
    good = summarize_hybrid_pair(
        theta, raw["theta_result"], 0, raw["seq_result"],
        trusted_signers=trusted_signers,
    )
    assert good["theta_summary"]["verified"] is True
    assert good["sequence_summary"]["verified"] is True

    altered_sequence = copy.deepcopy(raw["seq_result"])
    altered_sequence["verification"]["match"] = False
    altered_sequence["verification"]["final_hash"] = "f" * 64
    altered_sequence["probe"]["history"][-1]["stability_score"] = 9.0
    altered_sequence["invariants"]["history"][-1]["invariants"]["strong_invariants"] = []
    spoof = summarize_hybrid_pair(
        theta, raw["theta_result"], 0, altered_sequence,
        trusted_signers=trusted_signers,
    )
    assert spoof["sequence_summary"]["verified"] is False

    # The correct, unchanged signed data also fails under the wrong identity.
    swapped = summarize_hybrid_pair(
        theta, raw["theta_result"], 1, raw["seq_result"],
        trusted_signers=trusted_signers,
    )
    assert swapped["sequence_summary"]["verified"] is False

    altered_uff = copy.deepcopy(raw["theta_result"])
    altered_uff["features"]["energy"] = 98765.0
    uff_spoof = summarize_hybrid_pair(
        theta, altered_uff, 0, raw["seq_result"],
        trusted_signers=trusted_signers,
    )
    assert uff_spoof["theta_summary"]["verified"] is False

    # Legacy replay-only proofs are not signed attestations of scored metrics.
    legacy = copy.deepcopy(raw["seq_result"])
    legacy["proof"]["signed_payload"]["metadata"].pop("experiment_binding_v1")
    assert summarize_hybrid_pair(
        theta, raw["theta_result"], 0, legacy,
        trusted_signers=trusted_signers,
    )["sequence_summary"]["verified"] is False


class TestScoreHybridPair:
    def test_aligned(self):
        scored = score_hybrid_pair({"distance": 0.5})
        assert scored["alignment"] == "aligned"
        assert 0.0 < scored["compatibility"] <= 1.0

    def test_near_aligned(self):
        scored = score_hybrid_pair({"distance": 3.5})
        assert scored["alignment"] == "near_aligned"

    def test_mismatched(self):
        scored = score_hybrid_pair({"distance": 10.0})
        assert scored["alignment"] == "mismatched"

    def test_compatibility_monotonic(self):
        """Lower distance must produce higher compatibility."""
        s_low = score_hybrid_pair({"distance": 1.0})
        s_high = score_hybrid_pair({"distance": 5.0})
        assert s_low["compatibility"] > s_high["compatibility"]

    def test_zero_distance(self):
        scored = score_hybrid_pair({"distance": 0.0})
        assert scored["compatibility"] == 1.0
        assert scored["alignment"] == "aligned"


class TestRunHybridCodesign:
    @pytest.fixture()
    def _patch_uff(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )

    def test_deterministic(self, _patch_uff):
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]]
        seqs = [0, 1, 2]
        r1 = run_hybrid_codesign(grid, seqs, pipeline_fn=_fake_pipeline)
        r2 = run_hybrid_codesign(grid, seqs, pipeline_fn=_fake_pipeline)
        assert r1["n_pairs"] == r2["n_pairs"]
        assert r1["best_pair"] == r2["best_pair"]
        assert r1["worst_pair"] == r2["worst_pair"]
        for p1, p2 in zip(r1["pairs"], r2["pairs"]):
            assert p1["distance"] == p2["distance"]
            assert p1["compatibility"] == p2["compatibility"]

    def test_no_mutation_theta_grid(self, _patch_uff):
        grid = [[1.0, 2.0, 0.5]]
        grid_copy = copy.deepcopy(grid)
        run_hybrid_codesign(grid, [0], pipeline_fn=_fake_pipeline)
        assert grid == grid_copy

    def test_no_mutation_sequences(self, _patch_uff):
        seqs = [0, 1, 2]
        seqs_copy = list(seqs)
        run_hybrid_codesign([[1.0, 2.0, 0.5]], seqs, pipeline_fn=_fake_pipeline)
        assert seqs == seqs_copy

    def test_best_worst_present(self, _authenticated_hybrid):
        trusted_signers, signed_pipeline = _authenticated_hybrid
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0, 1],
            pipeline_fn=signed_pipeline, trusted_signers=trusted_signers,
        )
        assert "theta" in result["best_pair"]
        assert "theta" in result["worst_pair"]
        assert result["best_pair"]["alignment"] != "invalid"

    def test_counts_sum(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
        )
        total = sum(result["alignment_counts"].values())
        assert total == result["n_pairs"]

    def test_pair_entry_keys(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        entry = result["pairs"][0]
        expected_keys = {
            "theta", "sequence", "distance", "compatibility",
            "alignment", "theta_phase", "sequence_phase",
            "theta_class", "sequence_class",
            "phase_agreement", "class_agreement",
        }
        assert expected_keys <= set(entry.keys())

    def test_single_pair(self, _authenticated_hybrid):
        trusted_signers, signed_pipeline = _authenticated_hybrid
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0],
            pipeline_fn=signed_pipeline, trusted_signers=trusted_signers,
        )
        assert result["n_pairs"] == 1
        assert result["best_pair"] == result["worst_pair"]
        assert result["best_pair"]["alignment"] != "invalid"

    def test_untrusted_reported_verified_does_not_score(self, _patch_uff):
        # Legacy mock advertises verified=True but has no authentic signature.
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        assert result["pairs"][0]["alignment"] == "invalid"
        assert result["best_pair"] == {}

    def test_empty_theta_grid(self, _patch_uff):
        result = run_hybrid_codesign([], [0, 1], pipeline_fn=_fake_pipeline)
        assert result["n_pairs"] == 0
        assert result["pairs"] == []

    def test_empty_sequences(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [], pipeline_fn=_fake_pipeline,
        )
        assert result["n_pairs"] == 0
        assert result["pairs"] == []

    def test_invalid_pair_handling(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline_invalid,
        )
        entry = result["pairs"][0]
        assert entry["alignment"] == "invalid"
        assert entry["compatibility"] == 0.0

    def test_invalid_counts(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0, 1], pipeline_fn=_fake_pipeline_invalid,
        )
        assert result["alignment_counts"]["invalid"] == 2

    def test_best_pair_excludes_invalid(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline_invalid,
        )
        # All pairs invalid → best_pair should be empty
        assert result["best_pair"] == {}

    def test_phase_class_agreement_fields(self, _patch_uff):
        result = run_hybrid_codesign(
            [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        entry = result["pairs"][0]
        assert isinstance(entry["phase_agreement"], bool)
        assert isinstance(entry["class_agreement"], bool)

    def test_requires_pipeline_fn(self):
        with pytest.raises(ValueError, match="pipeline_fn"):
            run_hybrid_codesign([[1.0, 2.0, 0.5]], [0])
