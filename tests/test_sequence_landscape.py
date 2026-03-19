"""Tests for v82.4.0 — Sequence Intelligence Engine.

Covers:
- Deterministic outputs
- Correct counts (phase + class)
- Classification consistency
- No mutation of inputs
- Empty / single sequence edge cases
- Unified input_type field
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

from qec.experiments.sequence_landscape import (
    _extract_sequence_summary,
    classify_sequence,
    run_sequence_landscape,
)


# -----------------------------------------------------------------------
# Helpers — deterministic fake pipeline
# -----------------------------------------------------------------------

def _make_fake_result(
    *,
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "convergent",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: list | None = None,
) -> dict:
    """Build a synthetic pipeline result dict for testing."""
    if strong_invariants is None:
        strong_invariants = ["energy", "centroid"]

    return {
        "probe": {
            "final_state": "ACCEPT",
            "steps": 5,
            "history": [
                {
                    "from_state": "INVARIANT",
                    "to_state": "EVALUATE",
                    "stability_score": stability_score,
                    "phase": phase,
                    "epsilon": 1e-3,
                    "reject_cycle": 0,
                    "decision": "ACCEPT",
                    "thresholds": None,
                    "reason": None,
                },
            ],
        },
        "invariants": {
            "history": [
                {
                    "from_state": "INVARIANT",
                    "to_state": "EVALUATE",
                    "stability_score": stability_score,
                    "phase": phase,
                    "invariants": {
                        "strong_invariants": strong_invariants,
                        "weak_invariants": [],
                        "non_invariants": [],
                    },
                },
            ],
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


def _fake_pipeline_stable(seq):
    """Pipeline that always returns stable results."""
    return _make_fake_result(
        stability_score=0.1,
        phase="stable_region",
        classification="convergent",
    )


def _fake_pipeline_chaotic(seq):
    """Pipeline that always returns chaotic results."""
    return _make_fake_result(
        stability_score=3.0,
        phase="chaotic_transition",
        classification="divergent",
        strong_invariants=[],
    )


def _fake_pipeline_mixed(seq):
    """Pipeline that varies by sequence index."""
    idx = seq if isinstance(seq, int) else 0
    if idx % 3 == 0:
        return _make_fake_result(
            stability_score=0.05, phase="stable_region",
            strong_invariants=["energy", "centroid", "spread"],
        )
    elif idx % 3 == 1:
        return _make_fake_result(
            stability_score=0.8, phase="near_boundary",
            strong_invariants=["energy"],
        )
    else:
        return _make_fake_result(
            stability_score=2.5, phase="chaotic_transition",
            strong_invariants=[],
        )


# -----------------------------------------------------------------------
# _extract_sequence_summary tests
# -----------------------------------------------------------------------

class TestExtractSequenceSummary:
    """Tests for _extract_sequence_summary."""

    def test_returns_required_keys(self):
        result = _make_fake_result()
        summary = _extract_sequence_summary("test_seq", result)
        required = {
            "input", "input_type", "stability_score", "phase",
            "invariant_strength", "consensus", "verified",
            "trajectory_class",
        }
        assert required.issubset(summary.keys())

    def test_input_type_is_sequence(self):
        result = _make_fake_result()
        summary = _extract_sequence_summary("seq_0", result)
        assert summary["input_type"] == "sequence"

    def test_stability_score_extracted(self):
        result = _make_fake_result(stability_score=1.23)
        summary = _extract_sequence_summary("s", result)
        assert summary["stability_score"] == 1.23

    def test_phase_extracted(self):
        result = _make_fake_result(phase="near_boundary")
        summary = _extract_sequence_summary("s", result)
        assert summary["phase"] == "near_boundary"

    def test_invariant_strength_counts_strong(self):
        result = _make_fake_result(strong_invariants=["a", "b", "c"])
        summary = _extract_sequence_summary("s", result)
        assert summary["invariant_strength"] == 3

    def test_consensus_and_verified(self):
        result = _make_fake_result(consensus=True, verified=True)
        summary = _extract_sequence_summary("s", result)
        assert summary["consensus"] is True
        assert summary["verified"] is True

    def test_trajectory_class_extracted(self):
        result = _make_fake_result(classification="oscillatory")
        summary = _extract_sequence_summary("s", result)
        assert summary["trajectory_class"] == "oscillatory"

    def test_no_mutation_of_result(self):
        result = _make_fake_result()
        original = copy.deepcopy(result)
        _extract_sequence_summary("s", result)
        assert result == original


# -----------------------------------------------------------------------
# classify_sequence tests
# -----------------------------------------------------------------------

class TestClassifySequence:
    """Tests for classify_sequence."""

    def test_stable(self):
        summary = {
            "stability_score": 0.1,
            "phase": "stable_region",
            "invariant_strength": 3,
        }
        assert classify_sequence(summary) == "stable"

    def test_chaotic_by_score(self):
        summary = {
            "stability_score": 2.5,
            "phase": "stable_region",
            "invariant_strength": 3,
        }
        assert classify_sequence(summary) == "chaotic"

    def test_chaotic_by_phase(self):
        summary = {
            "stability_score": 0.1,
            "phase": "chaotic_transition",
            "invariant_strength": 3,
        }
        assert classify_sequence(summary) == "chaotic"

    def test_boundary_rider_near_boundary(self):
        summary = {
            "stability_score": 0.3,
            "phase": "near_boundary",
            "invariant_strength": 2,
        }
        assert classify_sequence(summary) == "boundary_rider"

    def test_boundary_rider_unstable(self):
        summary = {
            "stability_score": 0.3,
            "phase": "unstable_region",
            "invariant_strength": 2,
        }
        assert classify_sequence(summary) == "boundary_rider"

    def test_fragile(self):
        summary = {
            "stability_score": 0.7,
            "phase": "stable_region",
            "invariant_strength": 1,
        }
        assert classify_sequence(summary) == "fragile"

    def test_deterministic(self):
        summary = {
            "stability_score": 0.6,
            "phase": "stable_region",
            "invariant_strength": 0,
        }
        c1 = classify_sequence(summary)
        c2 = classify_sequence(summary)
        assert c1 == c2


# -----------------------------------------------------------------------
# run_sequence_landscape tests
# -----------------------------------------------------------------------

class TestRunSequenceLandscape:
    """Tests for run_sequence_landscape."""

    def test_deterministic_repeated_run(self):
        seqs = [0, 1, 2]
        r1 = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        r2 = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        assert r1["n_sequences"] == r2["n_sequences"]
        assert r1["phase_counts"] == r2["phase_counts"]
        assert r1["class_counts"] == r2["class_counts"]
        for p1, p2 in zip(r1["points"], r2["points"]):
            assert p1["stability_score"] == p2["stability_score"]
            assert p1["class"] == p2["class"]

    def test_correct_counts(self):
        seqs = [0, 1, 2, 3, 4, 5]
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        # 0,3 → stable; 1,4 → near_boundary; 2,5 → chaotic
        assert result["n_sequences"] == 6
        assert sum(result["phase_counts"].values()) == 6
        assert sum(result["class_counts"].values()) == 6

    def test_all_stable(self):
        seqs = ["a", "b", "c"]
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_stable)
        assert result["class_counts"].get("stable", 0) == 3

    def test_all_chaotic(self):
        seqs = ["x", "y"]
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_chaotic)
        assert result["class_counts"].get("chaotic", 0) == 2

    def test_best_and_worst_sequences(self):
        seqs = [0, 1, 2]
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        assert len(result["best_sequences"]) == 1
        assert len(result["worst_sequences"]) == 1
        # Best should be idx 0 (score 0.05), worst should be idx 2 (score 2.5)
        assert result["best_sequences"][0] == 0
        assert result["worst_sequences"][0] == 2

    def test_every_point_has_required_keys(self):
        required = {
            "input", "input_type", "stability_score", "phase",
            "invariant_strength", "consensus", "verified",
            "trajectory_class", "class",
        }
        result = run_sequence_landscape([0], pipeline_fn=_fake_pipeline_stable)
        for point in result["points"]:
            assert required.issubset(point.keys())

    def test_input_type_always_sequence(self):
        result = run_sequence_landscape(
            [0, 1], pipeline_fn=_fake_pipeline_stable,
        )
        for point in result["points"]:
            assert point["input_type"] == "sequence"

    def test_no_mutation_of_sequences_list(self):
        seqs = [0, 1, 2]
        original = list(seqs)
        run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_stable)
        assert seqs == original

    def test_empty_sequences(self):
        result = run_sequence_landscape([], pipeline_fn=_fake_pipeline_stable)
        assert result["n_sequences"] == 0
        assert result["phase_counts"] == {}
        assert result["class_counts"] == {}
        assert result["best_sequences"] == []
        assert result["worst_sequences"] == []
        assert result["points"] == []

    def test_single_sequence(self):
        result = run_sequence_landscape(
            ["only_one"], pipeline_fn=_fake_pipeline_stable,
        )
        assert result["n_sequences"] == 1
        assert result["best_sequences"] == result["worst_sequences"]

    def test_output_dir_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_sequence_landscape(
                [0, 1], pipeline_fn=_fake_pipeline_stable,
                output_dir=tmpdir,
            )
            out_path = os.path.join(tmpdir, "sequence_landscape.json")
            assert os.path.isfile(out_path)
            with open(out_path) as f:
                loaded = json.load(f)
            assert loaded["n_sequences"] == result["n_sequences"]

    def test_phase_counts_sum_to_n_sequences(self):
        seqs = list(range(9))
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        total = sum(result["phase_counts"].values())
        assert total == result["n_sequences"]

    def test_class_counts_sum_to_n_sequences(self):
        seqs = list(range(9))
        result = run_sequence_landscape(seqs, pipeline_fn=_fake_pipeline_mixed)
        total = sum(result["class_counts"].values())
        assert total == result["n_sequences"]
