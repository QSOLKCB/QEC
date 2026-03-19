"""Tests for v82.5.0 — Inverse Design Engine."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from qec.experiments.inverse_design import (
    _WORST_SCORE,
    evaluate_sequence,
    generate_candidate_sequences,
    run_inverse_design,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Mock pipeline helpers
# ---------------------------------------------------------------------------

def _make_pipeline_result(
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "stable",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: int = 3,
) -> Dict[str, Any]:
    """Build a synthetic pipeline result dict."""
    return {
        "probe": {
            "history": [
                {"stability_score": stability_score, "phase": phase},
            ],
            "final_state": "accept",
        },
        "invariants": {
            "history": [
                {
                    "invariants": {
                        "strong_invariants": ["a"] * strong_invariants,
                        "weak_invariants": [],
                        "non_invariants": [],
                    },
                },
            ],
            "final_state": "accept",
        },
        "trajectory": {"classification": classification},
        "proof": {"verified": verified},
        "consensus": {"consensus": consensus},
    }


def _stable_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(
        stability_score=0.1,
        phase="stable_region",
        classification="stable",
    )


def _chaotic_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(
        stability_score=3.0,
        phase="chaotic_transition",
        classification="chaotic",
        strong_invariants=0,
    )


def _fragile_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(
        stability_score=0.8,
        phase="normal",
        classification="fragile",
        strong_invariants=1,
    )


def _boundary_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(
        stability_score=1.5,
        phase="near_boundary",
        classification="boundary",
        strong_invariants=1,
    )


def _unverified_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(verified=False)


def _no_consensus_pipeline(_seq: Any) -> Dict[str, Any]:
    return _make_pipeline_result(consensus=False)


# ---------------------------------------------------------------------------
# Tests — Candidate Generation
# ---------------------------------------------------------------------------


class TestGenerateCandidateSequences:
    """Tests for generate_candidate_sequences."""

    def test_deterministic_output(self) -> None:
        """Same inputs produce identical candidates."""
        notes = [60, 62, 64, 67]
        lengths = [2, 3]
        a = generate_candidate_sequences(notes, lengths)
        b = generate_candidate_sequences(notes, lengths)
        assert a == b

    def test_bounded_count(self) -> None:
        """Candidate count stays bounded (< 50 for small inputs)."""
        notes = [60, 62, 64, 67]
        lengths = [2, 3, 4]
        candidates = generate_candidate_sequences(notes, lengths)
        assert len(candidates) < 50

    def test_nonempty_for_valid_inputs(self) -> None:
        """At least one candidate for valid inputs."""
        candidates = generate_candidate_sequences([60], [2])
        assert len(candidates) > 0

    def test_empty_notes(self) -> None:
        """Empty notes produces no candidates."""
        assert generate_candidate_sequences([], [2, 3]) == []

    def test_empty_lengths(self) -> None:
        """Empty lengths produces no candidates."""
        assert generate_candidate_sequences([60, 62], []) == []

    def test_event_structure(self) -> None:
        """Each event has note, velocity, time keys."""
        candidates = generate_candidate_sequences([60], [3])
        for seq in candidates:
            for event in seq:
                assert "note" in event
                assert "velocity" in event
                assert "time" in event

    def test_sequence_lengths_match(self) -> None:
        """All sequences have a length from the requested set."""
        lengths = [2, 4]
        candidates = generate_candidate_sequences([60, 62], lengths)
        for seq in candidates:
            assert len(seq) in lengths

    def test_no_duplicates(self) -> None:
        """No duplicate sequences in output."""
        candidates = generate_candidate_sequences([60, 62, 64], [2, 3])
        keys = [
            tuple((e["note"], e["time"]) for e in seq)
            for seq in candidates
        ]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Tests — Evaluation
# ---------------------------------------------------------------------------


class TestEvaluateSequence:
    """Tests for evaluate_sequence."""

    def test_returns_summary_keys(self) -> None:
        """Summary contains expected keys."""
        seq = [{"note": 60, "velocity": 100, "time": 0.0}]
        summary = evaluate_sequence(seq, _stable_pipeline)
        assert "stability_score" in summary
        assert "phase" in summary
        assert "consensus" in summary
        assert "verified" in summary
        assert "sequence_class" in summary


# ---------------------------------------------------------------------------
# Tests — Scoring
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    """Tests for score_candidate."""

    def test_stable_prefers_low_stability(self) -> None:
        """Stable target scores lower for low stability_score."""
        low = evaluate_sequence([], _stable_pipeline)
        high = evaluate_sequence([], _chaotic_pipeline)
        assert score_candidate(low, "stable") < score_candidate(high, "stable")

    def test_chaotic_prefers_high_instability(self) -> None:
        """Chaotic target scores lower for high stability_score."""
        stable = evaluate_sequence([], _stable_pipeline)
        chaotic = evaluate_sequence([], _chaotic_pipeline)
        assert score_candidate(chaotic, "chaotic") < score_candidate(stable, "chaotic")

    def test_invalid_target_raises(self) -> None:
        """Unknown target raises ValueError."""
        summary = evaluate_sequence([], _stable_pipeline)
        with pytest.raises(ValueError, match="Unknown target"):
            score_candidate(summary, "nonexistent")

    def test_unverified_gets_worst_score(self) -> None:
        """Unverified candidate receives worst possible score."""
        summary = evaluate_sequence([], _unverified_pipeline)
        assert score_candidate(summary, "stable") == _WORST_SCORE

    def test_no_consensus_gets_worst_score(self) -> None:
        """No-consensus candidate receives worst possible score."""
        summary = evaluate_sequence([], _no_consensus_pipeline)
        assert score_candidate(summary, "stable") == _WORST_SCORE

    def test_score_is_nonnegative(self) -> None:
        """All scores are non-negative."""
        for target in ("stable", "fragile", "chaotic", "boundary_rider"):
            summary = evaluate_sequence([], _stable_pipeline)
            assert score_candidate(summary, target) >= 0.0


# ---------------------------------------------------------------------------
# Tests — Search Engine
# ---------------------------------------------------------------------------


class TestRunInverseDesign:
    """Tests for run_inverse_design."""

    def test_deterministic_results(self) -> None:
        """Repeated calls return identical results."""
        kwargs = dict(
            target="stable",
            notes=[60, 62],
            lengths=[2],
            pipeline_fn=_stable_pipeline,
            top_k=3,
        )
        a = run_inverse_design(**kwargs)
        b = run_inverse_design(**kwargs)
        assert a == b

    def test_top_k_respected(self) -> None:
        """Best list length is at most top_k."""
        result = run_inverse_design(
            "stable", [60, 62, 64], [2, 3], _stable_pipeline, top_k=2,
        )
        assert len(result["best"]) <= 2

    def test_scores_ascending(self) -> None:
        """all_scores list is sorted ascending."""
        result = run_inverse_design(
            "stable", [60, 62, 64], [2, 3], _stable_pipeline,
        )
        scores = result["all_scores"]
        assert scores == sorted(scores)

    def test_empty_candidate_space(self) -> None:
        """Empty notes/lengths yields zero candidates."""
        result = run_inverse_design(
            "stable", [], [2], _stable_pipeline,
        )
        assert result["n_candidates"] == 0
        assert result["best"] == []

    def test_result_structure(self) -> None:
        """Result contains required top-level keys."""
        result = run_inverse_design(
            "stable", [60], [2], _stable_pipeline,
        )
        assert "target" in result
        assert "n_candidates" in result
        assert "best" in result
        assert "all_scores" in result
        assert result["target"] == "stable"

    def test_best_entry_structure(self) -> None:
        """Each best entry has sequence, score, summary."""
        result = run_inverse_design(
            "stable", [60, 62], [2], _stable_pipeline,
        )
        for entry in result["best"]:
            assert "sequence" in entry
            assert "score" in entry
            assert "summary" in entry

    def test_invalid_target_raises(self) -> None:
        """Invalid target raises ValueError."""
        with pytest.raises(ValueError):
            run_inverse_design("invalid", [60], [2], _stable_pipeline)

    def test_no_mutation_of_inputs(self) -> None:
        """Input lists are not mutated."""
        notes = [60, 62, 64]
        lengths = [2, 3]
        notes_copy = copy.deepcopy(notes)
        lengths_copy = copy.deepcopy(lengths)
        run_inverse_design("stable", notes, lengths, _stable_pipeline)
        assert notes == notes_copy
        assert lengths == lengths_copy
