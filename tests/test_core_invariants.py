"""Tests for core invariant promotion and overlay evaluation (v96.0.0).

Covers:
  - Promotion rule correctness
  - Class support counting
  - Ranking stability
  - Overlay application deterministic
  - Full pipeline integration
  - Print layer stability
"""

import numpy as np
import pytest

from qec.analysis.core_invariants import (
    _best_from_hierarchical,
    _best_from_records,
    _empty_metrics,
    _hierarchical_to_records,
    _pick_winner,
    _rank_core_invariants,
    apply_core_invariant_overlay,
    evaluate_core_overlay_with_hierarchy,
    identify_core_invariants,
    print_core_invariant_report,
)
from qec.experiments.hierarchical_correction import (
    print_hierarchical_report,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_invariant_report():
    """Invariant report with cross-class effectiveness data."""
    return {
        "global_ranking": [
            {
                "rank": 1,
                "invariant_type": "explicit_allowed_state_constraint",
                "avg_score": 2.0,
                "count": 4,
                "improved_count": 3,
                "total_score": 8,
            },
            {
                "rank": 2,
                "invariant_type": "geometry_alignment_constraint",
                "avg_score": 1.5,
                "count": 3,
                "improved_count": 2,
                "total_score": 4,
            },
            {
                "rank": 3,
                "invariant_type": "local_stability_constraint",
                "avg_score": 0.5,
                "count": 2,
                "improved_count": 1,
                "total_score": 1,
            },
        ],
        "class_effectiveness": {
            "cycle_like": {
                "explicit_allowed_state_constraint": {
                    "count": 2,
                    "improved_count": 2,
                    "improved_ratio": 1.0,
                    "avg_score": 2.0,
                    "total_score": 4,
                },
                "geometry_alignment_constraint": {
                    "count": 1,
                    "improved_count": 1,
                    "improved_ratio": 1.0,
                    "avg_score": 1.5,
                    "total_score": 2,
                },
                "local_stability_constraint": {
                    "count": 1,
                    "improved_count": 1,
                    "improved_ratio": 1.0,
                    "avg_score": 0.5,
                    "total_score": 1,
                },
            },
            "basin_like": {
                "explicit_allowed_state_constraint": {
                    "count": 2,
                    "improved_count": 1,
                    "improved_ratio": 0.5,
                    "avg_score": 1.0,
                    "total_score": 2,
                },
                "local_stability_constraint": {
                    "count": 1,
                    "improved_count": 0,
                    "improved_ratio": 0.0,
                    "avg_score": -0.5,
                    "total_score": -1,
                },
            },
            "chain_like": {
                "geometry_alignment_constraint": {
                    "count": 2,
                    "improved_count": 1,
                    "improved_ratio": 0.5,
                    "avg_score": 0.5,
                    "total_score": 1,
                },
            },
        },
        "class_ranking": {},
        "interactions": [],
    }


@pytest.fixture
def sample_system_records():
    """Sample system records for overlay testing."""
    return [
        {
            "mode": "none",
            "stability_efficiency": 0.0,
            "compression_efficiency": 0.0,
            "stability_gain": 0,
        },
        {
            "mode": "square",
            "stability_efficiency": 0.2,
            "compression_efficiency": 0.3,
            "stability_gain": 1,
        },
        {
            "mode": "d4",
            "stability_efficiency": 0.4,
            "compression_efficiency": 0.5,
            "stability_gain": 2,
        },
        {
            "mode": "d4+inv",
            "stability_efficiency": 0.5,
            "compression_efficiency": 0.4,
            "stability_gain": 2,
        },
    ]


@pytest.fixture
def sample_hierarchical_results():
    """Sample hierarchical correction results."""
    return [
        {
            "mode": "square",
            "stages": ["square"],
            "projection_distances": [0.5],
            "total_projection_distance": 0.5,
            "stability_efficiency": 0.2,
            "compression_efficiency": 0.3,
            "stability_gain": 1,
            "metrics": {},
        },
        {
            "mode": "d4>e8_like",
            "stages": ["d4", "e8_like"],
            "projection_distances": [0.3, 0.2],
            "total_projection_distance": 0.5,
            "stability_efficiency": 0.6,
            "compression_efficiency": 0.5,
            "stability_gain": 3,
            "metrics": {},
        },
        {
            "mode": "square>d4>e8_like",
            "stages": ["square", "d4", "e8_like"],
            "projection_distances": [0.4, 0.2, 0.1],
            "total_projection_distance": 0.7,
            "stability_efficiency": 0.5,
            "compression_efficiency": 0.6,
            "stability_gain": 2,
            "metrics": {},
        },
    ]


# ---------------------------------------------------------------------------
# PART 1 — CORE INVARIANT IDENTIFICATION TESTS
# ---------------------------------------------------------------------------


class TestIdentifyCoreInvariants:
    """Tests for identify_core_invariants."""

    def test_basic_promotion(self, sample_invariant_report):
        """Invariants appearing in >=2 classes with good metrics are promoted."""
        result = identify_core_invariants(sample_invariant_report)
        core = result["core_invariants"]
        assert len(core) > 0
        # explicit_allowed_state_constraint: cycle_like + basin_like = 2 classes
        types = [c["invariant_type"] for c in core]
        assert "explicit_allowed_state_constraint" in types

    def test_geometry_promoted(self, sample_invariant_report):
        """geometry_alignment_constraint in cycle_like + chain_like."""
        result = identify_core_invariants(sample_invariant_report)
        types = [c["invariant_type"] for c in result["core_invariants"]]
        assert "geometry_alignment_constraint" in types

    def test_local_stability_not_promoted(self, sample_invariant_report):
        """local_stability_constraint: basin_like has avg_score<0, so not promoted."""
        result = identify_core_invariants(sample_invariant_report)
        types = [c["invariant_type"] for c in result["core_invariants"]]
        assert "local_stability_constraint" not in types

    def test_promotion_requires_two_classes(self):
        """Invariant in only 1 class should not be promoted."""
        report = {
            "global_ranking": [
                {"rank": 1, "invariant_type": "only_one_class", "avg_score": 3.0,
                 "count": 1, "improved_count": 1, "total_score": 3},
            ],
            "class_effectiveness": {
                "single_class": {
                    "only_one_class": {
                        "count": 1, "improved_count": 1,
                        "improved_ratio": 1.0, "avg_score": 3.0,
                        "total_score": 3,
                    },
                },
            },
            "class_ranking": {},
            "interactions": [],
        }
        result = identify_core_invariants(report)
        assert len(result["core_invariants"]) == 0

    def test_promotion_requires_improved_ratio(self):
        """Invariant with improved_ratio < 0.5 should not be promoted."""
        report = {
            "global_ranking": [
                {"rank": 1, "invariant_type": "low_ratio", "avg_score": 1.0,
                 "count": 4, "improved_count": 1, "total_score": 4},
            ],
            "class_effectiveness": {
                "class_a": {
                    "low_ratio": {
                        "count": 2, "improved_count": 0,
                        "improved_ratio": 0.0, "avg_score": 1.0,
                        "total_score": 2,
                    },
                },
                "class_b": {
                    "low_ratio": {
                        "count": 2, "improved_count": 0,
                        "improved_ratio": 0.0, "avg_score": 1.0,
                        "total_score": 2,
                    },
                },
            },
            "class_ranking": {},
            "interactions": [],
        }
        result = identify_core_invariants(report)
        assert len(result["core_invariants"]) == 0

    def test_promotion_requires_positive_avg_score(self):
        """Invariant with avg_score <= 0 should not be promoted."""
        report = {
            "global_ranking": [
                {"rank": 1, "invariant_type": "neg_score", "avg_score": -0.5,
                 "count": 4, "improved_count": 3, "total_score": -2},
            ],
            "class_effectiveness": {
                "class_a": {
                    "neg_score": {
                        "count": 2, "improved_count": 2,
                        "improved_ratio": 1.0, "avg_score": -0.5,
                        "total_score": -1,
                    },
                },
                "class_b": {
                    "neg_score": {
                        "count": 2, "improved_count": 1,
                        "improved_ratio": 0.5, "avg_score": 0.0,
                        "total_score": 0,
                    },
                },
            },
            "class_ranking": {},
            "interactions": [],
        }
        result = identify_core_invariants(report)
        assert len(result["core_invariants"]) == 0

    def test_deterministic(self, sample_invariant_report):
        """Result is deterministic."""
        a = identify_core_invariants(sample_invariant_report)
        b = identify_core_invariants(sample_invariant_report)
        assert a == b

    def test_empty_report(self):
        """Empty report produces empty core list."""
        result = identify_core_invariants({
            "global_ranking": [],
            "class_effectiveness": {},
            "class_ranking": {},
            "interactions": [],
        })
        assert result["core_invariants"] == []

    def test_evidence_structure(self, sample_invariant_report):
        """Each core invariant has proper evidence structure."""
        result = identify_core_invariants(sample_invariant_report)
        for core in result["core_invariants"]:
            assert "invariant_type" in core
            assert "classes" in core
            assert "global_rank" in core
            assert "evidence" in core
            ev = core["evidence"]
            assert "per_class" in ev
            assert "num_classes" in ev
            assert "avg_class_score" in ev
            assert ev["num_classes"] >= 2

    def test_classes_sorted(self, sample_invariant_report):
        """Classes list in each core invariant is sorted."""
        result = identify_core_invariants(sample_invariant_report)
        for core in result["core_invariants"]:
            assert core["classes"] == sorted(core["classes"])


# ---------------------------------------------------------------------------
# PART 2 — RANKING TESTS
# ---------------------------------------------------------------------------


class TestRankCoreInvariants:
    """Tests for _rank_core_invariants."""

    def test_more_classes_wins(self):
        """Invariant with more classes ranks higher."""
        core_list = [
            {
                "invariant_type": "few",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": 2.0},
            },
            {
                "invariant_type": "many",
                "classes": ["a", "b", "c"],
                "global_rank": 0,
                "evidence": {"num_classes": 3, "avg_class_score": 1.0},
            },
        ]
        ranked = _rank_core_invariants(core_list)
        assert ranked[0]["invariant_type"] == "many"
        assert ranked[0]["global_rank"] == 1

    def test_higher_score_breaks_tie(self):
        """Same num_classes: higher avg_class_score wins."""
        core_list = [
            {
                "invariant_type": "low",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": 0.5},
            },
            {
                "invariant_type": "high",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": 2.0},
            },
        ]
        ranked = _rank_core_invariants(core_list)
        assert ranked[0]["invariant_type"] == "high"

    def test_name_tiebreak(self):
        """Same classes and score: alphabetical tiebreak."""
        core_list = [
            {
                "invariant_type": "z_inv",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": 1.0},
            },
            {
                "invariant_type": "a_inv",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": 1.0},
            },
        ]
        ranked = _rank_core_invariants(core_list)
        assert ranked[0]["invariant_type"] == "a_inv"

    def test_sequential_ranks(self):
        """Ranks are 1-indexed and sequential."""
        core_list = [
            {
                "invariant_type": f"inv_{i}",
                "classes": ["a", "b"],
                "global_rank": 0,
                "evidence": {"num_classes": 2, "avg_class_score": float(i)},
            }
            for i in range(5)
        ]
        ranked = _rank_core_invariants(core_list)
        ranks = [r["global_rank"] for r in ranked]
        assert ranks == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# PART 3 — CORE INVARIANT OVERLAY TESTS
# ---------------------------------------------------------------------------


class TestApplyCoreInvariantOverlay:
    """Tests for apply_core_invariant_overlay."""

    def test_no_core_invariants(self, sample_system_records):
        """No core invariants returns copy of input."""
        result = apply_core_invariant_overlay(
            sample_system_records,
            {"core_invariants": []},
        )
        assert len(result) == len(sample_system_records)

    def test_with_core_invariants(self, sample_system_records):
        """Core invariant overlay changes record ordering."""
        core = {
            "core_invariants": [
                {
                    "invariant_type": "explicit_allowed_state_constraint",
                    "classes": ["cycle_like", "basin_like"],
                    "global_rank": 1,
                    "evidence": {"num_classes": 2, "avg_class_score": 1.5},
                },
            ],
        }
        result = apply_core_invariant_overlay(sample_system_records, core)
        # Should prioritize d4+inv mode.
        assert result[0]["mode"] == "d4+inv"

    def test_no_mutation(self, sample_system_records):
        """Input records are not mutated."""
        import copy
        original = copy.deepcopy(sample_system_records)
        core = {
            "core_invariants": [
                {
                    "invariant_type": "explicit_allowed_state_constraint",
                    "classes": ["a", "b"],
                    "global_rank": 1,
                    "evidence": {"num_classes": 2, "avg_class_score": 1.0},
                },
            ],
        }
        apply_core_invariant_overlay(sample_system_records, core)
        assert sample_system_records == original

    def test_empty_records(self):
        """Empty records returns empty list."""
        result = apply_core_invariant_overlay(
            [],
            {"core_invariants": [{"invariant_type": "x", "classes": ["a", "b"],
                                   "global_rank": 1, "evidence": {}}]},
        )
        assert result == []

    def test_deterministic(self, sample_system_records):
        """Overlay is deterministic."""
        core = {
            "core_invariants": [
                {
                    "invariant_type": "explicit_allowed_state_constraint",
                    "classes": ["a", "b"],
                    "global_rank": 1,
                    "evidence": {"num_classes": 2, "avg_class_score": 1.0},
                },
            ],
        }
        a = apply_core_invariant_overlay(sample_system_records, core)
        b = apply_core_invariant_overlay(sample_system_records, core)
        assert a == b

    def test_unknown_invariant_type_ignored(self, sample_system_records):
        """Unknown invariant types are silently skipped."""
        core = {
            "core_invariants": [
                {
                    "invariant_type": "nonexistent_constraint",
                    "classes": ["a", "b"],
                    "global_rank": 1,
                    "evidence": {"num_classes": 2, "avg_class_score": 1.0},
                },
            ],
        }
        result = apply_core_invariant_overlay(sample_system_records, core)
        assert len(result) == len(sample_system_records)


# ---------------------------------------------------------------------------
# PART 4 — EVALUATION TESTS
# ---------------------------------------------------------------------------


class TestEvaluateCoreOverlayWithHierarchy:
    """Tests for evaluate_core_overlay_with_hierarchy."""

    def test_basic_evaluation(
        self, sample_system_records, sample_hierarchical_results
    ):
        """Basic evaluation completes without error."""
        core = {"core_invariants": []}
        result = evaluate_core_overlay_with_hierarchy(
            sample_system_records, core, sample_hierarchical_results
        )
        assert "baseline_mode" in result
        assert "hierarchical_mode" in result
        assert "core_overlay_mode" in result
        assert "best_variant" in result
        assert "before_metrics" in result
        assert "after_metrics" in result

    def test_hierarchical_wins(
        self, sample_system_records, sample_hierarchical_results
    ):
        """Hierarchical mode with better metrics wins."""
        core = {"core_invariants": []}
        result = evaluate_core_overlay_with_hierarchy(
            sample_system_records, core, sample_hierarchical_results
        )
        # d4>e8_like has best stability (0.6) in hierarchical results.
        assert result["hierarchical_mode"] == "d4>e8_like"

    def test_deterministic(
        self, sample_system_records, sample_hierarchical_results
    ):
        """Evaluation is deterministic."""
        core = {"core_invariants": []}
        a = evaluate_core_overlay_with_hierarchy(
            sample_system_records, core, sample_hierarchical_results
        )
        b = evaluate_core_overlay_with_hierarchy(
            sample_system_records, core, sample_hierarchical_results
        )
        assert a == b

    def test_empty_hierarchical(self, sample_system_records):
        """Empty hierarchical results still works."""
        core = {"core_invariants": []}
        result = evaluate_core_overlay_with_hierarchy(
            sample_system_records, core, []
        )
        assert result["hierarchical_mode"] == "none"
        assert result["best_variant"] in ("baseline", "hierarchical", "core_overlay")


# ---------------------------------------------------------------------------
# PART 5 — HELPER FUNCTION TESTS
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_best_from_records_empty(self):
        """Empty records returns 'none' mode."""
        mode, metrics = _best_from_records([])
        assert mode == "none"
        assert metrics["stability_efficiency"] == 0.0

    def test_best_from_records_picks_best(self, sample_system_records):
        """Picks record with highest stability_efficiency."""
        mode, metrics = _best_from_records(sample_system_records)
        assert mode == "d4+inv"  # has stability_efficiency=0.5

    def test_best_from_hierarchical_empty(self):
        """Empty hierarchical returns 'none' mode."""
        mode, metrics = _best_from_hierarchical([])
        assert mode == "none"

    def test_hierarchical_to_records(self, sample_hierarchical_results):
        """Converts hierarchical results to record format."""
        records = _hierarchical_to_records(sample_hierarchical_results)
        assert len(records) == 3
        for r in records:
            assert "mode" in r
            assert "stability_efficiency" in r
            assert "compression_efficiency" in r
            assert "stability_gain" in r

    def test_empty_metrics(self):
        """Empty metrics has all zeros."""
        m = _empty_metrics()
        assert m["stability_efficiency"] == 0.0
        assert m["compression_efficiency"] == 0.0

    def test_pick_winner(self):
        """Winner is determined by stability > compression > gain."""
        candidates = [
            ("a", "mode_a", {"stability_efficiency": 0.5, "compression_efficiency": 0.3, "stability_gain": 1}),
            ("b", "mode_b", {"stability_efficiency": 0.8, "compression_efficiency": 0.2, "stability_gain": 0}),
            ("c", "mode_c", {"stability_efficiency": 0.3, "compression_efficiency": 0.9, "stability_gain": 5}),
        ]
        assert _pick_winner(candidates) == "b"


# ---------------------------------------------------------------------------
# PART 6 — PRINT LAYER TESTS
# ---------------------------------------------------------------------------


class TestPrintCoreInvariantReport:
    """Tests for print_core_invariant_report."""

    def test_empty_report(self):
        """Empty core report prints header."""
        text = print_core_invariant_report({"core_invariants": []})
        assert "=== Core Invariants ===" in text
        assert "No core invariants identified." in text

    def test_with_core_invariants(self):
        """Report with core invariants includes all fields."""
        report = {
            "core_invariants": [
                {
                    "invariant_type": "explicit_allowed_state_constraint",
                    "classes": ["basin_like", "cycle_like"],
                    "global_rank": 1,
                    "evidence": {},
                },
                {
                    "invariant_type": "geometry_alignment_constraint",
                    "classes": ["chain_like", "cycle_like"],
                    "global_rank": 2,
                    "evidence": {},
                },
            ],
        }
        text = print_core_invariant_report(report)
        assert "1. explicit_allowed_state_constraint" in text
        assert "basin_like, cycle_like" in text
        assert "2. geometry_alignment_constraint" in text
        assert "chain_like, cycle_like" in text

    def test_deterministic_output(self):
        """Print output is deterministic."""
        report = {
            "core_invariants": [
                {
                    "invariant_type": "test_inv",
                    "classes": ["a", "b"],
                    "global_rank": 1,
                    "evidence": {},
                },
            ],
        }
        a = print_core_invariant_report(report)
        b = print_core_invariant_report(report)
        assert a == b
