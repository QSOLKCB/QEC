"""Tests for deterministic law extraction and rulebook generation (v96.2.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.law_extraction import (
    LAW_TEMPLATES,
    aggregate_class_hierarchy,
    aggregate_class_invariant,
    aggregate_gap_invariant,
    aggregate_interactions,
    build_rulebook,
    extract_class_hierarchy_laws,
    extract_class_invariant_laws,
    extract_core_invariant_laws,
    extract_gap_invariant_laws,
    extract_interaction_laws,
    normalize_law_inputs,
    print_rulebook,
    rank_laws,
    render_law_text,
    run_law_extraction,
    _assign_confidence,
    _law_sort_key,
    _make_evidence_record,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_class_effectiveness(
    sys_class="cycle_like",
    inv_type="explicit_allowed_state_constraint",
    count=3,
    improved_ratio=0.8,
    avg_score=2.0,
):
    """Build a class_effectiveness dict for one class/invariant pair."""
    return {
        sys_class: {
            inv_type: {
                "count": count,
                "improved_count": int(count * improved_ratio),
                "improved_ratio": improved_ratio,
                "avg_score": avg_score,
            }
        }
    }


def _make_interaction(pair=("a", "b"), ix_type="conflict", evidence_count=3):
    """Build an interactions list entry."""
    return {
        "pair": pair,
        "type": ix_type,
        "evidence_count": evidence_count,
    }


def _make_structural_data(
    dfa_type="chain",
    n=5,
    gap_type="mode_disagreement",
    suggested_invariant="geometry_alignment_constraint",
    confidence="high",
):
    """Build structural diagnostics-like data."""
    return {
        "gaps": [
            {"dfa_type": dfa_type, "n": n, "gap_type": gap_type},
        ],
        "invariant_opportunities": [
            {
                "dfa_type": dfa_type,
                "n": n,
                "gap_type": gap_type,
                "suggested_invariant": suggested_invariant,
                "confidence": confidence,
            },
        ],
    }


def _make_core_invariants_data(
    inv_type="explicit_allowed_state_constraint",
    classes=None,
    avg_class_score=1.5,
):
    """Build core_invariants output."""
    if classes is None:
        classes = ["cycle_like", "chain_like"]
    per_class = {
        cls: {"improved_ratio": 0.7, "avg_score": avg_class_score, "count": 3}
        for cls in classes
    }
    return {
        "core_invariants": [
            {
                "invariant_type": inv_type,
                "classes": classes,
                "global_rank": 1,
                "evidence": {
                    "per_class": per_class,
                    "num_classes": len(classes),
                    "avg_class_score": avg_class_score,
                },
            }
        ]
    }


def _make_hierarchical_data(
    dfa_type="chain",
    n=5,
    sys_class="chain_like",
    best_mode="d4>e8_like",
    improved=True,
    score=2.5,
):
    """Build hierarchical pipeline output."""
    return {
        "global_best_modes": [
            {
                "dfa_type": dfa_type,
                "n": n,
                "system_class": sys_class,
                "best_mode": best_mode,
                "improved_over_baseline": improved,
                "score": score,
            }
        ]
    }


def _make_full_data():
    """Build a combined data dict covering all input paths."""
    data = {}
    data["class_effectiveness"] = {
        "cycle_like": {
            "explicit_allowed_state_constraint": {
                "count": 3,
                "improved_count": 3,
                "improved_ratio": 1.0,
                "avg_score": 2.0,
            },
        },
        "chain_like": {
            "local_stability_constraint": {
                "count": 2,
                "improved_count": 1,
                "improved_ratio": 0.5,
                "avg_score": 1.0,
            },
        },
    }
    data["interactions"] = [
        {
            "pair": (
                "geometry_alignment_constraint",
                "bounded_projection_constraint",
            ),
            "type": "conflict",
            "evidence_count": 3,
        },
    ]
    data["invariant_opportunities"] = [
        {
            "dfa_type": "chain",
            "n": 5,
            "gap_type": "mode_disagreement",
            "suggested_invariant": "geometry_alignment_constraint",
            "confidence": "high",
        },
        {
            "dfa_type": "cycle",
            "n": 8,
            "gap_type": "mode_disagreement",
            "suggested_invariant": "geometry_alignment_constraint",
            "confidence": "medium",
        },
    ]
    data["gaps"] = [
        {"dfa_type": "chain", "n": 5, "gap_type": "mode_disagreement"},
        {"dfa_type": "cycle", "n": 8, "gap_type": "mode_disagreement"},
    ]
    data["core_invariants"] = [
        {
            "invariant_type": "explicit_allowed_state_constraint",
            "classes": ["chain_like", "cycle_like"],
            "global_rank": 1,
            "evidence": {
                "per_class": {
                    "chain_like": {
                        "improved_ratio": 0.7,
                        "avg_score": 1.5,
                        "count": 3,
                    },
                    "cycle_like": {
                        "improved_ratio": 0.9,
                        "avg_score": 2.0,
                        "count": 3,
                    },
                },
                "num_classes": 2,
                "avg_class_score": 1.75,
            },
        }
    ]
    data["global_best_modes"] = [
        {
            "dfa_type": "chain",
            "n": 5,
            "system_class": "chain_like",
            "best_mode": "d4>e8_like",
            "improved_over_baseline": True,
            "score": 2.5,
        },
        {
            "dfa_type": "chain",
            "n": 7,
            "system_class": "chain_like",
            "best_mode": "d4>e8_like",
            "improved_over_baseline": True,
            "score": 2.0,
        },
    ]
    return data


# ---------------------------------------------------------------------------
# TEST: LAW TEMPLATES
# ---------------------------------------------------------------------------


class TestLawTemplates:
    def test_templates_defined(self):
        assert len(LAW_TEMPLATES) == 5

    def test_template_names(self):
        assert "class_invariant_law" in LAW_TEMPLATES
        assert "class_hierarchy_law" in LAW_TEMPLATES
        assert "interaction_law" in LAW_TEMPLATES
        assert "gap_invariant_law" in LAW_TEMPLATES
        assert "core_invariant_law" in LAW_TEMPLATES


# ---------------------------------------------------------------------------
# TEST: NORMALIZATION
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_empty_input(self):
        assert normalize_law_inputs({}) == []

    def test_non_dict_input(self):
        assert normalize_law_inputs(None) == []
        assert normalize_law_inputs([]) == []
        assert normalize_law_inputs("bad") == []

    def test_class_effectiveness_extraction(self):
        data = {"class_effectiveness": _make_class_effectiveness()}
        records = normalize_law_inputs(data)
        assert len(records) == 3
        assert all(r["system_class"] == "cycle_like" for r in records)
        assert all(
            r["invariant_type"] == "explicit_allowed_state_constraint"
            for r in records
        )

    def test_interaction_extraction(self):
        data = {"interactions": [_make_interaction()]}
        records = normalize_law_inputs(data)
        assert len(records) == 3
        assert all(r["interaction_type"] == "conflict" for r in records)

    def test_structural_extraction(self):
        data = _make_structural_data()
        records = normalize_law_inputs(data)
        assert len(records) >= 2
        gap_records = [r for r in records if r["gap_type"]]
        assert len(gap_records) >= 1

    def test_core_invariants_extraction(self):
        data = _make_core_invariants_data()
        records = normalize_law_inputs(data)
        assert len(records) == 2
        classes = sorted(r["system_class"] for r in records)
        assert classes == ["chain_like", "cycle_like"]

    def test_hierarchical_extraction(self):
        data = _make_hierarchical_data()
        records = normalize_law_inputs(data)
        assert len(records) == 1
        assert records[0]["hierarchical_mode"] == "d4>e8_like"

    def test_no_mutation(self):
        data = _make_full_data()
        data_copy = copy.deepcopy(data)
        normalize_law_inputs(data)
        assert data == data_copy

    def test_deterministic_ordering(self):
        data = _make_full_data()
        r1 = normalize_law_inputs(data)
        r2 = normalize_law_inputs(data)
        assert r1 == r2

    def test_combined_inputs(self):
        data = _make_full_data()
        records = normalize_law_inputs(data)
        assert len(records) > 0
        # Must have records from multiple sources.
        has_class = any(r["system_class"] for r in records)
        has_gap = any(r["gap_type"] for r in records)
        assert has_class
        assert has_gap


# ---------------------------------------------------------------------------
# TEST: AGGREGATION
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_class_invariant_aggregation(self):
        records = [
            _make_evidence_record(
                system_class="cycle_like",
                invariant_type="explicit_allowed_state_constraint",
                improved=True,
                avg_score=2.0,
            ),
            _make_evidence_record(
                system_class="cycle_like",
                invariant_type="explicit_allowed_state_constraint",
                improved=True,
                avg_score=1.0,
            ),
            _make_evidence_record(
                system_class="cycle_like",
                invariant_type="explicit_allowed_state_constraint",
                improved=False,
                avg_score=0.5,
            ),
        ]
        agg = aggregate_class_invariant(records)
        key = ("cycle_like", "explicit_allowed_state_constraint")
        assert key in agg
        assert agg[key]["support_count"] == 3
        assert agg[key]["improved_count"] == 2
        assert abs(agg[key]["improved_ratio"] - 0.666667) < 0.001
        assert abs(agg[key]["mean_score"] - 1.166667) < 0.001

    def test_class_invariant_skips_interactions(self):
        records = [
            _make_evidence_record(
                system_class="cycle_like",
                invariant_type="a",
                interaction_type="conflict",
            ),
        ]
        agg = aggregate_class_invariant(records)
        assert len(agg) == 0

    def test_class_hierarchy_aggregation(self):
        records = [
            _make_evidence_record(
                system_class="chain_like",
                hierarchical_mode="d4>e8_like",
                improved=True,
            ),
            _make_evidence_record(
                system_class="chain_like",
                hierarchical_mode="d4>e8_like",
                improved=False,
            ),
        ]
        agg = aggregate_class_hierarchy(records)
        key = ("chain_like", "d4>e8_like")
        assert key in agg
        assert agg[key]["count"] == 2
        assert agg[key]["wins_vs_baseline"] == 1
        assert agg[key]["win_ratio"] == 0.5

    def test_interaction_aggregation(self):
        records = [
            _make_evidence_record(
                invariant_type="a,b",
                interaction_type="conflict",
            ),
            _make_evidence_record(
                invariant_type="a,b",
                interaction_type="conflict",
            ),
        ]
        agg = aggregate_interactions(records)
        key = ("a,b", "conflict")
        assert key in agg
        assert agg[key]["evidence_count"] == 2

    def test_gap_invariant_aggregation(self):
        records = [
            _make_evidence_record(
                gap_type="mode_disagreement",
                invariant_type="geometry_alignment_constraint",
            ),
            _make_evidence_record(
                gap_type="mode_disagreement",
                invariant_type="geometry_alignment_constraint",
            ),
        ]
        agg = aggregate_gap_invariant(records)
        key = ("mode_disagreement", "geometry_alignment_constraint")
        assert key in agg
        assert agg[key]["count"] == 2

    def test_gap_aggregation_no_invariant(self):
        """Gaps without invariant_type still aggregate."""
        records = [
            _make_evidence_record(gap_type="mode_disagreement"),
            _make_evidence_record(gap_type="mode_disagreement"),
        ]
        agg = aggregate_gap_invariant(records)
        key = ("mode_disagreement", "")
        assert key in agg
        assert agg[key]["count"] == 2

    def test_aggregation_no_mutation(self):
        records = [
            _make_evidence_record(
                system_class="chain_like",
                invariant_type="a",
                improved=True,
                avg_score=1.0,
            ),
        ]
        records_copy = copy.deepcopy(records)
        aggregate_class_invariant(records)
        assert records == records_copy


# ---------------------------------------------------------------------------
# TEST: LAW EXTRACTION
# ---------------------------------------------------------------------------


class TestLawExtraction:
    def test_class_invariant_law_promoted(self):
        agg = {
            ("cycle_like", "explicit_allowed_state_constraint"): {
                "support_count": 3,
                "improved_count": 3,
                "improved_ratio": 1.0,
                "mean_score": 2.0,
            }
        }
        laws = extract_class_invariant_laws(agg)
        assert len(laws) == 1
        assert laws[0]["law_type"] == "class_invariant_law"
        assert laws[0]["condition"]["system_class"] == "cycle_like"

    def test_class_invariant_law_rejected_low_support(self):
        agg = {
            ("cycle_like", "a"): {
                "support_count": 1,
                "improved_count": 1,
                "improved_ratio": 1.0,
                "mean_score": 2.0,
            }
        }
        laws = extract_class_invariant_laws(agg)
        assert len(laws) == 0

    def test_class_invariant_law_rejected_low_ratio(self):
        agg = {
            ("cycle_like", "a"): {
                "support_count": 4,
                "improved_count": 1,
                "improved_ratio": 0.25,
                "mean_score": 2.0,
            }
        }
        laws = extract_class_invariant_laws(agg)
        assert len(laws) == 0

    def test_class_invariant_law_rejected_zero_score(self):
        agg = {
            ("cycle_like", "a"): {
                "support_count": 4,
                "improved_count": 4,
                "improved_ratio": 1.0,
                "mean_score": 0.0,
            }
        }
        laws = extract_class_invariant_laws(agg)
        assert len(laws) == 0

    def test_class_hierarchy_law_promoted(self):
        agg = {
            ("chain_like", "d4>e8_like"): {
                "count": 3,
                "wins_vs_baseline": 2,
                "win_ratio": 0.666667,
            }
        }
        laws = extract_class_hierarchy_laws(agg)
        assert len(laws) == 1
        assert laws[0]["law_type"] == "class_hierarchy_law"

    def test_class_hierarchy_law_rejected(self):
        agg = {
            ("chain_like", "d4>e8_like"): {
                "count": 1,
                "wins_vs_baseline": 1,
                "win_ratio": 1.0,
            }
        }
        laws = extract_class_hierarchy_laws(agg)
        assert len(laws) == 0

    def test_interaction_law_promoted(self):
        agg = {
            ("a,b", "conflict"): {
                "pair": "a,b",
                "interaction_type": "conflict",
                "evidence_count": 3,
            }
        }
        laws = extract_interaction_laws(agg)
        assert len(laws) == 1
        assert laws[0]["law_type"] == "interaction_law"

    def test_interaction_law_rejected(self):
        agg = {
            ("a,b", "conflict"): {
                "pair": "a,b",
                "interaction_type": "conflict",
                "evidence_count": 1,
            }
        }
        laws = extract_interaction_laws(agg)
        assert len(laws) == 0

    def test_gap_invariant_law_promoted(self):
        agg = {
            ("mode_disagreement", "geometry_alignment_constraint"): {
                "gap_type": "mode_disagreement",
                "suggested_invariant": "geometry_alignment_constraint",
                "count": 2,
            }
        }
        laws = extract_gap_invariant_laws(agg)
        assert len(laws) == 1
        assert laws[0]["law_type"] == "gap_invariant_law"

    def test_gap_invariant_law_rejected_no_invariant(self):
        agg = {
            ("mode_disagreement", ""): {
                "gap_type": "mode_disagreement",
                "suggested_invariant": "",
                "count": 5,
            }
        }
        laws = extract_gap_invariant_laws(agg)
        assert len(laws) == 0

    def test_core_invariant_law_extraction(self):
        data = _make_core_invariants_data()
        laws = extract_core_invariant_laws(data)
        assert len(laws) == 1
        assert laws[0]["law_type"] == "core_invariant_law"
        assert "chain_like" in laws[0]["conclusion"]["classes"]

    def test_core_invariant_law_empty(self):
        laws = extract_core_invariant_laws({})
        assert len(laws) == 0


# ---------------------------------------------------------------------------
# TEST: RANKING
# ---------------------------------------------------------------------------


class TestRanking:
    def test_rank_assigns_ranks(self):
        laws = [
            {
                "law_type": "class_invariant_law",
                "condition": {"system_class": "a"},
                "conclusion": {"invariant_type": "x", "effect": "good"},
                "evidence": {
                    "support_count": 3,
                    "improved_count": 3,
                    "improved_ratio": 1.0,
                    "mean_score": 2.0,
                },
            },
            {
                "law_type": "class_invariant_law",
                "condition": {"system_class": "b"},
                "conclusion": {"invariant_type": "y", "effect": "good"},
                "evidence": {
                    "support_count": 2,
                    "improved_count": 1,
                    "improved_ratio": 0.5,
                    "mean_score": 1.0,
                },
            },
        ]
        ranked = rank_laws(laws)
        assert len(ranked) == 2
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_rank_deterministic(self):
        laws = [
            {
                "law_type": "class_invariant_law",
                "condition": {"system_class": "a"},
                "conclusion": {"invariant_type": "x", "effect": "good"},
                "evidence": {
                    "support_count": 3,
                    "improved_count": 3,
                    "improved_ratio": 1.0,
                    "mean_score": 2.0,
                },
            },
            {
                "law_type": "interaction_law",
                "condition": {"invariant_pair": "a,b"},
                "conclusion": {
                    "interaction_type": "conflict",
                    "effect": "conflict is likely",
                },
                "evidence": {"evidence_count": 5},
            },
        ]
        r1 = rank_laws(laws)
        r2 = rank_laws(laws)
        assert r1 == r2

    def test_confidence_strong(self):
        law = {
            "law_type": "class_invariant_law",
            "condition": {},
            "conclusion": {},
            "evidence": {"support_count": 4, "improved_ratio": 0.8},
        }
        assert _assign_confidence(law) == "strong"

    def test_confidence_moderate_low_support(self):
        law = {
            "law_type": "class_invariant_law",
            "condition": {},
            "conclusion": {},
            "evidence": {"support_count": 2, "improved_ratio": 0.9},
        }
        assert _assign_confidence(law) == "moderate"

    def test_confidence_moderate_low_ratio(self):
        law = {
            "law_type": "class_invariant_law",
            "condition": {},
            "conclusion": {},
            "evidence": {"support_count": 5, "improved_ratio": 0.5},
        }
        assert _assign_confidence(law) == "moderate"

    def test_rank_no_mutation(self):
        laws = [
            {
                "law_type": "class_invariant_law",
                "condition": {"system_class": "a"},
                "conclusion": {"invariant_type": "x", "effect": "good"},
                "evidence": {
                    "support_count": 3,
                    "improved_count": 3,
                    "improved_ratio": 1.0,
                    "mean_score": 2.0,
                },
            },
        ]
        laws_copy = copy.deepcopy(laws)
        rank_laws(laws)
        assert laws == laws_copy


# ---------------------------------------------------------------------------
# TEST: RULEBOOK
# ---------------------------------------------------------------------------


class TestRulebook:
    def test_build_rulebook_structure(self):
        laws = [
            {
                "law_type": "class_invariant_law",
                "condition": {"system_class": "cycle_like"},
                "conclusion": {
                    "invariant_type": "explicit_allowed_state_constraint",
                    "effect": "good",
                },
                "evidence": {},
                "confidence": "strong",
                "rank": 1,
            },
        ]
        rb = build_rulebook(laws)
        assert "laws" in rb
        assert "by_class" in rb
        assert "by_invariant" in rb
        assert "by_law_type" in rb
        assert "cycle_like" in rb["by_class"]
        assert "explicit_allowed_state_constraint" in rb["by_invariant"]
        assert "class_invariant_law" in rb["by_law_type"]

    def test_build_rulebook_empty(self):
        rb = build_rulebook([])
        assert rb["laws"] == []
        assert rb["by_class"] == {}
        assert rb["by_invariant"] == {}
        assert rb["by_law_type"] == {}

    def test_build_rulebook_interaction_indexes_pair(self):
        laws = [
            {
                "law_type": "interaction_law",
                "condition": {"invariant_pair": "alpha,beta"},
                "conclusion": {
                    "interaction_type": "conflict",
                    "effect": "bad",
                },
                "evidence": {},
                "confidence": "moderate",
                "rank": 1,
            },
        ]
        rb = build_rulebook(laws)
        assert "alpha" in rb["by_invariant"]
        assert "beta" in rb["by_invariant"]


# ---------------------------------------------------------------------------
# TEST: RENDERING
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_class_invariant(self):
        law = {
            "law_type": "class_invariant_law",
            "condition": {"system_class": "cycle_like"},
            "conclusion": {
                "invariant_type": "explicit_allowed_state_constraint",
                "effect": "tends to improve correction",
            },
            "confidence": "strong",
        }
        text = render_law_text(law)
        assert text.startswith("[strong]")
        assert "system_class=cycle_like" in text
        assert "explicit_allowed_state_constraint" in text

    def test_render_class_hierarchy(self):
        law = {
            "law_type": "class_hierarchy_law",
            "condition": {
                "system_class": "chain_like",
                "hierarchical_mode": "d4>e8_like",
            },
            "conclusion": {"effect": "tends to outperform baseline"},
            "confidence": "strong",
        }
        text = render_law_text(law)
        assert "hierarchical_mode=d4>e8_like" in text

    def test_render_interaction(self):
        law = {
            "law_type": "interaction_law",
            "condition": {"invariant_pair": "a,b"},
            "conclusion": {
                "interaction_type": "conflict",
                "effect": "conflict is likely",
            },
            "confidence": "moderate",
        }
        text = render_law_text(law)
        assert "[moderate]" in text
        assert "invariant_pair=(a,b)" in text

    def test_render_gap_invariant(self):
        law = {
            "law_type": "gap_invariant_law",
            "condition": {"gap_type": "mode_disagreement"},
            "conclusion": {
                "suggested_invariant": "geometry_alignment_constraint",
                "effect": "is usually indicated",
            },
            "confidence": "moderate",
        }
        text = render_law_text(law)
        assert "gap=mode_disagreement" in text
        assert "geometry_alignment_constraint" in text

    def test_render_core_invariant(self):
        law = {
            "law_type": "core_invariant_law",
            "condition": {
                "invariant_type": "explicit_allowed_state_constraint",
                "min_classes": 2,
            },
            "conclusion": {
                "classes": ["chain_like", "cycle_like"],
                "effect": "reusable structural law across classes",
            },
            "confidence": "strong",
        }
        text = render_law_text(law)
        assert "spans >=2 classes" in text
        assert "chain_like,cycle_like" in text

    def test_render_unknown_type(self):
        law = {
            "law_type": "unknown_type",
            "condition": {},
            "conclusion": {},
            "confidence": "moderate",
        }
        text = render_law_text(law)
        assert "UNKNOWN LAW TYPE" in text


# ---------------------------------------------------------------------------
# TEST: FULL PIPELINE
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_pipeline_runs(self):
        data = _make_full_data()
        result = run_law_extraction(data)
        assert "laws" in result
        assert "rulebook" in result
        assert "summary" in result

    def test_full_pipeline_laws_ranked(self):
        data = _make_full_data()
        result = run_law_extraction(data)
        laws = result["laws"]
        assert len(laws) > 0
        for law in laws:
            assert "rank" in law
            assert "confidence" in law

    def test_full_pipeline_summary(self):
        data = _make_full_data()
        result = run_law_extraction(data)
        summary = result["summary"]
        assert "law_count" in summary
        assert "strong_count" in summary
        assert "moderate_count" in summary
        assert "law_types" in summary
        assert "classes_covered" in summary
        assert "invariants_covered" in summary
        assert summary["law_count"] == summary["strong_count"] + summary["moderate_count"]

    def test_full_pipeline_deterministic(self):
        data = _make_full_data()
        r1 = run_law_extraction(data)
        r2 = run_law_extraction(data)
        assert r1 == r2

    def test_full_pipeline_no_mutation(self):
        data = _make_full_data()
        data_copy = copy.deepcopy(data)
        run_law_extraction(data)
        assert data == data_copy

    def test_full_pipeline_empty_input(self):
        result = run_law_extraction({})
        assert result["laws"] == []
        assert result["summary"]["law_count"] == 0


# ---------------------------------------------------------------------------
# TEST: PRINT LAYER
# ---------------------------------------------------------------------------


class TestPrintLayer:
    def test_print_rulebook_output(self, capsys):
        data = _make_full_data()
        report = run_law_extraction(data)
        text = print_rulebook(report)
        assert "=== Deterministic Rulebook ===" in text
        assert "--- Summary ---" in text
        assert "Total laws:" in text
        # Also verify it printed to stdout.
        captured = capsys.readouterr()
        assert "=== Deterministic Rulebook ===" in captured.out

    def test_print_rulebook_deterministic(self):
        data = _make_full_data()
        report = run_law_extraction(data)
        t1 = print_rulebook(report)
        t2 = print_rulebook(report)
        assert t1 == t2

    def test_print_rulebook_empty(self):
        report = {"laws": [], "summary": {"law_count": 0, "strong_count": 0, "moderate_count": 0, "law_types": {}, "classes_covered": [], "invariants_covered": []}}
        text = print_rulebook(report)
        assert "Total laws: 0" in text
