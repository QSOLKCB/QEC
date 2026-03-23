"""Tests for friction-aware correction control (v96.4.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.friction_control import (
    apply_friction_rules,
    compute_control_decision,
    compute_efficiency,
    normalize_control_inputs,
    print_control_report,
    rank_control_strategies,
    run_friction_control,
    select_best_strategy,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_candidate(
    dfa_type="chain",
    n=10,
    mode="d4",
    stability_efficiency=0.6,
    compression_efficiency=0.3,
    friction_score=1.0,
    regime="adaptive",
    core_invariants=None,
):
    """Build a single candidate record for testing."""
    return {
        "dfa_type": dfa_type,
        "n": n,
        "mode": mode,
        "stability_efficiency": stability_efficiency,
        "compression_efficiency": compression_efficiency,
        "friction_score": friction_score,
        "regime": regime,
        "core_invariants": core_invariants or [],
    }


def _make_dynamics_result(
    dfa_type="chain",
    n=10,
    friction_score=1.0,
    regime="adaptive",
    components=None,
):
    """Build a dynamics result for normalization testing."""
    return {
        "dfa_type": dfa_type,
        "n": n,
        "friction_score": friction_score,
        "regime": regime,
        "components": components or {
            "oscillation": 0.2,
            "hysteresis": 0.1,
            "switching": 0.3,
            "churn": 0.2,
            "conflict": 0.2,
        },
    }


def _make_hierarchical_result(
    dfa_name="chain",
    n=10,
    mode="d4",
    stability_efficiency=0.6,
    compression_efficiency=0.3,
    system_class="class_a",
):
    """Build a hierarchical result for normalization testing."""
    return {
        "dfa_name": dfa_name,
        "n": n,
        "mode": mode,
        "stability_efficiency": stability_efficiency,
        "compression_efficiency": compression_efficiency,
        "system_class": system_class,
    }


def _make_pipeline_data(
    dynamics_results=None,
    hierarchical_results=None,
    law_extraction=None,
):
    """Build combined upstream data for the full pipeline."""
    data = {}
    if dynamics_results is not None:
        data["dynamics_results"] = {"results": dynamics_results}
    if hierarchical_results is not None:
        data["hierarchical_results"] = hierarchical_results
    if law_extraction is not None:
        data["law_extraction"] = law_extraction
    return data


# ---------------------------------------------------------------------------
# PART 1 — normalize_control_inputs
# ---------------------------------------------------------------------------


class TestNormalizeControlInputs:
    def test_empty_input(self):
        assert normalize_control_inputs({}) == []

    def test_non_dict_input(self):
        assert normalize_control_inputs(None) == []
        assert normalize_control_inputs([]) == []

    def test_dynamics_only(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result()],
        )
        candidates = normalize_control_inputs(data)
        assert len(candidates) == 1
        assert candidates[0]["dfa_type"] == "chain"
        assert candidates[0]["mode"] == "baseline"
        assert candidates[0]["friction_score"] == 1.0

    def test_hierarchical_with_friction(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result(friction_score=1.5)],
            hierarchical_results=[_make_hierarchical_result()],
        )
        candidates = normalize_control_inputs(data)
        assert len(candidates) == 1
        assert candidates[0]["mode"] == "d4"
        assert candidates[0]["friction_score"] == 1.5

    def test_multiple_systems(self):
        data = _make_pipeline_data(
            dynamics_results=[
                _make_dynamics_result(dfa_type="chain", n=5),
                _make_dynamics_result(dfa_type="cycle", n=10),
            ],
            hierarchical_results=[
                _make_hierarchical_result(dfa_name="chain", n=5),
                _make_hierarchical_result(dfa_name="cycle", n=10, mode="square"),
            ],
        )
        candidates = normalize_control_inputs(data)
        assert len(candidates) == 2
        # Sorted by (dfa_type, n, mode).
        assert candidates[0]["dfa_type"] == "chain"
        assert candidates[1]["dfa_type"] == "cycle"

    def test_deduplication(self):
        data = _make_pipeline_data(
            hierarchical_results=[
                _make_hierarchical_result(mode="d4"),
                _make_hierarchical_result(mode="d4"),
            ],
        )
        candidates = normalize_control_inputs(data)
        assert len(candidates) == 1

    def test_no_mutation(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result()],
            hierarchical_results=[_make_hierarchical_result()],
        )
        original = copy.deepcopy(data)
        normalize_control_inputs(data)
        assert data == original

    def test_deterministic(self):
        data = _make_pipeline_data(
            dynamics_results=[
                _make_dynamics_result(dfa_type="cycle", n=10),
                _make_dynamics_result(dfa_type="chain", n=5),
            ],
            hierarchical_results=[
                _make_hierarchical_result(dfa_name="cycle", n=10),
                _make_hierarchical_result(dfa_name="chain", n=5),
            ],
        )
        r1 = normalize_control_inputs(data)
        r2 = normalize_control_inputs(data)
        assert r1 == r2

    def test_core_invariants_from_laws(self):
        law = {
            "law_type": "core_invariant_law",
            "condition": {"invariant_type": "spectral_gap"},
            "conclusion": {"classes": ["class_a", "class_b"]},
        }
        data = _make_pipeline_data(
            hierarchical_results=[
                _make_hierarchical_result(system_class="class_a"),
            ],
            law_extraction={"laws": [law]},
        )
        candidates = normalize_control_inputs(data)
        assert len(candidates) == 1
        assert candidates[0]["core_invariants"] == ["spectral_gap"]


# ---------------------------------------------------------------------------
# PART 2 — compute_efficiency
# ---------------------------------------------------------------------------


class TestComputeEfficiency:
    def test_basic(self):
        record = _make_candidate(
            stability_efficiency=0.6, friction_score=1.0,
        )
        # 0.6 / (1 + 1.0) = 0.3
        assert compute_efficiency(record) == 0.3

    def test_zero_friction(self):
        record = _make_candidate(
            stability_efficiency=0.8, friction_score=0.0,
        )
        # 0.8 / (1 + 0.0) = 0.8
        assert compute_efficiency(record) == 0.8

    def test_high_friction(self):
        record = _make_candidate(
            stability_efficiency=0.6, friction_score=5.0,
        )
        # 0.6 / (1 + 5.0) = 0.1
        assert compute_efficiency(record) == 0.1

    def test_zero_stability(self):
        record = _make_candidate(stability_efficiency=0.0)
        assert compute_efficiency(record) == 0.0

    def test_negative_stability(self):
        record = _make_candidate(stability_efficiency=-0.1)
        assert compute_efficiency(record) == 0.0

    def test_deterministic(self):
        record = _make_candidate(
            stability_efficiency=0.7, friction_score=1.3,
        )
        r1 = compute_efficiency(record)
        r2 = compute_efficiency(record)
        assert r1 == r2

    def test_no_mutation(self):
        record = _make_candidate()
        original = copy.deepcopy(record)
        compute_efficiency(record)
        assert record == original

    def test_rounding(self):
        record = _make_candidate(
            stability_efficiency=1.0, friction_score=2.0,
        )
        # 1.0 / 3.0 = 0.333333...
        eff = compute_efficiency(record)
        assert eff == round(1.0 / 3.0, 6)


# ---------------------------------------------------------------------------
# PART 3 — rank_control_strategies / select_best_strategy
# ---------------------------------------------------------------------------


class TestRankControlStrategies:
    def test_empty(self):
        assert rank_control_strategies([]) == []

    def test_single_record(self):
        records = [_make_candidate()]
        ranked = rank_control_strategies(records)
        assert len(ranked) == 1
        assert ranked[0]["rank"] == 1
        assert "efficiency" in ranked[0]

    def test_ordering_by_efficiency(self):
        records = [
            _make_candidate(mode="low", stability_efficiency=0.3, friction_score=1.0),
            _make_candidate(mode="high", stability_efficiency=0.9, friction_score=1.0),
        ]
        ranked = rank_control_strategies(records)
        assert ranked[0]["mode"] == "high"
        assert ranked[1]["mode"] == "low"

    def test_friction_penalizes(self):
        records = [
            _make_candidate(mode="low_fric", stability_efficiency=0.6, friction_score=0.0),
            _make_candidate(mode="high_fric", stability_efficiency=0.6, friction_score=5.0),
        ]
        ranked = rank_control_strategies(records)
        assert ranked[0]["mode"] == "low_fric"

    def test_no_mutation(self):
        records = [_make_candidate(), _make_candidate(mode="square")]
        original = copy.deepcopy(records)
        rank_control_strategies(records)
        assert records == original

    def test_deterministic(self):
        records = [
            _make_candidate(mode="a", stability_efficiency=0.5),
            _make_candidate(mode="b", stability_efficiency=0.7),
            _make_candidate(mode="c", stability_efficiency=0.3),
        ]
        r1 = rank_control_strategies(records)
        r2 = rank_control_strategies(records)
        assert r1 == r2

    def test_tiebreak_by_mode(self):
        records = [
            _make_candidate(mode="z_mode", stability_efficiency=0.5, friction_score=0.0),
            _make_candidate(mode="a_mode", stability_efficiency=0.5, friction_score=0.0),
        ]
        ranked = rank_control_strategies(records)
        assert ranked[0]["mode"] == "a_mode"
        assert ranked[1]["mode"] == "z_mode"


class TestSelectBestStrategy:
    def test_empty(self):
        result = select_best_strategy([])
        assert result["mode"] == "none"
        assert result["efficiency"] == 0.0

    def test_single(self):
        records = [_make_candidate(mode="d4", stability_efficiency=0.8)]
        result = select_best_strategy(records)
        assert result["mode"] == "d4"
        assert result["efficiency"] > 0

    def test_selects_best(self):
        records = [
            _make_candidate(mode="low", stability_efficiency=0.3),
            _make_candidate(mode="high", stability_efficiency=0.9),
        ]
        result = select_best_strategy(records)
        assert result["mode"] == "high"

    def test_reason_present(self):
        records = [_make_candidate()]
        result = select_best_strategy(records)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# PART 4 — apply_friction_rules
# ---------------------------------------------------------------------------


class TestApplyFrictionRules:
    def test_no_rules_triggered(self):
        record = _make_candidate(
            mode="d4", friction_score=1.0, stability_efficiency=0.6,
        )
        result = apply_friction_rules(record)
        assert result["rejected"] is False
        assert result["rules_applied"] == []
        assert result["adjusted_efficiency"] == compute_efficiency(record)

    def test_high_friction_override(self):
        record = _make_candidate(
            mode="d4>e8_like", friction_score=3.0,
        )
        result = apply_friction_rules(record)
        assert any("high_friction_override" in r for r in result["rules_applied"])
        assert result["adjusted_efficiency"] < compute_efficiency(record)

    def test_high_friction_simple_mode_not_penalized(self):
        record = _make_candidate(
            mode="d4", friction_score=3.0,
        )
        result = apply_friction_rules(record)
        assert not any("high_friction_override" in r for r in result["rules_applied"])

    def test_oscillation_suppression(self):
        record = _make_candidate(mode="e8_like")
        components = {"oscillation": 0.8, "churn": 0.0, "conflict": 0.0}
        result = apply_friction_rules(record, components)
        assert any("oscillation_suppression" in r for r in result["rules_applied"])

    def test_oscillation_simple_mode_not_penalized(self):
        record = _make_candidate(mode="square")
        components = {"oscillation": 0.8, "churn": 0.0, "conflict": 0.0}
        result = apply_friction_rules(record, components)
        assert not any("oscillation_suppression" in r for r in result["rules_applied"])

    def test_churn_rejection(self):
        record = _make_candidate(stability_efficiency=0.1)
        components = {"oscillation": 0.0, "churn": 0.8, "conflict": 0.0}
        result = apply_friction_rules(record, components)
        assert result["rejected"] is True
        assert any("churn_rejection" in r for r in result["rules_applied"])

    def test_churn_not_rejected_high_stability(self):
        record = _make_candidate(stability_efficiency=0.6)
        components = {"oscillation": 0.0, "churn": 0.8, "conflict": 0.0}
        result = apply_friction_rules(record, components)
        assert result["rejected"] is False

    def test_conflict_avoidance(self):
        record = _make_candidate(
            core_invariants=["inv_a", "inv_b", "inv_c"],
        )
        components = {"oscillation": 0.0, "churn": 0.0, "conflict": 0.5}
        result = apply_friction_rules(record, components)
        assert any("conflict_avoidance" in r for r in result["rules_applied"])

    def test_conflict_not_triggered_few_invariants(self):
        record = _make_candidate(core_invariants=["inv_a"])
        components = {"oscillation": 0.0, "churn": 0.0, "conflict": 0.5}
        result = apply_friction_rules(record, components)
        assert not any("conflict_avoidance" in r for r in result["rules_applied"])

    def test_no_mutation(self):
        record = _make_candidate(mode="d4>e8_like", friction_score=3.0)
        original = copy.deepcopy(record)
        apply_friction_rules(record)
        assert record == original

    def test_adjusted_never_negative(self):
        record = _make_candidate(
            mode="d4>e8_like",
            friction_score=10.0,
            stability_efficiency=0.01,
            core_invariants=["a", "b", "c"],
        )
        components = {"oscillation": 1.0, "churn": 0.0, "conflict": 1.0}
        result = apply_friction_rules(record, components)
        assert result["adjusted_efficiency"] >= 0.0

    def test_deterministic(self):
        record = _make_candidate(mode="d4>e8_like", friction_score=3.0)
        r1 = apply_friction_rules(record)
        r2 = apply_friction_rules(record)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART 5 — compute_control_decision
# ---------------------------------------------------------------------------


class TestComputeControlDecision:
    def test_empty(self):
        result = compute_control_decision([])
        assert result["best_mode"] == "none"
        assert result["alternatives"] == []
        assert result["decision_trace"] == []

    def test_single_candidate(self):
        records = [_make_candidate(mode="d4")]
        result = compute_control_decision(records)
        assert result["best_mode"] == "d4"
        assert result["efficiency"] > 0

    def test_selects_best_adjusted(self):
        records = [
            _make_candidate(mode="d4", stability_efficiency=0.6, friction_score=0.5),
            _make_candidate(mode="square", stability_efficiency=0.5, friction_score=0.1),
        ]
        result = compute_control_decision(records)
        # d4: 0.6/1.5=0.4, square: 0.5/1.1=0.4545
        assert result["best_mode"] == "square"

    def test_rejects_churn_candidate(self):
        records = [
            _make_candidate(mode="bad", stability_efficiency=0.1, friction_score=0.5),
            _make_candidate(mode="good", stability_efficiency=0.6, friction_score=0.5),
        ]
        comp_map = {
            ("chain", 10): {"oscillation": 0.0, "churn": 0.8, "conflict": 0.0},
        }
        result = compute_control_decision(records, comp_map)
        # "bad" should be rejected due to churn + low stability.
        assert result["best_mode"] == "good"

    def test_all_rejected_picks_least_bad(self):
        records = [
            _make_candidate(mode="a", stability_efficiency=0.2, friction_score=0.5),
            _make_candidate(mode="b", stability_efficiency=0.1, friction_score=0.5),
        ]
        comp_map = {
            ("chain", 10): {"oscillation": 0.0, "churn": 0.8, "conflict": 0.0},
        }
        result = compute_control_decision(records, comp_map)
        # Both rejected, but "a" has higher adjusted efficiency.
        assert result["best_mode"] in ("a", "b")

    def test_alternatives_present(self):
        records = [
            _make_candidate(mode="d4", stability_efficiency=0.8),
            _make_candidate(mode="square", stability_efficiency=0.5),
        ]
        result = compute_control_decision(records)
        assert len(result["alternatives"]) == 1

    def test_decision_trace(self):
        records = [_make_candidate()]
        result = compute_control_decision(records)
        assert len(result["decision_trace"]) == 1
        assert "mode" in result["decision_trace"][0]
        assert "efficiency" in result["decision_trace"][0]

    def test_no_mutation(self):
        records = [_make_candidate(), _make_candidate(mode="square")]
        original = copy.deepcopy(records)
        compute_control_decision(records)
        assert records == original

    def test_deterministic(self):
        records = [
            _make_candidate(mode="a", stability_efficiency=0.5),
            _make_candidate(mode="b", stability_efficiency=0.7),
        ]
        r1 = compute_control_decision(records)
        r2 = compute_control_decision(records)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART 6 — run_friction_control (full pipeline)
# ---------------------------------------------------------------------------


class TestRunFrictionControl:
    def test_empty_input(self):
        result = run_friction_control({})
        assert result["candidates"] == []
        assert result["decision"]["best_mode"] == "none"

    def test_full_pipeline(self):
        data = _make_pipeline_data(
            dynamics_results=[
                _make_dynamics_result(friction_score=0.5, regime="stable"),
            ],
            hierarchical_results=[
                _make_hierarchical_result(mode="d4", stability_efficiency=0.8),
            ],
        )
        result = run_friction_control(data)
        assert len(result["candidates"]) == 1
        assert result["decision"]["best_mode"] == "d4"
        assert "summary" in result

    def test_summary_fields(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result()],
            hierarchical_results=[_make_hierarchical_result()],
        )
        result = run_friction_control(data)
        summary = result["summary"]
        assert "total_candidates" in summary
        assert "rejected_count" in summary
        assert "accepted_count" in summary
        assert "best_mode" in summary
        assert "best_efficiency" in summary
        assert "regimes" in summary

    def test_deterministic(self):
        data = _make_pipeline_data(
            dynamics_results=[
                _make_dynamics_result(dfa_type="chain", n=5),
                _make_dynamics_result(dfa_type="cycle", n=10),
            ],
            hierarchical_results=[
                _make_hierarchical_result(dfa_name="chain", n=5),
                _make_hierarchical_result(dfa_name="cycle", n=10, mode="square"),
            ],
        )
        r1 = run_friction_control(data)
        r2 = run_friction_control(data)
        assert r1 == r2

    def test_no_mutation(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result()],
            hierarchical_results=[_make_hierarchical_result()],
        )
        original = copy.deepcopy(data)
        run_friction_control(data)
        assert data == original

    def test_with_components_map(self):
        data = _make_pipeline_data(
            dynamics_results=[_make_dynamics_result(friction_score=3.0)],
            hierarchical_results=[
                _make_hierarchical_result(mode="d4>e8_like"),
            ],
        )
        comp_map = {
            ("chain", 10): {
                "oscillation": 0.8,
                "churn": 0.0,
                "conflict": 0.0,
            },
        }
        result = run_friction_control(data, comp_map)
        # Should have applied friction rules.
        trace = result["decision"]["decision_trace"]
        assert len(trace) == 1


# ---------------------------------------------------------------------------
# PART 7 — print_control_report
# ---------------------------------------------------------------------------


class TestPrintControlReport:
    def test_empty_report(self):
        report = {
            "decision": {
                "best_mode": "none",
                "efficiency": 0.0,
                "friction_score": 0.0,
                "regime": "stable",
                "alternatives": [],
                "decision_trace": [],
            },
            "summary": {
                "total_candidates": 0,
                "rejected_count": 0,
                "accepted_count": 0,
                "best_mode": "none",
                "best_efficiency": 0.0,
                "regimes": {},
            },
        }
        text = print_control_report(report)
        assert "Friction-Aware Control" in text
        assert "best_mode: none" in text

    def test_with_alternatives(self):
        report = {
            "decision": {
                "best_mode": "d4",
                "efficiency": 0.4,
                "friction_score": 1.0,
                "regime": "adaptive",
                "alternatives": [
                    {
                        "mode": "square",
                        "efficiency": 0.3,
                        "adjusted_efficiency": 0.3,
                        "friction_score": 0.5,
                        "rejected": False,
                    },
                ],
                "decision_trace": [],
            },
            "summary": {
                "total_candidates": 2,
                "rejected_count": 0,
                "accepted_count": 2,
                "best_mode": "d4",
                "best_efficiency": 0.4,
                "regimes": {"adaptive": 2},
            },
        }
        text = print_control_report(report)
        assert "best_mode: d4" in text
        assert "alternatives:" in text
        assert "square" in text

    def test_with_rejected(self):
        report = {
            "decision": {
                "best_mode": "d4",
                "efficiency": 0.4,
                "friction_score": 1.0,
                "regime": "adaptive",
                "alternatives": [
                    {
                        "mode": "bad",
                        "efficiency": 0.1,
                        "adjusted_efficiency": 0.1,
                        "friction_score": 3.0,
                        "rejected": True,
                    },
                ],
                "decision_trace": [],
            },
            "summary": {
                "total_candidates": 2,
                "rejected_count": 1,
                "accepted_count": 1,
                "best_mode": "d4",
                "best_efficiency": 0.4,
                "regimes": {"adaptive": 1, "frictional": 1},
            },
        }
        text = print_control_report(report)
        assert "[REJECTED]" in text
        assert "rejected: 1" in text

    def test_deterministic(self):
        report = run_friction_control(_make_pipeline_data(
            dynamics_results=[_make_dynamics_result()],
            hierarchical_results=[_make_hierarchical_result()],
        ))
        t1 = print_control_report(report)
        t2 = print_control_report(report)
        assert t1 == t2
