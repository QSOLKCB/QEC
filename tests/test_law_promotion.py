"""Tests for the Deterministic Law Promotion Engine (v97.5.0)."""

import json

from qec.analysis.law_promotion import (
    CONSISTENCY_REQUIRED,
    PROMOTION_THRESHOLD,
    Condition,
    Law,
    LawPromoter,
    LawRegistry,
    compute_law_score,
    compute_metrics,
    compute_simplicity,
    minimize_conditions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(metrics, action="apply"):
    return {"metrics": metrics, "action": action}


# ---------------------------------------------------------------------------
# Condition tests
# ---------------------------------------------------------------------------


class TestCondition:
    def test_gt(self):
        c = Condition("x", "gt", 5.0)
        assert c.evaluate({"x": 6.0}) is True
        assert c.evaluate({"x": 5.0}) is False
        assert c.evaluate({"x": 4.0}) is False

    def test_gte(self):
        c = Condition("x", "gte", 5.0)
        assert c.evaluate({"x": 5.0}) is True
        assert c.evaluate({"x": 4.9}) is False

    def test_lt(self):
        c = Condition("x", "lt", 5.0)
        assert c.evaluate({"x": 4.0}) is True
        assert c.evaluate({"x": 5.0}) is False

    def test_lte(self):
        c = Condition("x", "lte", 5.0)
        assert c.evaluate({"x": 5.0}) is True
        assert c.evaluate({"x": 5.1}) is False

    def test_eq(self):
        c = Condition("x", "eq", 3.0)
        assert c.evaluate({"x": 3.0}) is True
        assert c.evaluate({"x": 3.1}) is False

    def test_neq(self):
        c = Condition("x", "neq", 3.0)
        assert c.evaluate({"x": 4.0}) is True
        assert c.evaluate({"x": 3.0}) is False

    def test_missing_metric_returns_false(self):
        c = Condition("x", "gt", 0.0)
        assert c.evaluate({"y": 10.0}) is False

    def test_invalid_operator_raises(self):
        try:
            Condition("x", "invalid", 0.0)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_sort_key_deterministic(self):
        c = Condition("alpha", "gt", 1.0)
        assert c.sort_key() == "alpha:gt:1.0"

    def test_to_dict(self):
        c = Condition("x", "lt", 2.5)
        d = c.to_dict()
        assert d == {"metric": "x", "operator": "lt", "value": 2.5}


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestScoring:
    def test_coverage_all_match(self):
        conds = [Condition("x", "gt", 0.0)]
        runs = [_run({"x": 1.0}), _run({"x": 2.0})]
        m = compute_metrics(conds, "apply", runs)
        assert m["coverage"] == 1.0

    def test_coverage_half(self):
        conds = [Condition("x", "gt", 5.0)]
        runs = [_run({"x": 10.0}), _run({"x": 1.0})]
        m = compute_metrics(conds, "apply", runs)
        assert m["coverage"] == 0.5

    def test_consistency_perfect(self):
        conds = [Condition("x", "gt", 0.0)]
        runs = [_run({"x": 1.0}, "apply"), _run({"x": 2.0}, "apply")]
        m = compute_metrics(conds, "apply", runs)
        assert m["consistency"] == 1.0

    def test_consistency_imperfect(self):
        conds = [Condition("x", "gt", 0.0)]
        runs = [_run({"x": 1.0}, "apply"), _run({"x": 2.0}, "skip")]
        m = compute_metrics(conds, "apply", runs)
        assert m["consistency"] == 0.5

    def test_empty_runs(self):
        m = compute_metrics([Condition("x", "gt", 0.0)], "apply", [])
        assert m["coverage"] == 0.0
        assert m["consistency"] == 0.0

    def test_simplicity(self):
        assert compute_simplicity(0) == 1.0
        assert compute_simplicity(1) == 0.5
        assert compute_simplicity(3) == 0.25

    def test_law_score(self):
        assert compute_law_score(1.0, 0.5) == 0.5
        assert compute_law_score(0.5, 0.5) == 0.25


# ---------------------------------------------------------------------------
# Minimization tests
# ---------------------------------------------------------------------------


class TestMinimization:
    def test_redundant_condition_removed(self):
        """If one condition is always true when another is, it's redundant."""
        # x > 5 implies x > 0, so x > 0 is redundant
        conds = [Condition("x", "gt", 5.0), Condition("x", "gt", 0.0)]
        runs = [
            _run({"x": 10.0}, "apply"),
            _run({"x": 7.0}, "apply"),
            _run({"x": 1.0}, "skip"),  # x > 5 fails, action differs
        ]
        result = minimize_conditions(conds, "apply", runs)
        assert len(result) == 1
        assert result[0].metric == "x"
        assert result[0].value == 5.0

    def test_required_condition_preserved(self):
        """Both conditions needed for consistency."""
        conds = [Condition("x", "gt", 0.0), Condition("y", "gt", 0.0)]
        runs = [
            _run({"x": 1.0, "y": 1.0}, "apply"),
            _run({"x": 1.0, "y": -1.0}, "skip"),  # y fails → different action
            _run({"x": -1.0, "y": 1.0}, "skip"),  # x fails → different action
        ]
        result = minimize_conditions(conds, "apply", runs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Law tests
# ---------------------------------------------------------------------------


class TestLaw:
    def test_evaluate_all_pass(self):
        law = Law(
            "L1", 1,
            [Condition("x", "gt", 0.0), Condition("y", "lt", 10.0)],
            "apply", [], {}, 0.0,
        )
        assert law.evaluate({"x": 1.0, "y": 5.0}) is True

    def test_evaluate_one_fails(self):
        law = Law(
            "L1", 1,
            [Condition("x", "gt", 0.0), Condition("y", "lt", 10.0)],
            "apply", [], {}, 0.0,
        )
        assert law.evaluate({"x": 1.0, "y": 20.0}) is False

    def test_condition_count(self):
        law = Law("L1", 1, [Condition("x", "gt", 0.0)], "a", [], {}, 0.0)
        assert law.condition_count() == 1

    def test_to_json_deterministic(self):
        law = Law(
            "L1", 1, [Condition("x", "gt", 1.0)],
            "apply", ["e1"], {"law_score": 0.5}, 100.0,
        )
        j1 = law.to_json()
        j2 = law.to_json()
        assert j1 == j2
        parsed = json.loads(j1)
        assert parsed["id"] == "L1"
        assert parsed["conditions"] == [{"metric": "x", "operator": "gt", "value": 1.0}]


# ---------------------------------------------------------------------------
# Promotion tests
# ---------------------------------------------------------------------------


def _make_runs():
    """Standard validation runs for promotion tests."""
    return [
        _run({"x": 10.0, "y": 1.0}, "apply"),
        _run({"x": 8.0, "y": 2.0}, "apply"),
        _run({"x": 1.0, "y": 5.0}, "skip"),
        _run({"x": 2.0, "y": 6.0}, "skip"),
    ]


class TestPromotion:
    def test_valid_law_promoted(self):
        runs = _make_runs()
        registry = LawRegistry()
        promoter = LawPromoter(registry, runs, threshold=0.2)
        conjecture = {
            "conditions": [{"metric": "x", "operator": "gt", "value": 5.0}],
            "action": "apply",
            "evidence": ["run1", "run2"],
        }
        law = promoter.promote(conjecture)
        assert law is not None
        assert law.scores["consistency"] == 1.0
        assert law.action == "apply"
        assert len(registry.laws) == 1

    def test_inconsistent_rejected(self):
        runs = [
            _run({"x": 10.0}, "apply"),
            _run({"x": 8.0}, "skip"),  # condition matches but action differs
        ]
        registry = LawRegistry()
        promoter = LawPromoter(registry, runs, threshold=0.1)
        conjecture = {
            "conditions": [{"metric": "x", "operator": "gt", "value": 5.0}],
            "action": "apply",
        }
        law = promoter.promote(conjecture)
        assert law is None

    def test_low_score_rejected(self):
        # Only 1 of 10 runs matches → low coverage → low score
        runs = [_run({"x": 100.0}, "apply")] + [
            _run({"x": 0.0}, "skip") for _ in range(9)
        ]
        registry = LawRegistry()
        promoter = LawPromoter(registry, runs, threshold=0.5)
        conjecture = {
            "conditions": [{"metric": "x", "operator": "gt", "value": 50.0}],
            "action": "apply",
        }
        law = promoter.promote(conjecture)
        assert law is None  # coverage=0.1, simplicity=0.5, score=0.05


# ---------------------------------------------------------------------------
# Conflict tests
# ---------------------------------------------------------------------------


class TestConflict:
    def test_higher_score_replaces(self):
        runs = [
            _run({"x": 10.0}, "apply"),
            _run({"x": 10.0}, "apply"),
            _run({"x": 0.0}, "skip"),
        ]
        registry = LawRegistry()

        # Add a weaker law first (more conditions → lower simplicity)
        weak = Law(
            "weak", 1,
            [Condition("x", "gt", 5.0), Condition("x", "lt", 100.0)],
            "reject",
            [], {"law_score": 0.2}, 0.0,
        )
        registry.add_law(weak, runs)

        # Now a stronger conflicting law
        strong = Law(
            "strong", 1,
            [Condition("x", "gt", 5.0)],
            "apply",
            [], {"law_score": 0.8}, 0.0,
        )
        added = registry.add_law(strong, runs)
        assert added is True
        assert len(registry.laws) == 1
        assert registry.laws[0].id == "strong"

    def test_lower_score_rejected(self):
        runs = [_run({"x": 10.0}, "apply")]
        registry = LawRegistry()

        strong = Law(
            "strong", 1,
            [Condition("x", "gt", 5.0)],
            "apply",
            [], {"law_score": 0.9}, 0.0,
        )
        registry.add_law(strong, runs)

        weak = Law(
            "weak", 1,
            [Condition("x", "gt", 5.0)],
            "reject",
            [], {"law_score": 0.1}, 0.0,
        )
        added = registry.add_law(weak, runs)
        assert added is False
        assert len(registry.laws) == 1
        assert registry.laws[0].id == "strong"

    def test_specificity_tiebreak(self):
        """Equal score → more conditions (more specific) wins."""
        runs = [_run({"x": 10.0, "y": 1.0}, "apply")]
        registry = LawRegistry()

        general = Law(
            "general", 1,
            [Condition("x", "gt", 5.0)],
            "reject",
            [], {"law_score": 0.5}, 0.0,
        )
        registry.add_law(general, runs)

        specific = Law(
            "specific", 1,
            [Condition("x", "gt", 5.0), Condition("y", "lt", 10.0)],
            "apply",
            [], {"law_score": 0.5}, 0.0,
        )
        added = registry.add_law(specific, runs)
        assert added is True
        assert registry.laws[0].id == "specific"

    def test_lexicographic_tiebreak(self):
        """Equal score, equal specificity → lex smaller wins."""
        runs = [_run({"x": 10.0, "a": 5.0}, "apply")]
        registry = LawRegistry()

        law_b = Law(
            "law_b", 1,
            [Condition("x", "gt", 5.0)],
            "reject",
            [], {"law_score": 0.5}, 0.0,
        )
        registry.add_law(law_b, runs)

        # Same specificity, same score, but "a" conditions sort before "x"
        law_a = Law(
            "law_a", 1,
            [Condition("a", "gt", 0.0)],
            "apply",
            [], {"law_score": 0.5}, 0.0,
        )
        # law_a condition key: "a:gt:0.0" < "x:gt:5.0" → law_a wins
        added = registry.add_law(law_a, runs)
        assert added is True
        assert registry.laws[0].id == "law_a"

    def test_no_conflict_different_runs(self):
        """Non-overlapping conditions → no conflict → both kept."""
        runs = [
            _run({"x": 10.0, "y": 0.0}, "apply"),
            _run({"x": 0.0, "y": 10.0}, "reject"),
        ]
        registry = LawRegistry()
        law1 = Law("L1", 1, [Condition("x", "gt", 5.0)], "apply", [], {"law_score": 0.5}, 0.0)
        law2 = Law("L2", 1, [Condition("y", "gt", 5.0)], "reject", [], {"law_score": 0.5}, 0.0)
        registry.add_law(law1, runs)
        added = registry.add_law(law2, runs)
        assert added is True
        assert len(registry.laws) == 2


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_promotion_identical(self):
        runs = _make_runs()
        conjecture = {
            "conditions": [
                {"metric": "x", "operator": "gt", "value": 5.0},
                {"metric": "x", "operator": "gt", "value": 0.0},
            ],
            "action": "apply",
            "evidence": ["e1"],
            "created_at": 42.0,
        }

        results = []
        for _ in range(5):
            reg = LawRegistry()
            p = LawPromoter(reg, runs, threshold=0.2)
            law = p.promote(conjecture)
            assert law is not None
            results.append(law.to_json())

        assert all(r == results[0] for r in results)

    def test_law_id_deterministic(self):
        runs = _make_runs()
        conjecture = {
            "conditions": [{"metric": "x", "operator": "gt", "value": 5.0}],
            "action": "apply",
            "created_at": 0.0,
        }
        ids = []
        for _ in range(3):
            reg = LawRegistry()
            p = LawPromoter(reg, runs, threshold=0.2)
            law = p.promote(conjecture)
            ids.append(law.id)
        assert all(i == ids[0] for i in ids)


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------


class TestFloatPrecision:
    def test_float_drift_equality(self):
        """0.1+0.1+0.1 != 0.3 in raw float — _norm guards this."""
        c = Condition("x", "eq", 0.3)
        # 0.1+0.1+0.1 has floating-point drift
        assert c.evaluate({"x": 0.1 + 0.1 + 0.1}) is True

    def test_scoring_precision(self):
        """Coverage from 1/3 stays stable through _norm."""
        conds = [Condition("x", "gt", 0.0)]
        runs = [_run({"x": 1.0}), _run({"x": -1.0}), _run({"x": -1.0})]
        m = compute_metrics(conds, "apply", runs)
        # 1/3 normalized to 12 digits
        assert m["coverage"] == round(1.0 / 3.0, 12)


class TestStructuredEvidence:
    def test_string_evidence_normalized(self):
        law = Law("L1", 1, [], "a", ["run1"], {}, 0.0)
        assert law.evidence == [{"type": "run_ref", "ref": "run1"}]

    def test_dict_evidence_passthrough(self):
        ev = {"type": "run_ref", "ref": "run2", "metrics": {"x": 1.0}}
        law = Law("L1", 1, [], "a", [ev], {}, 0.0)
        assert law.evidence[0]["ref"] == "run2"
        assert law.evidence[0]["metrics"] == {"x": 1.0}

    def test_mixed_evidence(self):
        law = Law("L1", 1, [], "a", ["s1", {"ref": "d1"}], {}, 0.0)
        assert law.evidence[0] == {"type": "run_ref", "ref": "s1"}
        assert law.evidence[1]["ref"] == "d1"
        assert law.evidence[1]["type"] == "run_ref"


class TestThresholdConstants:
    def test_promotion_threshold_value(self):
        assert PROMOTION_THRESHOLD == 0.5

    def test_consistency_required_value(self):
        assert CONSISTENCY_REQUIRED == 1.0

    def test_promoter_uses_default_threshold(self):
        reg = LawRegistry()
        p = LawPromoter(reg, [])
        assert p.threshold == PROMOTION_THRESHOLD
