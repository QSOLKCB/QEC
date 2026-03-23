"""Tests for the Law Evolution & Theory Consistency Engine."""

import copy

from qec.analysis.law_promotion import Condition, Law
from qec.analysis.law_evolution import (
    CONFIDENCE_DECREMENT,
    CONFIDENCE_INCREMENT,
    CONFLICT_LOSS_FACTOR,
    DECAY_FACTOR,
    PRUNE_THRESHOLD,
    extract_intervals,
    intervals_overlap,
    laws_conflict,
    resolve_conflict,
    update_confidence,
    init_law,
    compute_theory_consistency,
    prune_laws,
    evolve_laws,
    _copy_law,
    _law_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_law(
    law_id: str = "law_aaa",
    conditions=None,
    action: str = "increase_iterations",
    confidence: float = 0.8,
    version: int = 1,
) -> Law:
    """Create a law with confidence and history for testing."""
    if conditions is None:
        conditions = [Condition("snr", "gt", 3.0)]
    law = Law(
        law_id=law_id,
        version=version,
        conditions=conditions,
        action=action,
        evidence=["run_1"],
        scores={"consistency": 1.0, "coverage": 0.5, "simplicity": 0.5, "law_score": 0.25},
        created_at=0.0,
    )
    return init_law(law, confidence=confidence, timestamp=0)


# ===========================================================================
# STEP 1 — INTERVAL EXTRACTION
# ===========================================================================


class TestExtractIntervals:
    def test_gt_interval(self):
        law = _make_law(conditions=[Condition("snr", "gt", 3.0)])
        ivs = extract_intervals(law)
        assert ivs["snr"] == (3.0, float("inf"))

    def test_lt_interval(self):
        law = _make_law(conditions=[Condition("snr", "lt", 5.0)])
        ivs = extract_intervals(law)
        assert ivs["snr"] == (float("-inf"), 5.0)

    def test_eq_interval(self):
        law = _make_law(conditions=[Condition("snr", "eq", 4.0)])
        ivs = extract_intervals(law)
        assert ivs["snr"] == (4.0, 4.0)

    def test_multiple_conditions_same_metric_intersect(self):
        law = _make_law(conditions=[
            Condition("snr", "gt", 2.0),
            Condition("snr", "lt", 8.0),
        ])
        ivs = extract_intervals(law)
        assert ivs["snr"] == (2.0, 8.0)

    def test_multiple_metrics(self):
        law = _make_law(conditions=[
            Condition("snr", "gt", 2.0),
            Condition("error_rate", "lt", 0.1),
        ])
        ivs = extract_intervals(law)
        assert "snr" in ivs
        assert "error_rate" in ivs

    def test_neq_skipped(self):
        law = _make_law(conditions=[Condition("snr", "neq", 0.0)])
        ivs = extract_intervals(law)
        assert "snr" not in ivs


# ===========================================================================
# STEP 2 — INTERVAL OVERLAP
# ===========================================================================


class TestIntervalsOverlap:
    def test_disjoint(self):
        assert intervals_overlap((1.0, 3.0), (5.0, 7.0)) == "disjoint"

    def test_equal(self):
        assert intervals_overlap((1.0, 5.0), (1.0, 5.0)) == "equal"

    def test_subset_a_in_b(self):
        assert intervals_overlap((2.0, 4.0), (1.0, 5.0)) == "subset"

    def test_subset_b_in_a(self):
        assert intervals_overlap((1.0, 5.0), (2.0, 4.0)) == "subset"

    def test_partial_overlap(self):
        assert intervals_overlap((1.0, 4.0), (3.0, 6.0)) == "partial"

    def test_touching_point(self):
        # lo = max(3,3) = 3, hi = min(3,6) = 3 -> not disjoint -> subset
        assert intervals_overlap((1.0, 3.0), (3.0, 6.0)) != "disjoint"

    def test_disjoint_no_touch(self):
        assert intervals_overlap((1.0, 2.0), (3.0, 4.0)) == "disjoint"


# ===========================================================================
# STEP 3 — LAW CONFLICT DETECTION
# ===========================================================================


class TestLawsConflict:
    def test_same_action_no_conflict(self):
        a = _make_law(law_id="law_a", action="act_x")
        b = _make_law(law_id="law_b", action="act_x")
        assert not laws_conflict(a, b)

    def test_different_action_overlapping_domain_conflicts(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="act_x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "gt", 3.0)],
            action="act_y",
        )
        assert laws_conflict(a, b)

    def test_different_action_disjoint_domain_no_conflict(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 5.0)],
            action="act_x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "lt", 3.0)],
            action="act_y",
        )
        assert not laws_conflict(a, b)

    def test_no_shared_metrics_treated_as_overlapping(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="act_x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("error_rate", "lt", 0.1)],
            action="act_y",
        )
        assert laws_conflict(a, b)


# ===========================================================================
# STEP 4 — CONFLICT RESOLUTION
# ===========================================================================


class TestResolveConflict:
    def test_higher_confidence_wins(self):
        a = _make_law(law_id="law_a", confidence=0.9, action="x")
        b = _make_law(law_id="law_b", confidence=0.7, action="y")
        assert resolve_conflict(a, b).id == "law_a"

    def test_higher_specificity_wins_on_tie(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            confidence=0.8,
            action="x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "gt", 2.0), Condition("er", "lt", 0.1)],
            confidence=0.8,
            action="y",
        )
        assert resolve_conflict(a, b).id == "law_b"

    def test_lexicographic_tiebreak(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            confidence=0.8,
            action="x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "gt", 2.0)],
            confidence=0.8,
            action="y",
        )
        assert resolve_conflict(a, b).id == "law_a"


# ===========================================================================
# STEP 5 — CONFIDENCE EVOLUTION
# ===========================================================================


class TestUpdateConfidence:
    def test_correct_increments(self):
        law = _make_law(confidence=0.5)
        updated = update_confidence(law, "correct", timestamp=1)
        assert updated.confidence == 0.5 + CONFIDENCE_INCREMENT

    def test_correct_capped_at_one(self):
        law = _make_law(confidence=0.999)
        updated = update_confidence(law, "correct", timestamp=1)
        assert updated.confidence == 1.0

    def test_incorrect_decrements(self):
        law = _make_law(confidence=0.5)
        updated = update_confidence(law, "incorrect", timestamp=1)
        assert updated.confidence == 0.5 - CONFIDENCE_DECREMENT

    def test_incorrect_floored_at_zero(self):
        law = _make_law(confidence=0.02)
        updated = update_confidence(law, "incorrect", timestamp=1)
        assert updated.confidence == 0.0

    def test_conflict_loss_multiplies(self):
        law = _make_law(confidence=0.8)
        updated = update_confidence(law, "conflict_loss", timestamp=1)
        expected = round(0.8 * CONFLICT_LOSS_FACTOR, 12)
        assert updated.confidence == expected

    def test_decay_multiplies(self):
        law = _make_law(confidence=0.5)
        updated = update_confidence(law, "decay", timestamp=1)
        expected = round(0.5 * DECAY_FACTOR, 12)
        assert updated.confidence == expected

    def test_history_appended(self):
        law = _make_law(confidence=0.5)
        updated = update_confidence(law, "correct", timestamp=1)
        assert len(updated.history) == len(law.history) + 1
        assert updated.history[-1]["event"] == "validated"

    def test_original_not_mutated(self):
        law = _make_law(confidence=0.5)
        original_conf = law.confidence
        _ = update_confidence(law, "correct", timestamp=1)
        assert law.confidence == original_conf


# ===========================================================================
# STEP 6 — HISTORY TRACKING
# ===========================================================================


class TestHistoryTracking:
    def test_init_law_creates_history(self):
        law = _make_law()
        assert len(law.history) == 1
        assert law.history[0]["event"] == "created"

    def test_history_is_append_only(self):
        law = _make_law(confidence=0.5)
        u1 = update_confidence(law, "correct", timestamp=1)
        u2 = update_confidence(u1, "incorrect", timestamp=2)
        assert len(u2.history) == 3  # created + validated + contradicted
        assert u2.history[0]["event"] == "created"
        assert u2.history[1]["event"] == "validated"
        assert u2.history[2]["event"] == "contradicted"

    def test_history_contains_timestamps(self):
        law = _make_law(confidence=0.5)
        u = update_confidence(law, "decay", timestamp=42)
        assert u.history[-1]["timestamp"] == 42


# ===========================================================================
# STEP 7 — THEORY CONSISTENCY
# ===========================================================================


class TestTheoryConsistency:
    def test_no_laws_returns_one(self):
        score = compute_theory_consistency([], [{"snr": 5.0}])
        assert score == 1.0

    def test_no_coverage_returns_one(self):
        law = _make_law(conditions=[Condition("snr", "gt", 10.0)])
        score = compute_theory_consistency([law], [{"snr": 1.0}])
        assert score == 1.0

    def test_single_law_no_conflict(self):
        law = _make_law(conditions=[Condition("snr", "gt", 2.0)], action="act_x")
        points = [{"snr": 5.0}, {"snr": 1.0}]
        score = compute_theory_consistency([law], points)
        assert score == 1.0

    def test_conflicting_laws_reduce_score(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="act_x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "gt", 2.0)],
            action="act_y",
        )
        points = [{"snr": 5.0}, {"snr": 1.0}]
        score = compute_theory_consistency([a, b], points)
        # 1 covered point (snr=5), both laws apply with different actions -> conflict
        # consistency = 1 - 1/1 = 0.0
        assert score == 0.0

    def test_mixed_coverage(self):
        a = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="act_x",
        )
        b = _make_law(
            law_id="law_b",
            conditions=[Condition("snr", "gt", 4.0)],
            action="act_y",
        )
        points = [{"snr": 3.0}, {"snr": 5.0}]
        # snr=3: only law_a applies -> no conflict
        # snr=5: both apply, different actions -> conflict
        # covered=2, conflicting=1 -> 1 - 1/2 = 0.5
        score = compute_theory_consistency([a, b], points)
        assert score == 0.5


# ===========================================================================
# STEP 8 — LAW PRUNING
# ===========================================================================


class TestPruneLaws:
    def test_low_confidence_removed(self):
        law = _make_law(confidence=0.05)
        result = prune_laws([law])
        assert len(result) == 0

    def test_healthy_law_kept(self):
        law = _make_law(confidence=0.8)
        result = prune_laws([law])
        assert len(result) == 1

    def test_redundant_lower_confidence_removed(self):
        a = _make_law(law_id="law_a", confidence=0.9, action="act_x",
                       conditions=[Condition("snr", "gt", 3.0)])
        b = _make_law(law_id="law_b", confidence=0.5, action="act_x",
                       conditions=[Condition("snr", "gt", 3.0)])
        result = prune_laws([a, b])
        assert len(result) == 1
        assert result[0].id == "law_a"

    def test_repeated_conflict_losers_removed(self):
        law = _make_law(confidence=0.5)
        # Simulate 3 conflict losses
        for i in range(3):
            law = update_confidence(law, "conflict_loss", timestamp=i + 1)
        result = prune_laws([law])
        assert len(result) == 0


# ===========================================================================
# STEP 9 — EVOLUTION LOOP
# ===========================================================================


class TestEvolveLaws:
    def test_correct_prediction_increases_confidence(self):
        law = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="increase_iterations",
            confidence=0.5,
        )
        results = [{"metrics": {"snr": 5.0}, "observed_action": "increase_iterations"}]
        evolved = evolve_laws([law], results, timestamp=1)
        assert len(evolved) == 1
        # Correct +0.01, then decay *0.99
        expected = round((0.5 + CONFIDENCE_INCREMENT) * DECAY_FACTOR, 12)
        assert evolved[0].confidence == expected

    def test_incorrect_prediction_decreases_confidence(self):
        law = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 2.0)],
            action="increase_iterations",
            confidence=0.5,
        )
        results = [{"metrics": {"snr": 5.0}, "observed_action": "decrease_iterations"}]
        evolved = evolve_laws([law], results, timestamp=1)
        assert len(evolved) == 1
        expected = round((0.5 - CONFIDENCE_DECREMENT) * DECAY_FACTOR, 12)
        assert evolved[0].confidence == expected

    def test_non_applicable_law_only_decays(self):
        law = _make_law(
            law_id="law_a",
            conditions=[Condition("snr", "gt", 10.0)],
            action="act_x",
            confidence=0.5,
        )
        results = [{"metrics": {"snr": 1.0}, "observed_action": "act_x"}]
        evolved = evolve_laws([law], results, timestamp=1)
        assert len(evolved) == 1
        expected = round(0.5 * DECAY_FACTOR, 12)
        assert evolved[0].confidence == expected

    def test_original_laws_not_mutated(self):
        law = _make_law(confidence=0.5)
        original_conf = law.confidence
        _ = evolve_laws([law], [], timestamp=1)
        assert law.confidence == original_conf


# ===========================================================================
# DETERMINISM
# ===========================================================================


class TestDeterminism:
    def test_evolve_deterministic_across_runs(self):
        """Identical inputs produce identical outputs."""
        laws = [
            _make_law(law_id="law_a", conditions=[Condition("snr", "gt", 2.0)],
                      action="act_x", confidence=0.7),
            _make_law(law_id="law_b", conditions=[Condition("snr", "gt", 3.0)],
                      action="act_y", confidence=0.6),
        ]
        results = [
            {"metrics": {"snr": 5.0}, "observed_action": "act_x"},
            {"metrics": {"snr": 1.0}, "observed_action": "act_z"},
        ]

        run1 = evolve_laws(laws, results, timestamp=1)
        run2 = evolve_laws(laws, results, timestamp=1)

        assert len(run1) == len(run2)
        for a, b in zip(run1, run2):
            assert a.id == b.id
            assert a.confidence == b.confidence
            assert a.action == b.action

    def test_consistency_deterministic(self):
        laws = [
            _make_law(law_id="law_a", conditions=[Condition("snr", "gt", 2.0)], action="x"),
            _make_law(law_id="law_b", conditions=[Condition("snr", "gt", 2.0)], action="y"),
        ]
        points = [{"snr": 5.0}, {"snr": 1.0}]
        s1 = compute_theory_consistency(laws, points)
        s2 = compute_theory_consistency(laws, points)
        assert s1 == s2
