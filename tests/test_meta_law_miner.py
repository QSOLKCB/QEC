"""Tests for meta_law_miner (v98.0.0)."""

import pytest

from qec.analysis.law_promotion import Condition, Law
from qec.analysis.meta_law_miner import (
    _condition_signature,
    _extract_cooccurrences,
    _extract_shared_conditions,
    _law_condition_set,
    detect_conflicts,
    detect_redundant_laws,
    extract_meta_laws,
    group_by_action,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_law(
    law_id: str,
    action: str,
    conditions: list,
    confidence: float = 0.8,
) -> Law:
    conds = [Condition(m, op, v) for m, op, v in conditions]
    return Law(
        law_id=law_id,
        version=1,
        conditions=conds,
        action=action,
        evidence=["test_run"],
        scores={"confidence": confidence, "coverage": 0.5, "law_score": 0.4},
        created_at=0.0,
    )


# ---------------------------------------------------------------------------
# CONDITION SIGNATURE
# ---------------------------------------------------------------------------


class TestConditionSignature:
    def test_basic(self):
        c = Condition("snr", "gt", 3.0)
        sig = _condition_signature(c)
        assert sig == "snr:gt:3.0"

    def test_deterministic(self):
        c = Condition("variance", "lt", 0.5)
        assert _condition_signature(c) == _condition_signature(c)


class TestLawConditionSet:
    def test_sorted(self):
        law = _make_law("L1", "stabilize", [
            ("snr", "gt", 3.0),
            ("error_rate", "lt", 0.1),
        ])
        cs = _law_condition_set(law)
        assert cs == ("error_rate:lt:0.1", "snr:gt:3.0")


# ---------------------------------------------------------------------------
# GROUP BY ACTION
# ---------------------------------------------------------------------------


class TestGroupByAction:
    def test_single_group(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 5.0)]),
        ]
        groups = group_by_action(laws)
        assert "stabilize" in groups
        assert len(groups["stabilize"]) == 2

    def test_multiple_groups(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "reduce_oscillation", [("variance", "gt", 0.5)]),
        ]
        groups = group_by_action(laws)
        assert len(groups) == 2

    def test_empty(self):
        assert group_by_action([]) == {}


# ---------------------------------------------------------------------------
# SHARED CONDITIONS
# ---------------------------------------------------------------------------


class TestSharedConditions:
    def test_all_share(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0), ("err", "lt", 0.1)]),
        ]
        shared = _extract_shared_conditions(laws)
        assert "snr:gt:3.0" in shared

    def test_none_shared(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("err", "lt", 0.1)]),
        ]
        shared = _extract_shared_conditions(laws)
        assert shared == []

    def test_empty(self):
        assert _extract_shared_conditions([]) == []


# ---------------------------------------------------------------------------
# CO-OCCURRENCES
# ---------------------------------------------------------------------------


class TestCooccurrences:
    def test_pair_counted(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0), ("err", "lt", 0.1)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0), ("err", "lt", 0.1)]),
        ]
        co = _extract_cooccurrences(laws)
        pair = ("err:lt:0.1", "snr:gt:3.0")
        assert co[pair] == 2


# ---------------------------------------------------------------------------
# REDUNDANT LAWS
# ---------------------------------------------------------------------------


class TestRedundantLaws:
    def test_identical_laws(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        pairs = detect_redundant_laws(laws)
        assert ("L1", "L2") in pairs

    def test_different_actions(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "reduce_oscillation", [("snr", "gt", 3.0)]),
        ]
        pairs = detect_redundant_laws(laws)
        assert pairs == []

    def test_different_conditions(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 5.0)]),
        ]
        pairs = detect_redundant_laws(laws)
        assert pairs == []


# ---------------------------------------------------------------------------
# CONFLICT DETECTION
# ---------------------------------------------------------------------------


class TestConflicts:
    def test_conflicting_actions(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "reduce_oscillation", [("snr", "gt", 3.0)]),
        ]
        conflicts = detect_conflicts(laws)
        assert len(conflicts) == 1
        assert "stabilize" in conflicts[0]["actions"]
        assert "reduce_oscillation" in conflicts[0]["actions"]

    def test_no_conflict(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        conflicts = detect_conflicts(laws)
        assert conflicts == []


# ---------------------------------------------------------------------------
# EXTRACT META-LAWS (INTEGRATION)
# ---------------------------------------------------------------------------


class TestExtractMetaLaws:
    def test_empty(self):
        result = extract_meta_laws([])
        assert result["meta_laws"] == []
        assert result["redundant_pairs"] == []
        assert result["conflicts"] == []

    def test_merged_identical(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0)]),
        ]
        result = extract_meta_laws(laws)
        assert len(result["meta_laws"]) >= 1
        ml = result["meta_laws"][0]
        assert ml["implies"] == "stabilize"
        assert ml["support"] == 2
        assert ("L1", "L2") in result["redundant_pairs"]

    def test_conflicting_separate(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "reduce_oscillation", [("snr", "gt", 3.0)]),
        ]
        result = extract_meta_laws(laws)
        assert len(result["conflicts"]) == 1

    def test_deterministic_output(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0), ("err", "lt", 0.1)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0), ("var", "lt", 0.5)]),
            _make_law("L3", "reduce_oscillation", [("var", "gt", 0.5)]),
        ]
        r1 = extract_meta_laws(laws)
        r2 = extract_meta_laws(laws)
        assert r1 == r2

    def test_deterministic_100_runs(self):
        laws = [
            _make_law("L1", "stabilize", [("snr", "gt", 3.0)]),
            _make_law("L2", "stabilize", [("snr", "gt", 3.0), ("err", "lt", 0.1)]),
            _make_law("L3", "reduce_oscillation", [("var", "gt", 0.5)]),
        ]
        baseline = extract_meta_laws(laws)
        for _ in range(100):
            assert extract_meta_laws(laws) == baseline
