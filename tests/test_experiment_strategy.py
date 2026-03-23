"""Tests for the Deterministic Experiment Selection Engine (v97.7.1).

Covers:
  - Grid construction and initialization
  - Law applicability mapping
  - Conflict detection
  - Uncertainty scoring
  - State classification
  - Boundary detection
  - Priority scoring
  - Novelty detection
  - Experiment selection (non-adjacent)
  - Cell updates
  - Full strategy loop
  - Determinism guarantees
"""

import math

import pytest

from qec.analysis.law_promotion import Condition, Law
from qec.analysis.experiment_strategy import (
    _is_adjacent,
    _manhattan_neighbors,
    _norm,
    _normalize_variance,
    BOUNDARY_BONUS,
    build_grid,
    cell_center,
    classify_state,
    compute_applicability,
    compute_boundary_cells,
    compute_conflict_grid,
    compute_priority,
    compute_uncertainty,
    is_candidate,
    is_novel,
    run_strategy,
    select_experiments,
    update_cells,
    STATE_UNEXPLORED,
    STATE_STABLE,
    STATE_UNSTABLE,
    STATE_INTERMEDIATE,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_law(
    law_id: str,
    conditions: list,
    action: str = "action_a",
    confidence: float = 0.8,
) -> Law:
    """Create a minimal Law for testing."""
    return Law(
        law_id=law_id,
        version=1,
        conditions=conditions,
        action=action,
        evidence=["test_run"],
        scores={"confidence": confidence, "coverage": 0.5},
        created_at=0.0,
    )


def _simple_grid():
    """2-metric, 3-bin grid for testing."""
    return build_grid({"x": (0.0, 3.0), "y": (0.0, 3.0)}, bins_per_metric=3)


# ---------------------------------------------------------------------------
# STEP 1 — GRID CONSTRUCTION
# ---------------------------------------------------------------------------


class TestBuildGrid:
    def test_correct_shape_1d(self):
        grid = build_grid({"x": (0.0, 1.0)}, bins_per_metric=4)
        assert len(grid["cells"]) == 4
        assert grid["metrics"] == ["x"]
        assert grid["bins"] == 4

    def test_correct_shape_2d(self):
        grid = _simple_grid()
        assert len(grid["cells"]) == 9  # 3x3
        assert grid["metrics"] == ["x", "y"]

    def test_correct_shape_3d(self):
        grid = build_grid(
            {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)},
            bins_per_metric=2,
        )
        assert len(grid["cells"]) == 8  # 2x2x2

    def test_cell_initialization(self):
        grid = build_grid({"x": (0.0, 1.0)}, bins_per_metric=2)
        for cell in grid["cells"].values():
            assert cell["coverage"] == 0
            assert cell["sum_outcome"] == 0.0
            assert cell["sum_outcome_sq"] == 0.0

    def test_centers_correct(self):
        grid = build_grid({"x": (0.0, 4.0)}, bins_per_metric=4)
        assert grid["centers"]["x"] == [0.5, 1.5, 2.5, 3.5]

    def test_invalid_bins(self):
        with pytest.raises(ValueError):
            build_grid({"x": (0.0, 1.0)}, bins_per_metric=0)

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            build_grid({"x": (1.0, 0.0)}, bins_per_metric=2)

    def test_empty_ranges(self):
        with pytest.raises(ValueError):
            build_grid({}, bins_per_metric=2)


class TestCellCenter:
    def test_center_2d(self):
        grid = _simple_grid()
        center = cell_center(grid, (0, 0))
        assert center["x"] == 0.5
        assert center["y"] == 0.5

    def test_center_last_cell(self):
        grid = _simple_grid()
        center = cell_center(grid, (2, 2))
        assert center["x"] == 2.5
        assert center["y"] == 2.5


# ---------------------------------------------------------------------------
# STEP 2 — APPLICABILITY
# ---------------------------------------------------------------------------


class TestComputeApplicability:
    def test_law_applies_to_matching_cells(self):
        grid = _simple_grid()
        law = _make_law("L1", [Condition("x", "gt", 1.0)])
        app = compute_applicability([law], grid)
        # x centers are 0.5, 1.5, 2.5. gt 1.0 matches 1.5 and 2.5.
        for key, indices in app.items():
            center = cell_center(grid, key)
            if center["x"] > 1.0:
                assert 0 in indices
            else:
                assert 0 not in indices

    def test_no_laws(self):
        grid = _simple_grid()
        app = compute_applicability([], grid)
        for indices in app.values():
            assert indices == []

    def test_multiple_laws(self):
        grid = _simple_grid()
        law1 = _make_law("L1", [Condition("x", "gt", 1.0)])
        law2 = _make_law("L2", [Condition("y", "lt", 2.0)])
        app = compute_applicability([law1, law2], grid)
        # Cell (2, 0): x=2.5 > 1.0 (L1 applies), y=0.5 < 2.0 (L2 applies)
        assert 0 in app[(2, 0)]
        assert 1 in app[(2, 0)]


# ---------------------------------------------------------------------------
# STEP 3 — CONFLICT GRID
# ---------------------------------------------------------------------------


class TestComputeConflictGrid:
    def test_no_conflict_single_law(self):
        grid = _simple_grid()
        law = _make_law("L1", [Condition("x", "gt", 0.0)])
        app = compute_applicability([law], grid)
        conflicts = compute_conflict_grid([law], app)
        assert all(not v for v in conflicts.values())

    def test_conflict_when_actions_differ(self):
        grid = _simple_grid()
        law1 = _make_law("L1", [Condition("x", "gt", 0.0)], action="action_a")
        law2 = _make_law("L2", [Condition("x", "gt", 0.0)], action="action_b")
        app = compute_applicability([law1, law2], grid)
        conflicts = compute_conflict_grid([law1, law2], app)
        # All cells have x > 0 (centers 0.5, 1.5, 2.5), so both laws apply.
        assert all(v for v in conflicts.values())

    def test_no_conflict_same_actions(self):
        grid = _simple_grid()
        law1 = _make_law("L1", [Condition("x", "gt", 0.0)], action="same")
        law2 = _make_law("L2", [Condition("y", "gt", 0.0)], action="same")
        app = compute_applicability([law1, law2], grid)
        conflicts = compute_conflict_grid([law1, law2], app)
        assert all(not v for v in conflicts.values())


# ---------------------------------------------------------------------------
# STEP 4 — UNCERTAINTY SCORE
# ---------------------------------------------------------------------------


class TestComputeUncertainty:
    def test_zero_coverage_returns_one(self):
        cell = {"coverage": 0, "sum_outcome": 0.0, "sum_outcome_sq": 0.0}
        assert compute_uncertainty(cell, [], False, 10) == 1.0

    def test_full_confidence_no_conflict_full_coverage_no_variance(self):
        cell = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 10.0}
        law = _make_law("L1", [], confidence=1.0)
        u = compute_uncertainty(cell, [law], False, 10)
        # 0.25*(1-1) + 0.25*0 + 0.25*(1-1) + 0.25*0 = 0.0
        assert u == 0.0

    def test_conflict_increases_uncertainty(self):
        cell = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 10.0}
        law = _make_law("L1", [], confidence=1.0)
        u_no = compute_uncertainty(cell, [law], False, 10)
        u_yes = compute_uncertainty(cell, [law], True, 10)
        assert u_yes > u_no

    def test_result_in_unit_interval(self):
        cell = {"coverage": 5, "sum_outcome": 3.0, "sum_outcome_sq": 5.0}
        law = _make_law("L1", [], confidence=0.5)
        u = compute_uncertainty(cell, [law], True, 10)
        assert 0.0 <= u <= 1.0

    def test_high_variance_increases_uncertainty(self):
        cell_low = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 10.0}
        cell_high = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 100.0}
        law = _make_law("L1", [], confidence=0.8)
        u_low = compute_uncertainty(cell_low, [law], False, 10)
        u_high = compute_uncertainty(cell_high, [law], False, 10)
        assert u_high > u_low


# ---------------------------------------------------------------------------
# STEP 5 — STATE CLASSIFICATION
# ---------------------------------------------------------------------------


class TestClassifyState:
    def test_unexplored(self):
        assert classify_state(0, 0.5) == STATE_UNEXPLORED

    def test_stable(self):
        assert classify_state(5, 0.2) == STATE_STABLE

    def test_unstable(self):
        assert classify_state(5, 0.8) == STATE_UNSTABLE

    def test_intermediate(self):
        assert classify_state(5, 0.5) == STATE_INTERMEDIATE

    def test_threshold_stable_boundary(self):
        assert classify_state(1, 0.3) == STATE_STABLE

    def test_threshold_unstable_boundary(self):
        assert classify_state(1, 0.7) == STATE_UNSTABLE


# ---------------------------------------------------------------------------
# STEP 6 — BOUNDARY DETECTION
# ---------------------------------------------------------------------------


class TestBoundaryDetection:
    def test_uniform_grid_no_boundaries(self):
        states = {(0, 0): "stable", (0, 1): "stable", (1, 0): "stable", (1, 1): "stable"}
        boundaries = compute_boundary_cells(states, bins=2)
        assert all(not v for v in boundaries.values())

    def test_boundary_detected(self):
        states = {(0, 0): "stable", (0, 1): "unstable", (1, 0): "stable", (1, 1): "stable"}
        boundaries = compute_boundary_cells(states, bins=2)
        assert boundaries[(0, 0)]  # neighbor (0,1) differs
        assert boundaries[(0, 1)]  # neighbor (0,0) differs

    def test_manhattan_neighbors(self):
        nbs = _manhattan_neighbors((1, 1), bins=3)
        expected = {(0, 1), (2, 1), (1, 0), (1, 2)}
        assert set(nbs) == expected

    def test_corner_fewer_neighbors(self):
        nbs = _manhattan_neighbors((0, 0), bins=3)
        expected = {(1, 0), (0, 1)}
        assert set(nbs) == expected


# ---------------------------------------------------------------------------
# STEP 7 — PRIORITY SCORE
# ---------------------------------------------------------------------------


class TestPriority:
    def test_boundary_bonus(self):
        assert compute_priority(0.5, True) > compute_priority(0.5, False)

    def test_boundary_adds_0_2(self):
        p = compute_priority(0.5, True)
        assert abs(p - 0.7) < 1e-10

    def test_candidate_not_stable(self):
        assert is_candidate(STATE_UNSTABLE, False)
        assert is_candidate(STATE_INTERMEDIATE, False)
        assert is_candidate(STATE_UNEXPLORED, False)

    def test_candidate_stable_boundary(self):
        assert is_candidate(STATE_STABLE, True)

    def test_not_candidate_stable_non_boundary(self):
        assert not is_candidate(STATE_STABLE, False)


# ---------------------------------------------------------------------------
# STEP 8 — NOVELTY DETECTION
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_zero_coverage_is_novel(self):
        cell = {"coverage": 0, "sum_outcome": 0.0, "sum_outcome_sq": 0.0}
        assert is_novel(cell, [])

    def test_no_laws_is_novel(self):
        cell = {"coverage": 5, "sum_outcome": 5.0, "sum_outcome_sq": 5.0}
        assert is_novel(cell, [])

    def test_low_variance_not_novel(self):
        cell = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 10.0}
        law = _make_law("L1", [])
        assert not is_novel(cell, [law])

    def test_high_variance_is_novel(self):
        cell = {"coverage": 10, "sum_outcome": 10.0, "sum_outcome_sq": 100.0}
        law = _make_law("L1", [])
        assert is_novel(cell, [law])


# ---------------------------------------------------------------------------
# STEP 9 — EXPERIMENT SELECTION
# ---------------------------------------------------------------------------


class TestSelectExperiments:
    def test_highest_priority_first(self):
        candidates = [((0,), 0.3), ((1,), 0.9), ((2,), 0.6)]
        selected = select_experiments(candidates, n=3)
        assert selected[0] == (1,)

    def test_no_adjacent_cells(self):
        candidates = [((0,), 0.9), ((1,), 0.8), ((2,), 0.7)]
        selected = select_experiments(candidates, n=3)
        # (0,) and (1,) are adjacent, so (1,) should be skipped.
        assert (0,) in selected
        assert (1,) not in selected
        assert (2,) in selected

    def test_respects_n_limit(self):
        candidates = [((i,), 1.0 - 0.01 * i) for i in range(0, 20, 2)]
        selected = select_experiments(candidates, n=3)
        assert len(selected) <= 3

    def test_empty_candidates(self):
        assert select_experiments([], n=5) == []

    def test_adjacency_2d(self):
        candidates = [((0, 0), 0.9), ((0, 1), 0.8), ((1, 1), 0.7)]
        selected = select_experiments(candidates, n=3)
        assert (0, 0) in selected
        # (0,1) is adjacent to (0,0) so skipped.
        assert (0, 1) not in selected
        assert (1, 1) in selected


class TestIsAdjacent:
    def test_adjacent(self):
        assert _is_adjacent((0,), (1,))
        assert _is_adjacent((1, 2), (1, 3))

    def test_not_adjacent(self):
        assert not _is_adjacent((0,), (2,))
        assert not _is_adjacent((0, 0), (1, 1))


# ---------------------------------------------------------------------------
# STEP 10 — UPDATE CELLS
# ---------------------------------------------------------------------------


class TestUpdateCells:
    def test_update_increments_coverage(self):
        grid = build_grid({"x": (0.0, 1.0)}, bins_per_metric=2)
        results = [{"cell_key": (0,), "outcome": 1.0}]
        new_grid = update_cells(grid, results)
        assert new_grid["cells"][(0,)]["coverage"] == 1
        assert grid["cells"][(0,)]["coverage"] == 0  # Original unchanged.

    def test_update_accumulates_sums(self):
        grid = build_grid({"x": (0.0, 1.0)}, bins_per_metric=2)
        results = [
            {"cell_key": (0,), "outcome": 2.0},
            {"cell_key": (0,), "outcome": 3.0},
        ]
        new_grid = update_cells(grid, results)
        assert new_grid["cells"][(0,)]["coverage"] == 2
        assert abs(new_grid["cells"][(0,)]["sum_outcome"] - 5.0) < 1e-10
        assert abs(new_grid["cells"][(0,)]["sum_outcome_sq"] - 13.0) < 1e-10

    def test_unknown_cell_ignored(self):
        grid = build_grid({"x": (0.0, 1.0)}, bins_per_metric=2)
        results = [{"cell_key": (99,), "outcome": 1.0}]
        new_grid = update_cells(grid, results)
        assert (99,) not in new_grid["cells"]


# ---------------------------------------------------------------------------
# STEP 11 — FULL STRATEGY LOOP
# ---------------------------------------------------------------------------


class TestRunStrategy:
    def test_returns_all_keys(self):
        grid = _simple_grid()
        laws = [_make_law("L1", [Condition("x", "gt", 1.0)])]
        result = run_strategy(laws, grid, n=3)
        assert "selected" in result
        assert "applicability" in result
        assert "conflicts" in result
        assert "uncertainty" in result
        assert "states" in result
        assert "boundaries" in result
        assert "priorities" in result

    def test_selected_count(self):
        grid = _simple_grid()
        laws = [_make_law("L1", [Condition("x", "gt", 1.0)])]
        result = run_strategy(laws, grid, n=2)
        assert len(result["selected"]) <= 2

    def test_no_stable_non_boundary_selected(self):
        grid = _simple_grid()
        # Give every cell full coverage to make them potentially stable.
        for key in grid["cells"]:
            grid["cells"][key]["coverage"] = 100
            grid["cells"][key]["sum_outcome"] = 100.0
            grid["cells"][key]["sum_outcome_sq"] = 100.0
        law = _make_law("L1", [Condition("x", "gt", -1.0)], confidence=1.0)
        result = run_strategy([law], grid, n=5)
        for sel in result["selected"]:
            state = result["states"][sel]
            boundary = result["boundaries"][sel]
            assert state != STATE_STABLE or boundary


# ---------------------------------------------------------------------------
# VARIANCE NORMALIZATION
# ---------------------------------------------------------------------------


class TestNormalizeVariance:
    def test_zero_returns_zero(self):
        assert _normalize_variance(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert _normalize_variance(-1.0) == 0.0

    def test_output_in_unit_interval(self):
        for v in [0.001, 0.1, 1.0, 10.0, 100.0, 1e6]:
            result = _normalize_variance(v)
            assert 0.0 <= result <= 1.0, f"Failed for v={v}: got {result}"

    def test_monotonically_increasing(self):
        values = [0.0, 0.1, 1.0, 10.0, 100.0]
        results = [_normalize_variance(v) for v in values]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_known_value(self):
        # v=1 -> 1/(1+1) = 0.5
        assert _normalize_variance(1.0) == 0.5


# ---------------------------------------------------------------------------
# BOUNDARY BONUS CONSTANT
# ---------------------------------------------------------------------------


class TestBoundaryBonusConstant:
    def test_default_value(self):
        assert BOUNDARY_BONUS == 0.2

    def test_used_in_priority(self):
        p = compute_priority(0.5, True)
        expected = 0.5 + BOUNDARY_BONUS
        assert abs(p - expected) < 1e-10


# ---------------------------------------------------------------------------
# SELECT EXPERIMENTS SIGNATURE
# ---------------------------------------------------------------------------


class TestSelectExperimentsSignature:
    def test_two_arg_call(self):
        """select_experiments requires only candidates and n."""
        candidates = [((0,), 0.9), ((2,), 0.8)]
        result = select_experiments(candidates, n=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# DETERMINISM TESTS
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_grid_deterministic(self):
        g1 = build_grid({"x": (0.0, 1.0), "y": (0.0, 1.0)}, bins_per_metric=5)
        g2 = build_grid({"x": (0.0, 1.0), "y": (0.0, 1.0)}, bins_per_metric=5)
        assert list(g1["cells"].keys()) == list(g2["cells"].keys())
        assert g1["centers"] == g2["centers"]

    def test_strategy_deterministic(self):
        grid = _simple_grid()
        laws = [
            _make_law("L1", [Condition("x", "gt", 1.0)], action="a"),
            _make_law("L2", [Condition("y", "lt", 2.0)], action="b"),
        ]
        r1 = run_strategy(laws, grid, n=3)
        r2 = run_strategy(laws, grid, n=3)
        assert r1["selected"] == r2["selected"]
        assert r1["uncertainty"] == r2["uncertainty"]
        assert r1["states"] == r2["states"]

    def test_selection_deterministic(self):
        candidates = [((i, j), 0.5 + 0.01 * i) for i in range(5) for j in range(5)]
        s1 = select_experiments(candidates, n=5)
        s2 = select_experiments(candidates, n=5)
        assert s1 == s2
