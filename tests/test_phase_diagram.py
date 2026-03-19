"""
Tests for decoder phase diagram aggregation (v5.9.0).

Validates:
  - Determinism (identical outputs on repeated runs)
  - Correct fractions for synthetic ternary inputs
  - Correct dominant phase with tie-breaking
  - Correct phase entropy
  - Deterministic ordering of cells
  - JSON roundtrip stability
  - Empty and single-trial edge cases
  - Grid specification validation
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from qec.diagnostics.phase_diagram import (
    build_decoder_phase_diagram,
    make_phase_grid,
    _shannon_entropy,
    _safe_mean,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_trial(
    state: int,
    boundary_eps: float | None = None,
    barrier_eps: float | None = None,
    oscillation: float | None = None,
    alignment: float | None = None,
    metastability: float | None = None,
    cluster_count: int | None = None,
) -> dict[str, Any]:
    """Build a synthetic per-trial result dict."""
    evidence: dict[str, Any] = {}
    if boundary_eps is not None:
        evidence["boundary_eps_final"] = boundary_eps
    if barrier_eps is not None:
        evidence["barrier_eps_final"] = barrier_eps
    if oscillation is not None:
        evidence["oscillation_score"] = oscillation
    if alignment is not None:
        evidence["alignment_max_final"] = alignment

    result: dict[str, Any] = {
        "final_ternary_state": state,
        "evidence": evidence,
    }
    if metastability is not None:
        result["metastability_score"] = metastability
    if cluster_count is not None:
        result["cluster_count"] = cluster_count
    return result


# ── Grid specification tests ──────────────────────────────────────────


class TestMakePhaseGrid:
    def test_basic_grid(self):
        grid = make_phase_grid("p", [0.01, 0.02], "scale", [1.0, 2.0])
        assert grid["x_name"] == "p"
        assert grid["x_values"] == [0.01, 0.02]
        assert grid["y_name"] == "scale"
        assert grid["y_values"] == [1.0, 2.0]

    def test_single_values(self):
        grid = make_phase_grid("p", [0.01], "d", [3])
        assert len(grid["x_values"]) == 1
        assert len(grid["y_values"]) == 1

    def test_empty_x_raises(self):
        with pytest.raises(ValueError, match="x_values"):
            make_phase_grid("p", [], "d", [3])

    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="y_values"):
            make_phase_grid("p", [0.01], "d", [])

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_phase_grid("", [0.01], "d", [3])


# ── Shannon entropy tests ────────────────────────────────────────────


class TestShannonEntropy:
    def test_uniform_three(self):
        h = _shannon_entropy([1 / 3, 1 / 3, 1 / 3])
        assert abs(h - math.log(3)) < 1e-10

    def test_degenerate(self):
        h = _shannon_entropy([1.0, 0.0, 0.0])
        assert h == 0.0

    def test_binary(self):
        h = _shannon_entropy([0.5, 0.5, 0.0])
        assert abs(h - math.log(2)) < 1e-10


# ── Safe mean tests ──────────────────────────────────────────────────


class TestSafeMean:
    def test_all_none(self):
        assert _safe_mean([None, None]) is None

    def test_mixed(self):
        assert _safe_mean([1.0, None, 3.0]) == 2.0

    def test_all_values(self):
        assert _safe_mean([2.0, 4.0]) == 3.0


# ── Phase diagram aggregation tests ──────────────────────────────────


class TestBuildDecoderPhaseDiagram:
    def test_all_success(self):
        """All trials succeed → dominant_phase = +1."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1), _make_trial(1), _make_trial(1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)

        assert len(result["cells"]) == 1
        cell = result["cells"][0]
        assert cell["success_fraction"] == 1.0
        assert cell["boundary_fraction"] == 0.0
        assert cell["failure_fraction"] == 0.0
        assert cell["dominant_phase"] == 1
        assert cell["phase_entropy"] == 0.0

    def test_all_failure(self):
        """All trials fail → dominant_phase = -1."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(-1), _make_trial(-1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["failure_fraction"] == 1.0
        assert cell["dominant_phase"] == -1

    def test_mixed_fractions(self):
        """Mixed trials → correct fractions."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1), _make_trial(0), _make_trial(-1), _make_trial(1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["success_fraction"] == 0.5
        assert cell["boundary_fraction"] == 0.25
        assert cell["failure_fraction"] == 0.25
        assert cell["dominant_phase"] == 1
        assert cell["trial_count"] == 4

    def test_tie_breaking(self):
        """Equal success and failure → +1 wins (deterministic tie-break)."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1), _make_trial(-1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["dominant_phase"] == 1

    def test_tie_breaking_boundary_failure(self):
        """Equal boundary and failure → 0 wins (deterministic tie-break)."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(0), _make_trial(-1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["dominant_phase"] == 0

    def test_grid_2x2(self):
        """2x2 grid → 4 cells in correct order."""
        grid = make_phase_grid("p", [0.01, 0.02], "d", [3, 5])

        def runner(x, y):
            if x == 0.01:
                return [_make_trial(1)]
            else:
                return [_make_trial(-1)]

        result = build_decoder_phase_diagram(grid, runner)
        assert len(result["cells"]) == 4

        # First two cells: x=0.01 (success), last two: x=0.02 (failure).
        assert result["cells"][0]["dominant_phase"] == 1
        assert result["cells"][1]["dominant_phase"] == 1
        assert result["cells"][2]["dominant_phase"] == -1
        assert result["cells"][3]["dominant_phase"] == -1

    def test_empty_trials(self):
        """Zero trials at a grid point → zero fractions."""
        grid = make_phase_grid("p", [0.01], "d", [3])

        def runner(x, y):
            return []

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["trial_count"] == 0
        assert cell["dominant_phase"] == 0

    def test_continuous_observables(self):
        """Continuous observables are aggregated as means."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [
            _make_trial(1, boundary_eps=0.1, barrier_eps=0.2,
                        oscillation=0.01, alignment=0.5,
                        metastability=0.1, cluster_count=2),
            _make_trial(1, boundary_eps=0.3, barrier_eps=0.4,
                        oscillation=0.03, alignment=0.7,
                        metastability=0.3, cluster_count=4),
        ]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert abs(cell["mean_boundary_eps"] - 0.2) < 1e-10
        assert abs(cell["mean_barrier_eps"] - 0.3) < 1e-10
        assert abs(cell["mean_oscillation_score"] - 0.02) < 1e-10
        assert abs(cell["mean_alignment_max"] - 0.6) < 1e-10
        assert abs(cell["mean_metastability_score"] - 0.2) < 1e-10
        assert abs(cell["mean_cluster_count"] - 3.0) < 1e-10

    def test_partial_continuous_observables(self):
        """Missing continuous observables → None when all missing."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1), _make_trial(1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["mean_boundary_eps"] is None
        assert cell["mean_barrier_eps"] is None

    def test_determinism(self):
        """Repeated runs produce identical output."""
        grid = make_phase_grid("p", [0.01, 0.02], "d", [3, 5])
        trials_map = {
            (0.01, 3): [_make_trial(1), _make_trial(0)],
            (0.01, 5): [_make_trial(1), _make_trial(1)],
            (0.02, 3): [_make_trial(-1), _make_trial(0)],
            (0.02, 5): [_make_trial(-1), _make_trial(-1)],
        }

        def runner(x, y):
            return trials_map[(x, y)]

        r1 = build_decoder_phase_diagram(grid, runner)
        r2 = build_decoder_phase_diagram(grid, runner)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_roundtrip(self):
        """Output survives JSON serialization and deserialization."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1, boundary_eps=0.1, metastability=0.2)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_grid_axes_preserved(self):
        """Grid axes in output match input."""
        grid = make_phase_grid("error_rate", [0.01, 0.05], "distance", [3, 7])

        def runner(x, y):
            return [_make_trial(1)]

        result = build_decoder_phase_diagram(grid, runner)
        assert result["grid_axes"]["x_name"] == "error_rate"
        assert result["grid_axes"]["y_name"] == "distance"
        assert result["grid_axes"]["x_values"] == [0.01, 0.05]
        assert result["grid_axes"]["y_values"] == [3, 7]

    def test_invalid_grid_raises(self):
        """Missing grid keys raise ValueError."""
        with pytest.raises(ValueError):
            build_decoder_phase_diagram({"x_name": "p"}, lambda x, y: [])

    def test_non_callable_runner_raises(self):
        """Non-callable trial_runner raises TypeError."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        with pytest.raises(TypeError, match="callable"):
            build_decoder_phase_diagram(grid, "not_callable")

    def test_phase_entropy_near_boundary(self):
        """Phase entropy is positive for mixed cells."""
        grid = make_phase_grid("p", [0.01], "d", [3])
        trials = [_make_trial(1), _make_trial(0), _make_trial(-1)]

        def runner(x, y):
            return trials

        result = build_decoder_phase_diagram(grid, runner)
        cell = result["cells"][0]
        assert cell["phase_entropy"] > 0.0
        # Uniform 1/3 each → entropy = ln(3).
        assert abs(cell["phase_entropy"] - math.log(3)) < 1e-10
