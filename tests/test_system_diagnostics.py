"""Tests for v104.0.0 system diagnostics.

Verifies:
- deterministic output
- no mutation of inputs
- correct aggregation of global metrics
- integration across all analysis modules
- format output correctness
"""

from __future__ import annotations

import copy

from qec.analysis.system_diagnostics import (
    ROUND_PRECISION,
    format_system_diagnostics,
    run_system_diagnostics,
)
from qec.analysis.strategy_adapter import (
    format_system_diagnostics_adapter_summary,
    run_system_diagnostics_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(name: str, design_score: float) -> dict:
    return {"name": name, "metrics": {"design_score": design_score}}


def _make_runs() -> list[dict]:
    """Build minimal multi-run data for full pipeline execution."""
    return [
        {
            "strategies": [
                _make_strategy("A", 0.5),
                _make_strategy("B", 0.6),
                _make_strategy("C", 0.7),
            ],
        },
        {
            "strategies": [
                _make_strategy("A", 0.55),
                _make_strategy("B", 0.58),
                _make_strategy("C", 0.72),
            ],
        },
        {
            "strategies": [
                _make_strategy("A", 0.52),
                _make_strategy("B", 0.61),
                _make_strategy("C", 0.69),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# run_system_diagnostics
# ---------------------------------------------------------------------------


class TestRunSystemDiagnostics:
    """Tests for ``run_system_diagnostics``."""

    def test_returns_all_keys(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        expected_keys = {
            "trajectory",
            "taxonomy",
            "evolution",
            "transition_graph",
            "phase_space",
            "flow_geometry",
            "multistate",
            "coupled_dynamics",
            "control",
            "feedback",
            "global_control",
            "hierarchical",
            "policy_memory",
            "policy_clustering",
            "strategy_graph",
            "global_metrics",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_global_metrics_keys(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        expected = {
            "dominant_strategy",
            "system_stability",
            "convergence_rate",
            "volatility_score",
            "topology_type",
            "best_policy",
            "best_archetype",
            "angular_velocity",
            "spiral_score",
            "basin_switch_risk",
            "primary_diagnosis",
            "diagnosis_confidence",
            "baseline_response_class",
            "revised_diagnosis",
            "diagnosis_shift",
            "best_treatment",
            "treatment_score",
            "invariant_count",
            "strongest_invariant",
        }
        assert expected == set(gm.keys())

    def test_deterministic(self):
        runs = _make_runs()
        r1 = run_system_diagnostics(runs)
        r2 = run_system_diagnostics(runs)
        assert r1["global_metrics"] == r2["global_metrics"]

    def test_no_mutation(self):
        runs = _make_runs()
        runs_copy = copy.deepcopy(runs)
        run_system_diagnostics(runs)
        assert runs == runs_copy

    def test_stability_is_float(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["system_stability"], float)
        assert 0.0 <= gm["system_stability"] <= 1.0

    def test_volatility_is_float(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["volatility_score"], float)
        assert gm["volatility_score"] >= 0.0

    def test_convergence_rate_bounded(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["convergence_rate"], float)
        assert 0.0 <= gm["convergence_rate"] <= 1.0

    def test_topology_is_string(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["topology_type"], str)

    def test_dominant_strategy_is_string(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["dominant_strategy"], str)
        assert gm["dominant_strategy"] != ""

    def test_best_policy_is_string(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["best_policy"], str)

    def test_best_archetype_is_string(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        gm = result["global_metrics"]
        assert isinstance(gm["best_archetype"], str)

    def test_empty_runs(self):
        result = run_system_diagnostics([])
        gm = result["global_metrics"]
        assert gm["dominant_strategy"] == "unknown"
        assert gm["system_stability"] == 0.0
        assert gm["volatility_score"] == 0.0
        # best_policy may come from default meta-control policies.
        assert isinstance(gm["best_policy"], str)


# ---------------------------------------------------------------------------
# format_system_diagnostics
# ---------------------------------------------------------------------------


class TestFormatSystemDiagnostics:
    """Tests for ``format_system_diagnostics``."""

    def test_contains_header(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "=== System Diagnostics ===" in text

    def test_contains_stability(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Stability:" in text

    def test_contains_volatility(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Volatility:" in text

    def test_contains_topology(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Topology:" in text

    def test_contains_best_policy(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Best Policy:" in text

    def test_contains_best_archetype(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Best Archetype:" in text

    def test_contains_dominant_strategy(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Dominant Strategy:" in text

    def test_contains_convergence_rate(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        text = format_system_diagnostics(result)
        assert "Convergence Rate:" in text

    def test_deterministic_format(self):
        runs = _make_runs()
        result = run_system_diagnostics(runs)
        t1 = format_system_diagnostics(result)
        t2 = format_system_diagnostics(result)
        assert t1 == t2

    def test_format_with_empty_metrics(self):
        result = {"global_metrics": {}}
        text = format_system_diagnostics(result)
        assert "=== System Diagnostics ===" in text
        assert "Stability: 0.00" in text


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


class TestAdapterIntegration:
    """Tests for strategy_adapter wrapper functions."""

    def test_adapter_matches_direct(self):
        runs = _make_runs()
        direct = run_system_diagnostics(runs)
        adapter = run_system_diagnostics_analysis(runs)
        assert direct["global_metrics"] == adapter["global_metrics"]

    def test_adapter_format(self):
        runs = _make_runs()
        result = run_system_diagnostics_analysis(runs)
        text = format_system_diagnostics_adapter_summary(result)
        assert "=== System Diagnostics ===" in text
