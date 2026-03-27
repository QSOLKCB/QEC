"""Tests for active probing engine (v105.2.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.active_probing import (
    _NEUTRAL_THRESHOLD,
    _PRESSURE_HIGH,
    _SENSITIVITY_HIGH,
    _STABILITY_HIGH,
    _STABILITY_LOW,
    classify_regions,
    compute_influence_map,
    select_probes,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_state(
    basins=None,
    strategies=None,
    global_metrics=None,
):
    """Build a minimal system state dict."""
    state = {}
    if basins is not None:
        state["basins"] = basins
    if strategies is not None:
        state["strategies"] = strategies
    if global_metrics is not None:
        state["global_metrics"] = global_metrics
    return state


def _make_basin(stability=0.5, depth=0.5, escape_rate=0.1):
    return {"stability": stability, "depth": depth, "escape_rate": escape_rate}


def _make_strategy(score=0.5, volatility=0.1):
    return {"score": score, "volatility": volatility}


# ---------------------------------------------------------------------------
# INFLUENCE MAP TESTS
# ---------------------------------------------------------------------------


class TestComputeInfluenceMap:

    def test_empty_state(self):
        result = compute_influence_map({})
        assert result["nodes"] == {}
        assert result["influence_entropy"] == 0.0
        assert result["summary"]["node_count"] == 0

    def test_single_basin(self):
        state = _make_state(basins={"b0": _make_basin(stability=0.8, escape_rate=0.1)})
        result = compute_influence_map(state)
        assert "b0" in result["nodes"]
        node = result["nodes"]["b0"]
        assert node["source"] == "basin"
        assert 0.0 <= node["influence_score"] <= 1.0
        assert 0.0 <= node["instability_pressure"] <= 1.0
        assert 0.0 <= node["control_sensitivity"] <= 1.0

    def test_multiple_basins(self):
        state = _make_state(basins={
            "b0": _make_basin(stability=0.9, escape_rate=0.0),
            "b1": _make_basin(stability=0.2, escape_rate=0.8),
        })
        result = compute_influence_map(state)
        assert len(result["nodes"]) == 2
        # Stable basin has higher influence
        assert result["nodes"]["b0"]["influence_score"] > result["nodes"]["b1"]["influence_score"]

    def test_strategies_included(self):
        state = _make_state(strategies={"s0": _make_strategy(score=0.7, volatility=0.2)})
        result = compute_influence_map(state)
        assert "strategy_s0" in result["nodes"]
        assert result["nodes"]["strategy_s0"]["source"] == "strategy"

    def test_global_metrics_node(self):
        state = _make_state(global_metrics={"stability": 0.8, "convergence_rate": 0.9})
        result = compute_influence_map(state)
        assert "global" in result["nodes"]
        assert result["nodes"]["global"]["source"] == "global"

    def test_determinism(self):
        state = _make_state(
            basins={"b0": _make_basin(), "b1": _make_basin(stability=0.3)},
            strategies={"s0": _make_strategy()},
            global_metrics={"stability": 0.6, "convergence_rate": 0.7},
        )
        r1 = compute_influence_map(state)
        r2 = compute_influence_map(state)
        assert r1 == r2

    def test_no_mutation(self):
        state = _make_state(basins={"b0": _make_basin()})
        original = copy.deepcopy(state)
        compute_influence_map(state)
        assert state == original

    def test_influence_entropy_positive_for_mixed(self):
        state = _make_state(basins={
            "b0": _make_basin(stability=0.9),
            "b1": _make_basin(stability=0.1),
        })
        result = compute_influence_map(state)
        assert result["influence_entropy"] > 0.0

    def test_summary_statistics(self):
        state = _make_state(basins={
            "b0": _make_basin(stability=0.8, escape_rate=0.1),
            "b1": _make_basin(stability=0.3, escape_rate=0.6),
        })
        result = compute_influence_map(state)
        summary = result["summary"]
        assert summary["node_count"] == 2
        assert 0.0 <= summary["avg_influence"] <= 1.0
        assert 0.0 <= summary["avg_pressure"] <= 1.0
        assert summary["max_pressure_node"] in ("b0", "b1")


# ---------------------------------------------------------------------------
# REGION CLASSIFICATION TESTS
# ---------------------------------------------------------------------------


class TestClassifyRegions:

    def test_empty_map(self):
        result = classify_regions({"nodes": {}})
        assert result["stable_territory"] == []
        assert result["contested_regions"] == []
        assert result["unstable_regions"] == []
        assert result["neutral_regions"] == []
        assert result["contested_region_count"] == 0

    def test_stable_node(self):
        imap = {"nodes": {"n0": {
            "stability": 0.9, "instability_pressure": 0.1,
            "control_sensitivity": 0.1, "influence_score": 0.8,
        }}}
        result = classify_regions(imap)
        assert "n0" in result["stable_territory"]

    def test_unstable_node(self):
        imap = {"nodes": {"n0": {
            "stability": 0.1, "instability_pressure": 0.8,
            "control_sensitivity": 0.3, "influence_score": 0.2,
        }}}
        result = classify_regions(imap)
        assert "n0" in result["unstable_regions"]

    def test_contested_node(self):
        imap = {"nodes": {"n0": {
            "stability": 0.5, "instability_pressure": 0.7,
            "control_sensitivity": 0.8, "influence_score": 0.5,
        }}}
        result = classify_regions(imap)
        assert "n0" in result["contested_regions"]
        assert result["contested_region_count"] == 1

    def test_determinism(self):
        imap = {"nodes": {
            "a": {"stability": 0.9, "instability_pressure": 0.1,
                   "control_sensitivity": 0.1, "influence_score": 0.8},
            "b": {"stability": 0.1, "instability_pressure": 0.8,
                   "control_sensitivity": 0.3, "influence_score": 0.2},
        }}
        r1 = classify_regions(imap)
        r2 = classify_regions(imap)
        assert r1 == r2

    def test_no_mutation(self):
        imap = {"nodes": {"n0": {
            "stability": 0.5, "instability_pressure": 0.5,
            "control_sensitivity": 0.5, "influence_score": 0.5,
        }}}
        original = copy.deepcopy(imap)
        classify_regions(imap)
        assert imap == original


# ---------------------------------------------------------------------------
# PROBE SELECTION TESTS
# ---------------------------------------------------------------------------


class TestSelectProbes:

    def test_empty_state(self):
        result = select_probes({}, {}, k=3)
        assert result == []

    def test_k_zero(self):
        state = _make_state(basins={"b0": _make_basin(stability=0.1)})
        result = select_probes(state, {}, k=0)
        assert result == []

    def test_returns_at_most_k(self):
        state = _make_state(basins={
            f"b{i}": _make_basin(stability=0.2, escape_rate=0.7)
            for i in range(10)
        })
        result = select_probes(state, {}, k=3)
        assert len(result) <= 3

    def test_probe_structure(self):
        state = _make_state(basins={"b0": _make_basin(stability=0.2, escape_rate=0.8)})
        result = select_probes(state, {}, k=1)
        if result:
            probe = result[0]
            assert "target_node" in probe
            assert "probe_type" in probe
            assert "priority" in probe
            assert "reason" in probe

    def test_prioritizes_unstable(self):
        state = _make_state(basins={
            "stable": _make_basin(stability=0.95, escape_rate=0.01),
            "unstable": _make_basin(stability=0.1, escape_rate=0.9),
        })
        result = select_probes(state, {}, k=1)
        if result:
            # Should target unstable node, not stable
            assert result[0]["target_node"] == "unstable"

    def test_determinism(self):
        state = _make_state(basins={
            "b0": _make_basin(stability=0.3),
            "b1": _make_basin(stability=0.8),
        })
        r1 = select_probes(state, {}, k=2)
        r2 = select_probes(state, {}, k=2)
        assert r1 == r2

    def test_no_mutation(self):
        state = _make_state(basins={"b0": _make_basin()})
        registry = {"probe:b0": {"count": 2}}
        state_copy = copy.deepcopy(state)
        registry_copy = copy.deepcopy(registry)
        select_probes(state, registry, k=2)
        assert state == state_copy
        assert registry == registry_copy

    def test_no_duplicate_nodes(self):
        state = _make_state(basins={
            f"b{i}": _make_basin(stability=0.2, escape_rate=0.7)
            for i in range(5)
        })
        result = select_probes(state, {}, k=5)
        targets = [p["target_node"] for p in result]
        assert len(targets) == len(set(targets))
