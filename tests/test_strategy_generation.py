"""Tests for deterministic strategy generation (v101.6.0).

Covers:
- generate_strategies: count, determinism, ordering, naming
- no-mutation guarantees
- parameter bounds
- integration with scoring/selection pipeline
"""

from __future__ import annotations

import copy

from qec.analysis.strategy_generation import (
    CONF_SCALES,
    DEPTHS,
    EXPECTED_COUNT,
    NEUTRAL_BIAS,
    generate_strategies,
)
from qec.analysis.strategy_adapter import (
    format_generation_summary,
    run_generation_selection_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_strategy():
    return {
        "config": {
            "threshold": 0.3,
            "rounds": 3,
            "confidence_scale": 1.0,
        },
        "metrics": {
            "design_score": 0.7,
            "confidence_efficiency": 0.6,
        },
    }


# ---------------------------------------------------------------------------
# generate_strategies tests
# ---------------------------------------------------------------------------

class TestGenerateStrategies:

    def test_exactly_27_strategies(self):
        """Must return exactly 27 strategies."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        assert len(result) == 27
        assert len(result) == EXPECTED_COUNT

    def test_deterministic_output_order(self):
        """Two calls must return identical results."""
        base = _make_base_strategy()
        r1 = generate_strategies(base)
        r2 = generate_strategies(base)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["name"] == b["name"]
            assert a["config"] == b["config"]
            assert a["origin"] == b["origin"]

    def test_unique_names(self):
        """All strategy names must be unique."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        names = [s["name"] for s in result]
        assert len(names) == len(set(names))

    def test_sorted_by_name(self):
        """Strategies must be sorted by name."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        names = [s["name"] for s in result]
        assert names == sorted(names)

    def test_no_mutation_of_base_strategy(self):
        """Base strategy must not be mutated."""
        base = _make_base_strategy()
        original = copy.deepcopy(base)
        generate_strategies(base)
        assert base == original

    def test_confidence_scale_bounds(self):
        """Confidence scale must be clamped to [0.0, 1.0]."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        for s in result:
            cs = s["config"]["confidence_scale"]
            assert 0.0 <= cs <= 1.0

    def test_depth_values(self):
        """All depths must come from DEPTHS."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        for s in result:
            assert s["config"]["rounds"] in DEPTHS

    def test_bias_values(self):
        """All biases must come from NEUTRAL_BIAS."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        for s in result:
            assert s["config"]["neutral_bias"] in NEUTRAL_BIAS

    def test_origin_tuple_format(self):
        """Origin must be (scale, bias, depth) tuple."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        for s in result:
            origin = s["origin"]
            assert len(origin) == 3
            assert origin[0] in CONF_SCALES
            assert origin[1] in NEUTRAL_BIAS
            assert origin[2] in DEPTHS

    def test_name_format(self):
        """Name must match the expected format."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        for s in result:
            scale, bias, depth = s["origin"]
            expected = f"ternary__conf_{scale}__bias_{bias:+.1f}__depth_{depth}"
            assert s["name"] == expected

    def test_strategies_are_independent_copies(self):
        """Modifying one strategy config must not affect others."""
        base = _make_base_strategy()
        result = generate_strategies(base)
        result[0]["config"]["threshold"] = 999.0
        assert result[1]["config"]["threshold"] != 999.0

    def test_empty_base_config(self):
        """Works with an empty base config."""
        base = {"config": {}}
        result = generate_strategies(base)
        assert len(result) == 27

    def test_missing_config_key(self):
        """Works even if base_strategy has no config key."""
        base = {}
        result = generate_strategies(base)
        assert len(result) == 27


# ---------------------------------------------------------------------------
# Integration pipeline tests
# ---------------------------------------------------------------------------

class TestGenerationSelectionPipeline:

    def test_pipeline_returns_expected_keys(self):
        """Pipeline must return candidates, ranked, selected."""
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(base)
        assert "candidates" in result
        assert "ranked" in result
        assert "selected" in result

    def test_pipeline_27_candidates(self):
        """Pipeline must produce 27 candidates."""
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(base)
        assert len(result["candidates"]) == 27
        assert len(result["ranked"]) == 27

    def test_pipeline_determinism(self):
        """Pipeline must produce identical results on repeated calls."""
        base = _make_base_strategy()
        ts = {"stability": 0.8, "global_trust": 0.6}
        r1 = run_generation_selection_pipeline(base, trust_signals=ts)
        r2 = run_generation_selection_pipeline(base, trust_signals=ts)
        assert r1["selected"]["name"] == r2["selected"]["name"]
        assert r1["selected"]["_score"] == r2["selected"]["_score"]
        for a, b in zip(r1["ranked"], r2["ranked"]):
            assert a["name"] == b["name"]
            assert a["_score"] == b["_score"]

    def test_pipeline_selected_has_score(self):
        """Selected strategy must have a _score."""
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(base)
        assert "_score" in result["selected"]
        assert 0.0 <= result["selected"]["_score"] <= 1.0

    def test_pipeline_ranked_ordering(self):
        """Ranked list must be in descending score order."""
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(base)
        scores = [r["_score"] for r in result["ranked"]]
        assert scores == sorted(scores, reverse=True)

    def test_pipeline_no_mutation(self):
        """Base strategy must not be mutated by pipeline."""
        base = _make_base_strategy()
        original = copy.deepcopy(base)
        run_generation_selection_pipeline(base)
        assert base == original

    def test_format_generation_summary(self):
        """Summary must contain key information."""
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(base)
        summary = format_generation_summary(result)
        assert "Strategy Generation" in summary
        assert "Total candidates: 27" in summary
        assert "Selected:" in summary
        assert "Top 5:" in summary
        assert "Score distribution:" in summary
