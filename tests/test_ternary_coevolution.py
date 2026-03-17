"""
Tests for deterministic co-evolution of ternary decoder rule populations.

Covers base and extended population evaluation, determinism,
metric structure, and best-rule selection stability.
Tests for the deterministic co-evolution evaluation layer.

Verifies:
- Deterministic ordering of rule_results
- Correct best rule selection
- Reproducibility across runs
- Integration with experiment runner
- dtype correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.decoder.ternary.ternary_rule_variants import (
    RULE_REGISTRY,
    get_extended_rule_registry,
)
from src.qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
from tests.utils import simple_parity_matrix, received_vector


# ===========================================================================
# Base population (RULE_REGISTRY only)
# ===========================================================================

class TestBasePopulation:
    """Tests for evaluate_rule_population with use_extended_rules=False."""

    def test_num_rules_matches_registry(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=False)
        assert result["num_rules_evaluated"] == len(RULE_REGISTRY)

    def test_best_rule_in_registry(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=False)
        assert result["best_decoder_rule"] in RULE_REGISTRY

    def test_population_has_all_base_rules(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=False)
        names = {e["rule_name"] for e in result["decoder_rule_population"]}
        assert names == set(RULE_REGISTRY.keys())

    def test_deterministic(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        r1 = evaluate_rule_population(H, r, use_extended_rules=False)
        r2 = evaluate_rule_population(H, r, use_extended_rules=False)
        assert r1["best_decoder_rule"] == r2["best_decoder_rule"]
        assert r1["decoder_rule_population"] == r2["decoder_rule_population"]


# ===========================================================================
# Extended population (base + mutated)
# ===========================================================================

class TestExtendedPopulation:
    """Tests for evaluate_rule_population with use_extended_rules=True."""

    def test_num_rules_matches_extended_registry(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        ext = get_extended_rule_registry()
        assert result["num_rules_evaluated"] == len(ext)

    def test_best_rule_in_extended_registry(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        ext = get_extended_rule_registry()
        assert result["best_decoder_rule"] in ext

    def test_population_has_all_extended_rules(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        names = {e["rule_name"] for e in result["decoder_rule_population"]}
        ext = get_extended_rule_registry()
        assert names == set(ext.keys())

    def test_more_rules_than_base(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        assert result["num_rules_evaluated"] > len(RULE_REGISTRY)

    def test_deterministic(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        r1 = evaluate_rule_population(H, r, use_extended_rules=True)
        r2 = evaluate_rule_population(H, r, use_extended_rules=True)
        assert r1["best_decoder_rule"] == r2["best_decoder_rule"]
        assert r1["num_rules_evaluated"] == r2["num_rules_evaluated"]
        assert r1["decoder_rule_population"] == r2["decoder_rule_population"]


# ===========================================================================
# Metric structure
# ===========================================================================

class TestPopulationMetricStructure:
    """Tests for per-rule metric dict structure."""

    def test_required_keys(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        for entry in result["decoder_rule_population"]:
            assert "rule_name" in entry
            assert "stability" in entry
            assert "entropy" in entry
            assert "conflict_density" in entry
            assert "trapping_indicator" in entry

    def test_metrics_are_float(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        for entry in result["decoder_rule_population"]:
            assert isinstance(entry["stability"], float)
            assert isinstance(entry["entropy"], float)
            assert isinstance(entry["conflict_density"], float)
            assert isinstance(entry["trapping_indicator"], float)

    def test_metrics_bounded(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        for entry in result["decoder_rule_population"]:
            for key in ("stability", "entropy", "conflict_density", "trapping_indicator"):
                assert 0.0 <= entry[key] <= 1.0, (
                    f"Metric '{key}' for rule '{entry['rule_name']}' out of bounds"
                )

    def test_population_sorted_by_rule_name(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        names = [e["rule_name"] for e in result["decoder_rule_population"]]
        assert names == sorted(names)


# ===========================================================================
# Best-rule selection stability
# ===========================================================================

class TestBestRuleSelection:
    """Tests for deterministic best-rule selection via lexsort."""

    def test_best_rule_has_highest_stability(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        best = result["best_decoder_rule"]
        best_stab = None
        for entry in result["decoder_rule_population"]:
            if entry["rule_name"] == best:
                best_stab = entry["stability"]
                break
        max_stab = max(e["stability"] for e in result["decoder_rule_population"])
        assert best_stab == max_stab

    def test_selection_stable_across_calls(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        results = [
            evaluate_rule_population(H, r, use_extended_rules=True)
            for _ in range(3)
        ]
        bests = [r["best_decoder_rule"] for r in results]
        assert len(set(bests)) == 1
