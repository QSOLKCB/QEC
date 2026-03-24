"""
Tests for deterministic decoder rule mutation framework.

Covers mutated rule outputs, registry merge ordering, no base rule
overwriting, extended evaluation reproducibility, and dtype correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.decoder.ternary.ternary_rule_mutations import (
    flip_zero_bias_rule,
    conservative_rule,
    inverted_majority_rule,
    generate_mutated_rules,
)
from qec.decoder.ternary.ternary_rule_variants import (
    RULE_REGISTRY,
    get_extended_rule_registry,
)
from qec.decoder.ternary.ternary_rule_evaluator import (
    run_decoder_with_rule,
    evaluate_decoder_rule,
)
from tests.testing_utils import simple_parity_matrix, received_vector


# ===========================================================================
# Test received_vector utility
# ===========================================================================

def test_received_vector_length_and_determinism():
    v3 = received_vector(3)
    v5 = received_vector(5)
    v7 = received_vector(7)
    assert len(v3) == 3
    assert len(v5) == 5
    assert len(v7) == 7
    # Deterministic prefix property
    assert np.all(v5[:3] == v3)
    assert np.all(v7[:5] == v5)


# ===========================================================================
# Test flip_zero_bias_rule
# ===========================================================================

class TestFlipZeroBiasRule:
    """Tests for flip_zero_bias_rule."""

    def test_conflict_returns_zero(self) -> None:
        msgs = np.array([1, -1, 1], dtype=np.int8)
        assert flip_zero_bias_rule(msgs) == np.int8(0)

    def test_tie_returns_positive_one(self) -> None:
        # No conflict (only zeros), sum == 0 -> bias to +1
        msgs = np.array([0, 0, 0], dtype=np.int8)
        assert flip_zero_bias_rule(msgs) == np.int8(1)

    def test_positive_majority(self) -> None:
        msgs = np.array([1, 1, 0], dtype=np.int8)
        assert flip_zero_bias_rule(msgs) == np.int8(1)

    def test_negative_majority(self) -> None:
        msgs = np.array([-1, -1, 0], dtype=np.int8)
        assert flip_zero_bias_rule(msgs) == np.int8(-1)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        result = flip_zero_bias_rule(msgs)
        assert isinstance(result, np.int8)

    def test_deterministic(self) -> None:
        msgs = np.array([1, 0, -1, 0, 1], dtype=np.int8)
        r1 = flip_zero_bias_rule(msgs)
        r2 = flip_zero_bias_rule(msgs)
        assert r1 == r2


# ===========================================================================
# Test conservative_rule
# ===========================================================================

class TestConservativeRule:
    """Tests for conservative_rule."""

    def test_strong_positive(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(1)

    def test_strong_negative(self) -> None:
        msgs = np.array([-1, -1, -1], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(-1)

    def test_weak_agreement_returns_zero(self) -> None:
        # sum = 1, abs(1) < 2 -> 0
        msgs = np.array([1, 0, 0], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(0)

    def test_threshold_boundary(self) -> None:
        # sum = 2, abs(2) >= 2 -> +1
        msgs = np.array([1, 1, 0], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(1)

    def test_negative_threshold_boundary(self) -> None:
        # sum = -2, abs(-2) >= 2 -> -1
        msgs = np.array([-1, -1, 0], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(-1)

    def test_all_zeros(self) -> None:
        msgs = np.array([0, 0, 0], dtype=np.int8)
        assert conservative_rule(msgs) == np.int8(0)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        result = conservative_rule(msgs)
        assert isinstance(result, np.int8)

    def test_deterministic(self) -> None:
        msgs = np.array([1, -1, 1, 1, -1], dtype=np.int8)
        r1 = conservative_rule(msgs)
        r2 = conservative_rule(msgs)
        assert r1 == r2


# ===========================================================================
# Test inverted_majority_rule
# ===========================================================================

class TestInvertedMajorityRule:
    """Tests for inverted_majority_rule."""

    def test_positive_majority_returns_negative(self) -> None:
        msgs = np.array([1, 1, -1], dtype=np.int8)
        assert inverted_majority_rule(msgs) == np.int8(-1)

    def test_negative_majority_returns_positive(self) -> None:
        msgs = np.array([-1, -1, 1], dtype=np.int8)
        assert inverted_majority_rule(msgs) == np.int8(1)

    def test_tie_returns_zero(self) -> None:
        msgs = np.array([1, -1], dtype=np.int8)
        assert inverted_majority_rule(msgs) == np.int8(0)

    def test_all_zeros(self) -> None:
        msgs = np.array([0, 0, 0], dtype=np.int8)
        assert inverted_majority_rule(msgs) == np.int8(0)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        result = inverted_majority_rule(msgs)
        assert isinstance(result, np.int8)

    def test_deterministic(self) -> None:
        msgs = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        r1 = inverted_majority_rule(msgs)
        r2 = inverted_majority_rule(msgs)
        assert r1 == r2


# ===========================================================================
# Test generate_mutated_rules
# ===========================================================================

class TestGenerateMutatedRules:
    """Tests for generate_mutated_rules."""

    def test_returns_dict(self) -> None:
        rules = generate_mutated_rules()
        assert isinstance(rules, dict)

    def test_keys_sorted(self) -> None:
        rules = generate_mutated_rules()
        keys = list(rules.keys())
        assert keys == sorted(keys)

    def test_expected_keys(self) -> None:
        rules = generate_mutated_rules()
        expected = {"conservative", "flip_zero_bias", "inverted_majority"}
        assert set(rules.keys()) == expected

    def test_all_callable(self) -> None:
        rules = generate_mutated_rules()
        msgs = np.array([1, -1, 1], dtype=np.int8)
        for name, fn in rules.items():
            result = fn(msgs)
            assert isinstance(result, np.int8), f"Rule '{name}' did not return np.int8"

    def test_deterministic(self) -> None:
        r1 = generate_mutated_rules()
        r2 = generate_mutated_rules()
        assert list(r1.keys()) == list(r2.keys())


# ===========================================================================
# Test get_extended_rule_registry
# ===========================================================================

class TestExtendedRuleRegistry:
    """Tests for get_extended_rule_registry."""

    def test_keys_sorted(self) -> None:
        ext = get_extended_rule_registry()
        keys = list(ext.keys())
        assert keys == sorted(keys)

    def test_contains_base_rules(self) -> None:
        ext = get_extended_rule_registry()
        for name in RULE_REGISTRY:
            assert name in ext

    def test_contains_mutated_rules(self) -> None:
        ext = get_extended_rule_registry()
        mutated = generate_mutated_rules()
        for name in mutated:
            assert name in ext

    def test_no_overwrite_of_base_rules(self) -> None:
        """Mutated rules must not overwrite base RULE_REGISTRY entries."""
        ext = get_extended_rule_registry()
        for name, fn in RULE_REGISTRY.items():
            assert ext[name] is fn, f"Base rule '{name}' was overwritten"

    def test_no_rule_name_collisions(self) -> None:
        """Base and mutated rule name sets must be disjoint."""
        base = set(RULE_REGISTRY.keys())
        mutated = set(generate_mutated_rules().keys())
        assert base.isdisjoint(mutated)

    def test_base_registry_unchanged(self) -> None:
        """Calling get_extended_rule_registry must not modify RULE_REGISTRY."""
        keys_before = list(RULE_REGISTRY.keys())
        _ = get_extended_rule_registry()
        keys_after = list(RULE_REGISTRY.keys())
        assert keys_before == keys_after

    def test_extended_has_more_rules(self) -> None:
        ext = get_extended_rule_registry()
        assert len(ext) > len(RULE_REGISTRY)

    def test_merge_ordering_stable(self) -> None:
        ext1 = get_extended_rule_registry()
        ext2 = get_extended_rule_registry()
        assert list(ext1.keys()) == list(ext2.keys())

    def test_all_callable(self) -> None:
        ext = get_extended_rule_registry()
        msgs = np.array([1, -1, 1], dtype=np.int8)
        for name, fn in ext.items():
            result = fn(msgs)
            assert isinstance(result, np.int8), f"Rule '{name}' did not return np.int8"


# ===========================================================================
# Test extended evaluation is reproducible
# ===========================================================================

class TestExtendedEvaluation:
    """Tests for evaluating mutated rules through the evaluator."""

    def test_mutated_rules_run_in_evaluator(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        mutated = generate_mutated_rules()
        for name in mutated:
            result = run_decoder_with_rule(H, r, name)
            assert result["final_messages"].dtype == np.int8
            assert isinstance(result["iterations"], (int, np.integer))

    def test_mutated_rules_evaluate_metrics(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        mutated = generate_mutated_rules()
        for name in mutated:
            metrics = evaluate_decoder_rule(H, r, name)
            assert "stability" in metrics
            assert "entropy" in metrics
            assert "conflict_density" in metrics
            assert "trapping_indicator" in metrics

    def test_mutated_rule_metrics_dtype(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        metrics = evaluate_decoder_rule(H, r, "flip_zero_bias")
        for key, val in metrics.items():
            assert isinstance(val, np.float64), f"Metric '{key}' is not np.float64"

    def test_extended_evaluation_deterministic(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        m1 = evaluate_decoder_rule(H, r, "inverted_majority")
        m2 = evaluate_decoder_rule(H, r, "inverted_majority")
        for key in m1:
            assert m1[key] == m2[key], f"Metric '{key}' not deterministic"

    def test_metrics_bounded(self) -> None:
        H = simple_parity_matrix()
        r = received_vector()
        for name in generate_mutated_rules():
            metrics = evaluate_decoder_rule(H, r, name)
            for key, val in metrics.items():
                assert 0.0 <= float(val) <= 1.0, (
                    f"Metric '{key}' for rule '{name}' out of bounds: {val}"
                )


# ===========================================================================
# Test co-evolution
# ===========================================================================

class TestCoevolution:
    """Tests for evaluate_rule_population."""

    def test_base_population(self) -> None:
        from qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=False)
        assert result["num_rules_evaluated"] == len(RULE_REGISTRY)
        assert result["best_decoder_rule"] in RULE_REGISTRY

    def test_extended_population(self) -> None:
        from qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        ext = get_extended_rule_registry()
        assert result["num_rules_evaluated"] == len(ext)
        assert result["best_decoder_rule"] in ext

    def test_population_deterministic(self) -> None:
        from qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
        H = simple_parity_matrix()
        r = received_vector()
        r1 = evaluate_rule_population(H, r, use_extended_rules=True)
        r2 = evaluate_rule_population(H, r, use_extended_rules=True)
        assert r1["best_decoder_rule"] == r2["best_decoder_rule"]
        assert r1["num_rules_evaluated"] == r2["num_rules_evaluated"]
        assert r1["decoder_rule_population"] == r2["decoder_rule_population"]

    def test_population_metrics_structure(self) -> None:
        from qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
        H = simple_parity_matrix()
        r = received_vector()
        result = evaluate_rule_population(H, r, use_extended_rules=True)
        for entry in result["decoder_rule_population"]:
            assert "rule_name" in entry
            assert "stability" in entry
            assert "entropy" in entry
            assert "conflict_density" in entry
            assert "trapping_indicator" in entry


# ===========================================================================
# Test API wrappers
# ===========================================================================

class TestMutationAPIWrappers:
    """Tests for analysis API wrappers for mutation functions."""

    def test_get_extended_rule_registry_wrapper(self) -> None:
        from qec.analysis.api import get_extended_rule_registry as api_ext
        ext = api_ext()
        assert isinstance(ext, dict)
        keys = list(ext.keys())
        assert keys == sorted(keys)
        assert len(ext) > len(RULE_REGISTRY)

    def test_generate_mutated_rules_wrapper(self) -> None:
        from qec.analysis.api import generate_mutated_rules as api_mut
        rules = api_mut()
        assert isinstance(rules, dict)
        expected = {"conservative", "flip_zero_bias", "inverted_majority"}
        assert set(rules.keys()) == expected
