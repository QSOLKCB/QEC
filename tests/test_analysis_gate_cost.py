"""
Tests for analytical gate-cost estimation utilities (v3.0.1).

All outputs are deterministic and JSON-safe.
"""

import json

import pytest

from analysis.gate_cost import estimate_gate_costs, compare_costs


class TestEstimateGateCosts:
    """Golden tests for deterministic outputs across dimensions."""

    @pytest.mark.parametrize("dim", [2, 3, 5, 7])
    def test_qubit_decomp_v1_deterministic(self, dim):
        r1 = estimate_gate_costs(dim, "qubit_decomp_v1")
        r2 = estimate_gate_costs(dim, "qubit_decomp_v1")
        assert r1 == r2

    @pytest.mark.parametrize("dim", [2, 3, 5, 7])
    def test_native_placeholder_v1_deterministic(self, dim):
        r1 = estimate_gate_costs(dim, "native_placeholder_v1")
        r2 = estimate_gate_costs(dim, "native_placeholder_v1")
        assert r1 == r2

    def test_qubit_decomp_v1_golden_d2(self):
        r = estimate_gate_costs(2, "qubit_decomp_v1")
        assert r["dimension"] == 2
        assert r["qubits_per_qudit"] == 1
        assert r["single_qudit_gate_cost"] == 1   # (2-1)^2 = 1
        assert r["two_qudit_gate_cost"] == 2       # (2-1)^2 * 2 = 2
        assert r["model"] == "qubit_decomp_v1"

    def test_qubit_decomp_v1_golden_d3(self):
        r = estimate_gate_costs(3, "qubit_decomp_v1")
        assert r["dimension"] == 3
        assert r["qubits_per_qudit"] == 2
        assert r["single_qudit_gate_cost"] == 4   # (3-1)^2 = 4
        assert r["two_qudit_gate_cost"] == 8       # (3-1)^2 * 2 = 8

    def test_qubit_decomp_v1_golden_d5(self):
        r = estimate_gate_costs(5, "qubit_decomp_v1")
        assert r["dimension"] == 5
        assert r["qubits_per_qudit"] == 3
        assert r["single_qudit_gate_cost"] == 16   # (5-1)^2 = 16
        assert r["two_qudit_gate_cost"] == 32      # (5-1)^2 * 2 = 32

    def test_qubit_decomp_v1_golden_d7(self):
        r = estimate_gate_costs(7, "qubit_decomp_v1")
        assert r["dimension"] == 7
        assert r["qubits_per_qudit"] == 3
        assert r["single_qudit_gate_cost"] == 36   # (7-1)^2 = 36
        assert r["two_qudit_gate_cost"] == 72      # (7-1)^2 * 2 = 72

    def test_native_placeholder_always_unit_cost(self):
        for dim in [2, 3, 5, 7]:
            r = estimate_gate_costs(dim, "native_placeholder_v1")
            assert r["single_qudit_gate_cost"] == 1
            assert r["two_qudit_gate_cost"] == 1

    def test_json_safe(self):
        r = estimate_gate_costs(3, "qubit_decomp_v1")
        text = json.dumps(r, sort_keys=True, separators=(",", ":"))
        parsed = json.loads(text)
        assert parsed == r

    def test_assumptions_passthrough(self):
        r = estimate_gate_costs(
            3, "qubit_decomp_v1",
            assumptions={"custom_param": 42},
        )
        # Assumptions don't change the placeholder result but shouldn't error.
        assert r["model"] == "qubit_decomp_v1"

    def test_assumptions_reflected_in_output(self):
        assumptions = {"z_param": 99, "a_param": "hello"}
        r = estimate_gate_costs(
            3, "qubit_decomp_v1",
            assumptions=assumptions,
        )
        # Assumptions must appear canonicalized (sorted keys).
        assert r["assumptions"] == {"a_param": "hello", "z_param": 99}

    def test_assumptions_empty_when_none(self):
        r = estimate_gate_costs(2, "qubit_decomp_v1")
        assert r["assumptions"] == {}

    def test_assumptions_empty_when_explicit_empty(self):
        r = estimate_gate_costs(2, "native_placeholder_v1", assumptions={})
        assert r["assumptions"] == {}

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            estimate_gate_costs(2, "nonexistent_model")

    def test_dimension_below_2_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            estimate_gate_costs(1, "qubit_decomp_v1")


class TestCompareCosts:
    """Deterministic comparison helper."""

    def test_compare_d2(self):
        q = estimate_gate_costs(2, "qubit_decomp_v1")
        n = estimate_gate_costs(2, "native_placeholder_v1")
        c = compare_costs(q, n)
        assert c["single_gate_ratio"] == 1.0
        assert c["two_gate_ratio"] == 2.0
        assert c["dimension"] == 2

    def test_compare_d3(self):
        q = estimate_gate_costs(3, "qubit_decomp_v1")
        n = estimate_gate_costs(3, "native_placeholder_v1")
        c = compare_costs(q, n)
        assert c["single_gate_ratio"] == 4.0
        assert c["two_gate_ratio"] == 8.0

    def test_compare_deterministic(self):
        q = estimate_gate_costs(5, "qubit_decomp_v1")
        n = estimate_gate_costs(5, "native_placeholder_v1")
        c1 = compare_costs(q, n)
        c2 = compare_costs(q, n)
        assert c1 == c2

    def test_compare_json_safe(self):
        q = estimate_gate_costs(7, "qubit_decomp_v1")
        n = estimate_gate_costs(7, "native_placeholder_v1")
        c = compare_costs(q, n)
        text = json.dumps(c, sort_keys=True, separators=(",", ":"))
        parsed = json.loads(text)
        assert parsed == c

    def test_compare_serialization_stable(self):
        """Two serializations produce identical bytes."""
        q = estimate_gate_costs(5, "qubit_decomp_v1")
        n = estimate_gate_costs(5, "native_placeholder_v1")
        c = compare_costs(q, n)
        j1 = json.dumps(c, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(c, sort_keys=True, separators=(",", ":"))
        assert j1 == j2
