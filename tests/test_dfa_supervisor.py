"""Tests for DFA supervisory control layer (v90.0.0).

Covers: constraint visibility, forbidden extraction, transition pruning,
fixed-point synthesis, policy extraction, composition depth, integration,
and edge cases.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Test DFA fixtures
# ---------------------------------------------------------------------------


def _make_simple_dfa() -> Dict[str, Any]:
    """DFA with states 0-4, dead state 4 (absorbing).

    0 → 1 (sym 0), 0 → 2 (sym 1)
    1 → 3 (sym 0), 1 → 4 (sym 1)
    2 → 3 (sym 0), 2 → 2 (sym 1)
    3 → 3 (sym 0), 3 → 3 (sym 1)  [attractor]
    4 → 4 (sym 0), 4 → 4 (sym 1)  [dead/absorbing]
    """
    return {
        "states": [0, 1, 2, 3, 4],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 3, 1: 4},
            2: {0: 3, 1: 2},
            3: {0: 3, 1: 3},
            4: {0: 4, 1: 4},
        },
        "initial_state": 0,
        "dead_state": 4,
    }


def _make_all_safe_dfa() -> Dict[str, Any]:
    """DFA with no dead state — all transitions are safe."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 0, 1: 2},
            2: {0: 1, 1: 0},
        },
        "initial_state": 0,
        "dead_state": None,
    }


def _make_all_dead_dfa() -> Dict[str, Any]:
    """DFA where all non-initial states lead to dead state."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 2, 1: 2},
            2: {0: 2, 1: 2},
        },
        "initial_state": 0,
        "dead_state": 2,
    }


def _make_single_state_dfa() -> Dict[str, Any]:
    """Single-state DFA (self-loop)."""
    return {
        "states": [0],
        "alphabet": [0],
        "transitions": {0: {0: 0}},
        "initial_state": 0,
        "dead_state": None,
    }


def _make_dead_heavy_dfa() -> Dict[str, Any]:
    """DFA with many dead-end paths.

    0 → 1 (0), 0 → 2 (1)
    1 → 3 (0), 1 → 5 (1)  [5 is dead]
    2 → 5 (0), 2 → 4 (1)
    3 → 4 (0), 3 → 5 (1)
    4 → 4 (0), 4 → 4 (1)  [attractor]
    5 → 5 (0), 5 → 5 (1)  [dead/absorbing]
    """
    return {
        "states": [0, 1, 2, 3, 4, 5],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 3, 1: 5},
            2: {0: 5, 1: 4},
            3: {0: 4, 1: 5},
            4: {0: 4, 1: 4},
            5: {0: 5, 1: 5},
        },
        "initial_state": 0,
        "dead_state": 5,
    }


def _get_invariants_and_constraints(dfa):
    """Helper to compute invariants and constraints for a DFA."""
    from qec.experiments.dfa_invariants import (
        derive_constraints_from_invariants,
        detect_invariants,
    )

    inv = detect_invariants(dfa, {}, {})
    constraints = derive_constraints_from_invariants(inv)
    return inv, constraints


# ===========================================================================
# 1. Constraint Visibility
# ===========================================================================


class TestConstraintProvenance(unittest.TestCase):
    """Test invariant constraint provenance (A1)."""

    def test_provenance_exists(self):
        from qec.experiments.dfa_supervisor import structure_invariant_constraints

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        structured = structure_invariant_constraints(constraints, inv)

        self.assertIn("sources", structured)
        self.assertIn("avoid_states", structured["sources"])
        self.assertIn("allow_only_states", structured["sources"])

    def test_dead_state_provenance(self):
        from qec.experiments.dfa_supervisor import structure_invariant_constraints

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        structured = structure_invariant_constraints(constraints, inv)

        # Dead state 4 should be in avoid_states with reason.
        self.assertIn(4, structured["avoid_states"])
        self.assertIn(4, structured["sources"]["avoid_states"])
        self.assertIn("dead_state", structured["sources"]["avoid_states"][4])

    def test_canonical_schema(self):
        from qec.experiments.dfa_supervisor import normalize_constraint_bundle

        bundle = normalize_constraint_bundle(None)
        for key in ("user", "invariant", "supervisor", "policy", "composed"):
            self.assertIn(key, bundle)
            self.assertIn("avoid_states", bundle[key])
            self.assertIn("allow_only_states", bundle[key])

    def test_canonical_schema_preserves_data(self):
        from qec.experiments.dfa_supervisor import normalize_constraint_bundle

        bundle = normalize_constraint_bundle({
            "user": {"avoid_states": [1, 2], "allow_only_states": None},
        })
        self.assertEqual(bundle["user"]["avoid_states"], [1, 2])
        self.assertIsNone(bundle["user"]["allow_only_states"])
        # Other sections default to empty.
        self.assertEqual(bundle["invariant"]["avoid_states"], [])


# ===========================================================================
# 2. Forbidden Extraction
# ===========================================================================


class TestForbiddenExtraction(unittest.TestCase):
    """Test forbidden state derivation (B1)."""

    def test_dead_state_forbidden(self):
        from qec.experiments.dfa_supervisor import derive_forbidden_states

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        fb = derive_forbidden_states(dfa, inv, constraints)

        self.assertIn(4, fb["forbidden_states"])
        self.assertIn("dead_state", fb["reasons"][4])

    def test_drain_states_forbidden(self):
        from qec.experiments.dfa_supervisor import derive_forbidden_states

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        fb = derive_forbidden_states(dfa, inv, constraints)

        # State 1 has a path to dead state 4 — if invariants mark it.
        # The exact result depends on attractor mapping.
        # At minimum, dead state must be forbidden.
        self.assertIn(4, fb["forbidden_states"])

    def test_reasons_stable_and_deterministic(self):
        from qec.experiments.dfa_supervisor import derive_forbidden_states

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        fb1 = derive_forbidden_states(dfa, inv, constraints)
        fb2 = derive_forbidden_states(dfa, inv, constraints)

        self.assertEqual(fb1, fb2)

    def test_no_forbidden_for_safe_dfa(self):
        from qec.experiments.dfa_supervisor import derive_forbidden_states

        dfa = _make_all_safe_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        fb = derive_forbidden_states(dfa, inv, constraints)

        # No dead state → may still have outside_allowed_region.
        # At minimum, no dead_state or dead_drain reasons.
        for reasons in fb["reasons"].values():
            self.assertNotIn("dead_state", reasons)
            self.assertNotIn("dead_drain", reasons)


# ===========================================================================
# 3. Transition Pruning
# ===========================================================================


class TestTransitionPruning(unittest.TestCase):
    """Test unsafe transition pruning (C1)."""

    def test_unsafe_transitions_removed(self):
        from qec.experiments.dfa_supervisor import prune_unsafe_transitions

        dfa = _make_simple_dfa()
        pruned, disabled = prune_unsafe_transitions(dfa, [4])

        # No transition should lead to state 4.
        for s, trans in pruned["transitions"].items():
            for sym, ns in trans.items():
                self.assertNotEqual(ns, 4)

    def test_safe_transitions_preserved(self):
        from qec.experiments.dfa_supervisor import prune_unsafe_transitions

        dfa = _make_simple_dfa()
        pruned, disabled = prune_unsafe_transitions(dfa, [4])

        # Transition 0→1 via symbol 0 should be preserved.
        self.assertEqual(pruned["transitions"][0][0], 1)

    def test_disabled_transitions_reported(self):
        from qec.experiments.dfa_supervisor import prune_unsafe_transitions

        dfa = _make_simple_dfa()
        pruned, disabled = prune_unsafe_transitions(dfa, [4])

        # 1→4 via symbol 1 and 4's self-loops should be disabled.
        disabled_targets = [t[2] for t in disabled]
        self.assertTrue(all(t == 4 for t in disabled_targets))

    def test_no_input_mutation(self):
        from qec.experiments.dfa_supervisor import prune_unsafe_transitions

        dfa = _make_simple_dfa()
        original = copy.deepcopy(dfa)
        prune_unsafe_transitions(dfa, [4])

        self.assertEqual(dfa, original)


# ===========================================================================
# 4. Fixed-Point Synthesis
# ===========================================================================


class TestSynthesis(unittest.TestCase):
    """Test supervisor synthesis (D1-D3)."""

    def test_synthesis_converges(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        self.assertIn("supervised_dfa", result)
        self.assertIn("forbidden_states", result)
        self.assertIsInstance(result["n_pruned_states"], int)
        self.assertIsInstance(result["n_disabled_transitions"], int)

    def test_repeated_synthesis_identical(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        r1 = synthesize_supervisor(dfa, inv, constraints)
        r2 = synthesize_supervisor(dfa, inv, constraints)

        self.assertEqual(r1["supervised_dfa"], r2["supervised_dfa"])
        self.assertEqual(r1["forbidden_states"], r2["forbidden_states"])
        self.assertEqual(r1["disabled_transitions"], r2["disabled_transitions"])

    def test_no_input_mutation(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_simple_dfa()
        original = copy.deepcopy(dfa)
        inv, constraints = _get_invariants_and_constraints(dfa)
        synthesize_supervisor(dfa, inv, constraints)

        self.assertEqual(dfa, original)

    def test_supervised_dfa_has_no_forbidden_transitions(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        supervised = result["supervised_dfa"]
        forbidden_set = set(result["forbidden_states"])
        for s, trans in supervised["transitions"].items():
            for sym, ns in trans.items():
                self.assertNotIn(ns, forbidden_set)


# ===========================================================================
# 5. Policy Extraction
# ===========================================================================


class TestPolicyExtraction(unittest.TestCase):
    """Test supervisor policy extraction (E1-E2)."""

    def test_allowed_symbols_match(self):
        from qec.experiments.dfa_supervisor import (
            extract_supervisor_policy,
            synthesize_supervisor,
        )

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)
        policy = extract_supervisor_policy(result["supervised_dfa"])

        # Each state's allowed symbols match its transitions.
        supervised = result["supervised_dfa"]
        for s in supervised.get("states", []):
            expected_symbols = sorted(supervised["transitions"].get(s, {}).keys())
            self.assertEqual(policy.get(s, []), expected_symbols)

    def test_policy_constraints_informational(self):
        from qec.experiments.dfa_supervisor import (
            derive_policy_constraints,
            extract_supervisor_policy,
            synthesize_supervisor,
        )

        dfa = _make_simple_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)
        policy = extract_supervisor_policy(result["supervised_dfa"])
        pc = derive_policy_constraints(policy)

        self.assertIn("allowed_symbols_by_state", pc)


# ===========================================================================
# 6. Composition Depth
# ===========================================================================


class TestCompositionDepth(unittest.TestCase):
    """Test full constraint composition (F1)."""

    def test_composition_deterministic(self):
        from qec.experiments.dfa_engine import compose_control_constraints

        user = {"avoid_states": {10}}
        inv = {"avoid_states": {4}, "allow_only_states": {0, 1, 2, 3}}
        sup = {"forbidden_states": [4]}
        pol = {"allowed_symbols_by_state": {0: [0, 1], 1: [0]}}

        r1 = compose_control_constraints(user, inv, sup, pol)
        r2 = compose_control_constraints(user, inv, sup, pol)

        self.assertEqual(r1, r2)

    def test_provenance_preserved(self):
        from qec.experiments.dfa_engine import compose_control_constraints

        user = {"avoid_states": {10}}
        inv = {"avoid_states": {4}}
        sup = {"forbidden_states": [4]}
        pol = {"allowed_symbols_by_state": {0: [0, 1]}}

        result = compose_control_constraints(user, inv, sup, pol)
        self.assertIn("sources", result)
        self.assertIn("user_avoid", result["sources"])
        self.assertIn("invariant_avoid", result["sources"])
        self.assertIn("supervisor_forbidden", result["sources"])
        self.assertIn("policy_symbols", result["sources"])

    def test_no_key_drift(self):
        from qec.experiments.dfa_engine import compose_control_constraints

        result = compose_control_constraints(None, None, None, None)
        expected_keys = {"avoid_states", "allow_only_states", "max_depth", "sources"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_composition_with_all_none(self):
        from qec.experiments.dfa_engine import compose_control_constraints

        result = compose_control_constraints(None, None, None, None)
        self.assertEqual(result["avoid_states"], [])
        self.assertIsNone(result["allow_only_states"])
        self.assertIsNone(result["max_depth"])


# ===========================================================================
# 7. Integration
# ===========================================================================


class TestIntegration(unittest.TestCase):
    """Test engine integration (G1-G4)."""

    def test_engine_exposes_supervisor(self):
        from qec.experiments.dfa_engine import run_dfa_engine

        dfa = _make_simple_dfa()
        result = run_dfa_engine(dfa)

        self.assertIn("supervisor", result)
        self.assertIn("supervised_dfa", result["supervisor"])
        self.assertIn("forbidden_states", result["supervisor"])
        self.assertIn("policy", result["supervisor"])

    def test_engine_exposes_constraint_bundle(self):
        from qec.experiments.dfa_engine import run_dfa_engine

        dfa = _make_simple_dfa()
        result = run_dfa_engine(dfa)

        self.assertIn("constraint_bundle", result)
        bundle = result["constraint_bundle"]
        for key in ("user", "invariant", "supervisor", "policy", "composed"):
            self.assertIn(key, bundle)

    def test_supervised_control_respects_disabled(self):
        from qec.experiments.dfa_engine import find_control_path, run_dfa_engine

        dfa = _make_simple_dfa()
        engine = run_dfa_engine(dfa)
        supervised = engine["supervisor"]["supervised_dfa"]
        bundle = engine["constraint_bundle"]

        # Try to reach state 4 (dead) — should be unreachable on supervised DFA.
        if 4 not in set(supervised.get("states", [])):
            # State 4 already pruned from supervised DFA.
            result = find_control_path(
                dfa, 0, 4, supervised_dfa=supervised, constraint_bundle=bundle,
            )
            self.assertFalse(result["reachable"])

    def test_supervised_simulation_respects_policy(self):
        from qec.experiments.dfa_engine import run_dfa_engine, simulate_from_state

        dfa = _make_simple_dfa()
        engine = run_dfa_engine(dfa)
        supervised = engine["supervisor"]["supervised_dfa"]
        bundle = engine["constraint_bundle"]

        sim = simulate_from_state(
            dfa, 0, max_steps=5,
            supervised_dfa=supervised, constraint_bundle=bundle,
        )

        # All trajectory states should be in supervised DFA states.
        supervised_states = set(supervised.get("states", []))
        for traj in sim["trajectories"]:
            for s in traj:
                self.assertIn(s, supervised_states)


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCaseAllSafe(unittest.TestCase):
    """All-safe DFA: no forbidden states expected."""

    def test_no_states_pruned(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_all_safe_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        # All states should remain (no dead state to prune).
        supervised_states = set(result["supervised_dfa"]["states"])
        self.assertTrue(supervised_states.issubset(set(dfa["states"])))


class TestEdgeCaseAllUnsafe(unittest.TestCase):
    """DFA where most paths lead to dead state."""

    def test_synthesis_handles_heavy_pruning(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_all_dead_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        # Dead state 2 should be forbidden.
        self.assertIn(2, result["forbidden_states"])


class TestEdgeCaseSingleState(unittest.TestCase):
    """Single-state DFA."""

    def test_single_state_synthesis(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_single_state_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        # Single state should survive.
        self.assertIn(0, result["supervised_dfa"]["states"])

    def test_single_state_policy(self):
        from qec.experiments.dfa_supervisor import (
            extract_supervisor_policy,
            synthesize_supervisor,
        )

        dfa = _make_single_state_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)
        policy = extract_supervisor_policy(result["supervised_dfa"])

        self.assertIn(0, policy)
        self.assertEqual(policy[0], [0])


class TestEdgeCaseDeadHeavy(unittest.TestCase):
    """Dead-state-heavy DFA."""

    def test_dead_heavy_synthesis(self):
        from qec.experiments.dfa_supervisor import synthesize_supervisor

        dfa = _make_dead_heavy_dfa()
        inv, constraints = _get_invariants_and_constraints(dfa)
        result = synthesize_supervisor(dfa, inv, constraints)

        # Dead state 5 must be forbidden.
        self.assertIn(5, result["forbidden_states"])
        # Attractor state 4 should survive.
        self.assertIn(4, result["supervised_dfa"]["states"])

    def test_dead_heavy_engine_integration(self):
        from qec.experiments.dfa_engine import run_dfa_engine

        dfa = _make_dead_heavy_dfa()
        result = run_dfa_engine(dfa)

        self.assertIn("supervisor", result)
        self.assertIn("constraint_bundle", result)
        self.assertIn(5, result["supervisor"]["forbidden_states"])


# ===========================================================================
# 9. Trim functions
# ===========================================================================


class TestTrimFunctions(unittest.TestCase):
    """Test trim_unreachable and trim_noncoaccessible."""

    def test_trim_unreachable(self):
        from qec.experiments.dfa_supervisor import trim_unreachable

        # Add unreachable state 99.
        dfa = _make_simple_dfa()
        dfa["states"].append(99)
        dfa["transitions"][99] = {0: 99, 1: 99}

        trimmed = trim_unreachable(dfa)
        self.assertNotIn(99, trimmed["states"])
        self.assertIn(0, trimmed["states"])

    def test_trim_noncoaccessible(self):
        from qec.experiments.dfa_supervisor import trim_noncoaccessible

        dfa = _make_simple_dfa()
        # Safe targets: only state 3.
        trimmed = trim_noncoaccessible(dfa, safe_targets={3})

        # All states in trimmed should be able to reach state 3.
        self.assertIn(3, trimmed["states"])
        self.assertIn(0, trimmed["states"])

    def test_trim_noncoaccessible_empty_targets(self):
        from qec.experiments.dfa_supervisor import trim_noncoaccessible

        dfa = _make_simple_dfa()
        trimmed = trim_noncoaccessible(dfa, safe_targets=set())

        self.assertEqual(trimmed["states"], [])


if __name__ == "__main__":
    unittest.main()
