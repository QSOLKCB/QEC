"""Tests for dfa_benchmark — multi-DFA benchmark suite (v92.3.0).

Covers:
  1. DFA generators produce deterministic output
  2. Generators create correct number of states
  3. Benchmark runs without crash
  4. Results identical across runs (determinism)
  5. Efficiency metrics in valid range [0,1]
  6. Summary has stable ordering
  7. No mutation of DFA inputs
  8. Chain DFA terminal state is absorbing
  9. Cycle DFA loops correctly
  10. Branching DFA has two outgoing edges from state 0
  11. Two-basin DFA has two attractors
  12. Dead-state DFA has absorbing state
  13. Alignment output is well-formed
  14. print_alignment produces readable output
  15. print_changes_only filters correctly
"""

import copy

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec.experiments.dfa_benchmark import (
    DFA_REGISTRY,
    DFA_SIZES,
    MODES,
    build_branching_dfa,
    build_chain_dfa,
    build_cycle_dfa,
    build_dead_state_dfa,
    build_two_basin_dfa,
    run_single_mode,
    run_suite,
    summarize,
    print_summary,
)
from qec.experiments.correction_layer import safe_div
from qec.visualization.alignment_plot import print_alignment, print_changes_only


# ---------------------------------------------------------------------------
# 1. DFA generators are deterministic
# ---------------------------------------------------------------------------


class TestDFAGeneratorsDeterministic:
    def test_chain_deterministic(self):
        a = build_chain_dfa(5)
        b = build_chain_dfa(5)
        assert a == b

    def test_cycle_deterministic(self):
        a = build_cycle_dfa(5)
        b = build_cycle_dfa(5)
        assert a == b

    def test_branching_deterministic(self):
        a = build_branching_dfa(5)
        b = build_branching_dfa(5)
        assert a == b

    def test_two_basin_deterministic(self):
        a = build_two_basin_dfa()
        b = build_two_basin_dfa()
        assert a == b

    def test_dead_state_deterministic(self):
        a = build_dead_state_dfa()
        b = build_dead_state_dfa()
        assert a == b


# ---------------------------------------------------------------------------
# 2. Correct number of states
# ---------------------------------------------------------------------------


class TestDFAStateCount:
    def test_chain_states(self):
        dfa = build_chain_dfa(7)
        assert len(dfa["states"]) == 7

    def test_cycle_states(self):
        dfa = build_cycle_dfa(10)
        assert len(dfa["states"]) == 10

    def test_branching_states(self):
        dfa = build_branching_dfa(6)
        assert len(dfa["states"]) == 6

    def test_two_basin_states(self):
        dfa = build_two_basin_dfa()
        assert len(dfa["states"]) == 5

    def test_dead_state_states(self):
        dfa = build_dead_state_dfa()
        assert len(dfa["states"]) == 4


# ---------------------------------------------------------------------------
# 3. Benchmark runs without crash
# ---------------------------------------------------------------------------


class TestBenchmarkRuns:
    def test_chain_none_mode(self):
        dfa = build_chain_dfa(5)
        result = run_single_mode(dfa, "chain", 5, "none", None, False)
        assert "metrics" in result
        assert result["dfa_name"] == "chain"

    def test_cycle_square_mode(self):
        dfa = build_cycle_dfa(5)
        result = run_single_mode(dfa, "cycle", 5, "square", "square", False)
        assert "metrics" in result

    def test_branching_d4_mode(self):
        dfa = build_branching_dfa(5)
        result = run_single_mode(dfa, "branching", 5, "d4", "d4", False)
        assert "metrics" in result

    def test_two_basin_d4_inv(self):
        dfa = build_two_basin_dfa()
        result = run_single_mode(dfa, "two_basin", 5, "d4+inv", "d4", True)
        assert "metrics" in result

    def test_dead_state_runs(self):
        dfa = build_dead_state_dfa()
        result = run_single_mode(
            dfa, "dead_state", 4, "square", "square", False
        )
        assert "metrics" in result


# ---------------------------------------------------------------------------
# 4. Results identical across runs (determinism)
# ---------------------------------------------------------------------------


class TestResultDeterminism:
    def test_chain_determinism(self):
        dfa = build_chain_dfa(5)
        r1 = run_single_mode(dfa, "chain", 5, "d4", "d4", False)
        r2 = run_single_mode(dfa, "chain", 5, "d4", "d4", False)
        assert r1["metrics"] == r2["metrics"]

    def test_cycle_determinism(self):
        dfa = build_cycle_dfa(5)
        r1 = run_single_mode(dfa, "cycle", 5, "square", "square", False)
        r2 = run_single_mode(dfa, "cycle", 5, "square", "square", False)
        assert r1["metrics"] == r2["metrics"]

    def test_alignment_determinism(self):
        dfa = build_chain_dfa(5)
        r1 = run_single_mode(dfa, "chain", 5, "d4", "d4", False)
        r2 = run_single_mode(dfa, "chain", 5, "d4", "d4", False)
        assert r1["alignment"] == r2["alignment"]


# ---------------------------------------------------------------------------
# 5. Efficiency metrics in valid range [0,1]
# ---------------------------------------------------------------------------


class TestEfficiencyRange:
    def _check_range(self, metrics):
        ce = metrics["compression_efficiency"]
        se = metrics["stability_efficiency"]
        assert -1e-9 <= ce <= 1.0 + 1e-9, f"comp_eff out of range: {ce}"
        assert -1e-9 <= se <= 1.0 + 1e-9, f"stab_eff out of range: {se}"

    def test_chain_efficiency_range(self):
        dfa = build_chain_dfa(5)
        for mode_name, mode, use_inv in MODES:
            r = run_single_mode(dfa, "chain", 5, mode_name, mode, use_inv)
            self._check_range(r["metrics"])

    def test_cycle_efficiency_range(self):
        dfa = build_cycle_dfa(5)
        for mode_name, mode, use_inv in MODES:
            r = run_single_mode(dfa, "cycle", 5, mode_name, mode, use_inv)
            self._check_range(r["metrics"])


# ---------------------------------------------------------------------------
# 6. Summary has stable ordering
# ---------------------------------------------------------------------------


class TestSummaryOrdering:
    def test_summary_keys_sorted(self):
        dfa = build_chain_dfa(5)
        results = []
        for mode_name, mode, use_inv in MODES:
            results.append(
                run_single_mode(dfa, "chain", 5, mode_name, mode, use_inv)
            )
        summary = summarize(results)
        keys = list(summary.keys())
        assert keys == sorted(keys, key=lambda k: (k[0], str(k[1])))

    def test_summary_modes_sorted(self):
        dfa = build_chain_dfa(5)
        results = []
        for mode_name, mode, use_inv in MODES:
            results.append(
                run_single_mode(dfa, "chain", 5, mode_name, mode, use_inv)
            )
        summary = summarize(results)
        for key in summary:
            assert list(summary[key].keys()) == sorted(
                summary[key].keys()
            )


# ---------------------------------------------------------------------------
# 7. No mutation of DFA inputs
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_chain_not_mutated(self):
        dfa = build_chain_dfa(5)
        original = copy.deepcopy(dfa)
        run_single_mode(dfa, "chain", 5, "d4", "d4", True)
        assert dfa == original

    def test_cycle_not_mutated(self):
        dfa = build_cycle_dfa(5)
        original = copy.deepcopy(dfa)
        run_single_mode(dfa, "cycle", 5, "square", "square", False)
        assert dfa == original


# ---------------------------------------------------------------------------
# 8. Chain DFA terminal is absorbing
# ---------------------------------------------------------------------------


class TestChainTerminal:
    def test_terminal_self_loop(self):
        dfa = build_chain_dfa(5)
        terminal = 4
        assert dfa["transitions"][terminal][0] == terminal


# ---------------------------------------------------------------------------
# 9. Cycle DFA loops
# ---------------------------------------------------------------------------


class TestCycleLoop:
    def test_cycle_wraps(self):
        dfa = build_cycle_dfa(5)
        assert dfa["transitions"][4][0] == 0


# ---------------------------------------------------------------------------
# 10. Branching DFA has two outgoing edges from state 0
# ---------------------------------------------------------------------------


class TestBranchingStructure:
    def test_two_branches(self):
        dfa = build_branching_dfa(6)
        assert len(dfa["transitions"][0]) == 2
        assert 0 in dfa["transitions"][0]
        assert 1 in dfa["transitions"][0]


# ---------------------------------------------------------------------------
# 11. Two-basin DFA has two attractors
# ---------------------------------------------------------------------------


class TestTwoBasin:
    def test_two_absorbing(self):
        dfa = build_two_basin_dfa()
        absorbing = [
            s
            for s in dfa["states"]
            if dfa["transitions"].get(s, {}).get(0) == s
        ]
        assert len(absorbing) == 2
        assert set(absorbing) == {2, 4}


# ---------------------------------------------------------------------------
# 12. Dead-state DFA has absorbing state
# ---------------------------------------------------------------------------


class TestDeadState:
    def test_absorbing_dead(self):
        dfa = build_dead_state_dfa()
        # State 3 should self-loop on all symbols.
        for sym in dfa["alphabet"]:
            assert dfa["transitions"][3].get(sym) == 3


# ---------------------------------------------------------------------------
# 13. Alignment output is well-formed
# ---------------------------------------------------------------------------


class TestAlignmentStructure:
    def test_alignment_keys(self):
        dfa = build_chain_dfa(5)
        r = run_single_mode(dfa, "chain", 5, "d4", "d4", False)
        for row in r["alignment"]:
            assert "step" in row
            assert "state" in row
            assert "before" in row
            assert "after" in row

    def test_alignment_length(self):
        dfa = build_chain_dfa(5)
        r = run_single_mode(dfa, "chain", 5, "none", None, False)
        # steps = min(5*2, 20) = 10, trajectory has 11 entries
        assert len(r["alignment"]) == 11


# ---------------------------------------------------------------------------
# 14. print_alignment produces readable output
# ---------------------------------------------------------------------------


class TestPrintAlignment:
    def test_output_has_header(self):
        alignment = [
            {"step": 0, "state": 0, "before": [1, 0], "after": [0, 0]},
            {"step": 1, "state": 1, "before": [1, 1], "after": [1, 0]},
        ]
        text = print_alignment(alignment)
        assert "step" in text
        assert "state" in text
        assert "before" in text
        assert "after" in text

    def test_max_steps_truncation(self):
        alignment = [
            {"step": i, "state": i, "before": [0], "after": [0]}
            for i in range(20)
        ]
        text = print_alignment(alignment, max_steps=5)
        assert "more rows" in text


# ---------------------------------------------------------------------------
# 15. print_changes_only filters correctly
# ---------------------------------------------------------------------------


class TestPrintChangesOnly:
    def test_no_changes(self):
        alignment = [
            {"step": 0, "state": 0, "before": [1], "after": [1]},
        ]
        text = print_changes_only(alignment)
        assert "no changes" in text

    def test_shows_changes(self):
        alignment = [
            {"step": 0, "state": 0, "before": [1], "after": [0]},
            {"step": 1, "state": 1, "before": [1], "after": [1]},
        ]
        text = print_changes_only(alignment)
        # Should show step 0 but not step 1.
        lines = text.strip().split("\n")
        data_lines = [l for l in lines if l and not l.startswith("-") and "step" not in l.lower()]
        assert len(data_lines) == 1


# ---------------------------------------------------------------------------
# Additional: safe_div
# ---------------------------------------------------------------------------


class TestSafeDiv:
    def test_normal(self):
        assert safe_div(3.0, 2.0) == 1.5

    def test_zero_divisor(self):
        assert safe_div(5.0, 0) == 0.0

    def test_zero_numerator(self):
        assert safe_div(0.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# Additional: print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_summary_format(self):
        dfa = build_chain_dfa(5)
        results = [
            run_single_mode(dfa, "chain", 5, "none", None, False),
        ]
        summary = summarize(results)
        text = print_summary(summary)
        assert "DFA: chain (n=5)" in text
        assert "comp_eff" in text

    def test_none_size_label(self):
        dfa = build_two_basin_dfa()
        results = [
            run_single_mode(dfa, "two_basin", None, "none", None, False),
        ]
        summary = summarize(results)
        text = print_summary(summary)
        assert "DFA: two_basin (n=NA)" in text


# ---------------------------------------------------------------------------
# 16. All builders accept n parameter
# ---------------------------------------------------------------------------


class TestBuildersAcceptN:
    def test_builders_accept_n(self):
        for name, builder in DFA_REGISTRY.items():
            dfa = builder(5)
            assert "states" in dfa
            assert "transitions" in dfa


# ---------------------------------------------------------------------------
# 17. Summary keys are explicit (dfa_type, n) tuples
# ---------------------------------------------------------------------------


class TestSummaryKeysExplicit:
    def test_summary_keys_explicit(self):
        summary = summarize(run_suite())
        for (dfa_type, n), modes in summary.items():
            assert isinstance(dfa_type, str)
            assert n is None or isinstance(n, int)
            for mode_name, metrics in modes.items():
                assert isinstance(mode_name, str)
                assert "compression_efficiency" in metrics
                assert "stability_efficiency" in metrics


# ---------------------------------------------------------------------------
# 18. None size is handled for fixed-topology DFAs
# ---------------------------------------------------------------------------


class TestNoneSizeHandled:
    def test_none_size_in_results(self):
        results = run_suite()
        assert any(r["n"] is None for r in results)
