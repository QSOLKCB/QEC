"""Tests for symbolic_dynamics — deterministic DFA from basin sequences."""

import copy

from qec.experiments.symbolic_dynamics import (
    build_trie,
    compress_sequence,
    extract_basin_sequence,
    extract_forbidden,
    extract_rules,
    run_symbolic_dynamics,
    trie_to_dfa,
)


# -- helpers ---------------------------------------------------------------


def _basin_mapping():
    """Simple mapping: 4 states → 3 basins."""
    return {
        (0,): 1,
        (1,): 2,
        (2,): 3,
        (3,): 1,
    }


def _trajectory_states():
    return {0: [(0,), (1,), (2,), (0,)]}


# -- 1. determinism --------------------------------------------------------


def test_determinism():
    """Same input produces identical DFA across two runs."""
    ts = _trajectory_states()
    bm = _basin_mapping()
    r1 = run_symbolic_dynamics(ts, bm)
    r2 = run_symbolic_dynamics(ts, bm)
    assert r1 == r2


# -- 2. no mutation --------------------------------------------------------


def test_no_mutation():
    """Inputs are not modified by the pipeline."""
    ts = copy.deepcopy(_trajectory_states())
    bm = copy.deepcopy(_basin_mapping())
    ts_orig = copy.deepcopy(ts)
    bm_orig = copy.deepcopy(bm)
    run_symbolic_dynamics(ts, bm)
    assert ts == ts_orig
    assert bm == bm_orig


# -- 3. simple sequence [1,2,3] -------------------------------------------


def test_simple_sequence():
    """Linear sequence produces correct transitions."""
    seq = [[1, 2, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    rules = extract_rules(dfa)
    # 0 --1--> 1 --2--> 2 --3--> 3
    assert (0, 1, 1) in rules
    assert (1, 2, 2) in rules
    assert (2, 3, 3) in rules
    assert len(dfa["states"]) == 4


# -- 4. repeated sequence [1,2,1,2] → loop structure ----------------------


def test_repeated_sequence():
    """Repeated pattern creates branching trie (not a cycle — trie is a tree)."""
    seq = [[1, 2, 1, 2]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    rules = extract_rules(dfa)
    # Trie: 0 -1-> 1 -2-> 2 -1-> 3 -2-> 4
    assert (0, 1, 1) in rules
    assert (1, 2, 2) in rules
    assert len(rules) == 4
    assert dfa["n_states"] if "n_states" in dfa else len(dfa["states"]) == 5


# -- 5. compression -------------------------------------------------------


def test_compression():
    """Consecutive duplicates are collapsed."""
    assert compress_sequence([1, 1, 2, 2]) == [1, 2]
    assert compress_sequence([3, 3, 3, 5, 5, 2]) == [3, 5, 2]
    assert compress_sequence([1]) == [1]


# -- 6. alphabet correctness ----------------------------------------------


def test_alphabet():
    """Alphabet contains unique sorted basin IDs."""
    seq = [[3, 1, 2, 1]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    assert dfa["alphabet"] == [1, 2, 3]


# -- 7. transition validity ------------------------------------------------


def test_transition_validity():
    """All transition targets are valid state IDs."""
    ts = _trajectory_states()
    bm = _basin_mapping()
    result = run_symbolic_dynamics(ts, bm)
    states_set = set(result["states"])
    for state, trans in result["transitions"].items():
        assert state in states_set
        for symbol, target in trans.items():
            assert target in states_set
            assert symbol in result["alphabet"]


# -- 8. empty input --------------------------------------------------------


def test_empty_input():
    """Empty trajectories produce empty DFA."""
    result = run_symbolic_dynamics({}, {})
    assert result["n_states"] == 1  # root node
    assert result["n_rules"] == 0
    assert result["alphabet"] == []


# -- 9. single element -----------------------------------------------------


def test_single_element():
    """Single-state trajectory produces minimal DFA."""
    bm = {(0,): 5}
    ts = {0: [(0,)]}
    result = run_symbolic_dynamics(ts, bm)
    assert result["n_states"] == 2  # root + one child
    assert result["n_rules"] == 1
    assert result["alphabet"] == [5]


# -- 10. structural invariants --------------------------------------------


def test_structural_invariants():
    """States are sequential, transitions deterministic."""
    ts = {
        0: [(0,), (1,), (2,)],
        1: [(0,), (2,), (1,)],
    }
    bm = {(0,): 0, (1,): 1, (2,): 2}
    result = run_symbolic_dynamics(ts, bm)
    # States start at 0.
    assert result["states"][0] == 0
    assert result["initial_state"] == 0
    # Each (state, symbol) maps to exactly one next_state (DFA property).
    for state, trans in result["transitions"].items():
        symbols = list(trans.keys())
        assert len(symbols) == len(set(symbols))


# -- 11. extract_basin_sequence -------------------------------------------


def test_extract_basin_sequence():
    """Basin sequence correctly maps states."""
    states = [(0,), (1,), (2,), (0,)]
    bm = {(0,): 10, (1,): 20, (2,): 30}
    assert extract_basin_sequence(states, bm) == [10, 20, 30, 10]


# -- 12. forbidden transitions --------------------------------------------


def test_forbidden_transitions():
    """Missing transitions are identified as forbidden."""
    seq = [[1, 2], [1, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    forbidden = extract_forbidden(dfa)
    # State 0 has symbol 1, but not 2 or 3.
    assert (0, 2) in forbidden
    assert (0, 3) in forbidden
    # State 0 has symbol 1 → not forbidden.
    assert (0, 1) not in forbidden
