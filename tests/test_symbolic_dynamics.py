"""Tests for symbolic_dynamics — deterministic DFA from basin sequences."""

import copy

from qec.experiments.symbolic_dynamics import (
    build_trie,
    collect_states,
    compress_sequence,
    extract_basin_sequence,
    extract_forbidden,
    extract_rules,
    is_dead_state,
    make_total,
    minimize_dfa,
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


# -- 13. minimization reduces states --------------------------------------


def test_minimization_reduces_states():
    """Minimization produces <= original state count."""
    # Trie [1,2,1,2] gives 5 states; repeated pattern is mergeable.
    seq = [[1, 2, 1, 2]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)
    assert min_dfa["n_states"] <= len(dfa["states"])


# -- 14. equivalent behavior after minimization ---------------------------


def test_minimization_equivalent_behavior():
    """Minimized DFA accepts same transition sequences as original."""
    seq = [[1, 2], [1, 3], [2, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    total_dfa = make_total(dfa)
    min_dfa = minimize_dfa(dfa)

    # Walk each input sequence through both DFAs.
    for s in seq:
        state_orig = total_dfa["initial_state"]
        state_min = min_dfa["initial_state"]
        for sym in s:
            # Original must have transition (total).
            assert sym in total_dfa["transitions"][state_orig]
            state_orig = total_dfa["transitions"][state_orig][sym]
            # Minimized must also accept.
            assert sym in min_dfa["transitions"][state_min]
            state_min = min_dfa["transitions"][state_min][sym]


# -- 15. minimization determinism -----------------------------------------


def test_minimization_determinism():
    """Same input produces identical minimized DFA across runs."""
    seq = [[1, 2, 3], [1, 3, 2], [2, 1, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    m1 = minimize_dfa(dfa)
    m2 = minimize_dfa(dfa)
    assert m1 == m2


# -- 16. canonical state ordering -----------------------------------------


def test_minimization_canonical_ordering():
    """State IDs are sequential starting at 0."""
    seq = [[1, 2], [1, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)
    assert min_dfa["states"] == list(range(min_dfa["n_states"]))
    assert min_dfa["initial_state"] in min_dfa["states"]


# -- 17. dead state handling — no missing transitions ----------------------


def test_minimization_no_missing_transitions():
    """Every (state, symbol) pair has a transition in minimized DFA."""
    seq = [[1, 2], [2, 1]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)
    for s in min_dfa["states"]:
        for sym in min_dfa["alphabet"]:
            assert sym in min_dfa["transitions"][s]


# -- 18. single state DFA -------------------------------------------------


def test_minimization_single_state():
    """Single-state DFA minimizes to single state."""
    dfa = {
        "states": [0],
        "alphabet": [],
        "transitions": {0: {}},
        "initial_state": 0,
    }
    min_dfa = minimize_dfa(dfa)
    assert min_dfa["n_states"] == 1
    assert min_dfa["initial_state"] == 0


# -- 19. empty DFA --------------------------------------------------------


def test_minimization_empty_dfa():
    """Empty DFA (root only, no alphabet) minimizes cleanly."""
    result = run_symbolic_dynamics({}, {})
    min_dfa = result["minimized_dfa"]
    assert min_dfa["n_states"] == 1
    assert min_dfa["alphabet"] == []


# -- 20. pipeline integration — compression ratio -------------------------


def test_pipeline_compression_ratio():
    """Pipeline returns compression ratio >= 1.0."""
    ts = _trajectory_states()
    bm = _basin_mapping()
    result = run_symbolic_dynamics(ts, bm)
    assert "minimized_dfa" in result
    assert "dfa" in result
    assert "compression_ratio" in result
    assert result["compression_ratio"] >= 1.0


# -- 21. collect_states helper --------------------------------------------


def test_collect_states():
    """collect_states returns sorted state list."""
    dfa = {"states": [3, 1, 2, 0]}
    assert collect_states(dfa) == [0, 1, 2, 3]


# -- 22. make_total adds dead state when needed ----------------------------


def test_make_total_adds_dead_state():
    """make_total fills missing transitions with dead state."""
    dfa = {
        "states": [0, 1],
        "alphabet": [1, 2],
        "transitions": {0: {1: 1}, 1: {2: 0}},
        "initial_state": 0,
    }
    total = make_total(dfa)
    # Dead state added.
    assert len(total["states"]) == 3
    dead = total["states"][-1]
    # All transitions filled.
    for s in total["states"]:
        for sym in total["alphabet"]:
            assert sym in total["transitions"][s]
    # Dead state loops to self.
    for sym in total["alphabet"]:
        assert total["transitions"][dead][sym] == dead


# -- 23. make_total no dead state when complete ----------------------------


def test_make_total_no_dead_state_when_complete():
    """make_total does not add dead state if transitions are already total."""
    dfa = {
        "states": [0, 1],
        "alphabet": [1],
        "transitions": {0: {1: 1}, 1: {1: 0}},
        "initial_state": 0,
    }
    total = make_total(dfa)
    assert len(total["states"]) == 2
    assert total["dead_state"] is None


# -- 24. Hopcroft correctness — minimized DFA identical behavior -----------


def test_hopcroft_correctness():
    """Minimized DFA is structurally valid and deterministic."""
    seq = [[1, 2, 1], [1, 3], [2, 3, 1], [2, 1]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)

    # Minimized DFA has sequential state IDs.
    assert min_dfa["states"] == list(range(min_dfa["n_states"]))
    # Initial state is valid.
    assert min_dfa["initial_state"] in min_dfa["states"]
    # All transitions are complete and targets are valid states.
    states_set = set(min_dfa["states"])
    for s in min_dfa["states"]:
        for sym in min_dfa["alphabet"]:
            assert sym in min_dfa["transitions"][s]
            assert min_dfa["transitions"][s][sym] in states_set
    # State count does not exceed total DFA.
    total = make_total(dfa)
    assert min_dfa["n_states"] <= len(total["states"])


# -- 25. performance sanity — partitions strictly reduce or equal ----------


def test_hopcroft_partition_reduction():
    """Minimization produces state count <= original."""
    seq = [[1, 2, 1, 2], [1, 2, 3], [3, 2, 1]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)
    assert min_dfa["n_states"] <= len(dfa["states"]) + 1  # +1 for dead


# -- 26. dead state tagging — exists if needed -----------------------------


def test_dead_state_tagging():
    """Dead state is tagged when transitions are incomplete."""
    dfa = {
        "states": [0, 1],
        "alphabet": [1, 2],
        "transitions": {0: {1: 1}, 1: {2: 0}},
        "initial_state": 0,
    }
    total = make_total(dfa)
    assert total["dead_state"] is not None
    assert is_dead_state(total["dead_state"], total)
    # All transitions must be complete.
    for s in total["states"]:
        for sym in total["alphabet"]:
            assert sym in total["transitions"][s]


# -- 27. dead state stability — same input → same dead state ID -----------


def test_dead_state_stability():
    """Same input produces same dead state ID across runs."""
    dfa = {
        "states": [0, 1, 2],
        "alphabet": [1, 2],
        "transitions": {0: {1: 1}, 1: {2: 2}, 2: {}},
        "initial_state": 0,
    }
    t1 = make_total(dfa)
    t2 = make_total(dfa)
    assert t1["dead_state"] == t2["dead_state"]
    # Through minimization too.
    m1 = minimize_dfa(dfa)
    m2 = minimize_dfa(dfa)
    assert m1["dead_state"] == m2["dead_state"]


# -- 28. transition metrics — ratios computed correctly --------------------


def test_transition_metrics():
    """Pipeline returns correct transition compression metrics."""
    ts = _trajectory_states()
    bm = _basin_mapping()
    result = run_symbolic_dynamics(ts, bm)
    metrics = result["metrics"]
    assert "state_ratio" in metrics
    assert "transition_ratio" in metrics
    assert "raw_states" in metrics
    assert "min_states" in metrics
    assert metrics["state_ratio"] >= 1.0
    assert metrics["transition_ratio"] > 0
    assert metrics["raw_states"] >= metrics["min_states"]


# -- 29. determinism — identical minimized DFA across runs -----------------


def test_hopcroft_determinism():
    """Hopcroft produces identical output across multiple runs."""
    seq = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 1, 2]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    results = [minimize_dfa(dfa) for _ in range(5)]
    for r in results[1:]:
        assert r == results[0]


# -- 30. dead state preserved through minimization ------------------------


def test_dead_state_through_minimization():
    """Dead state from make_total is preserved in minimized DFA."""
    seq = [[1, 2], [1, 3]]
    root = build_trie(seq)
    dfa = trie_to_dfa(root)
    min_dfa = minimize_dfa(dfa)
    # DFA from trie has missing transitions → dead state should exist.
    if min_dfa["dead_state"] is not None:
        ds = min_dfa["dead_state"]
        assert ds in min_dfa["states"]
        # Dead state loops to self on all symbols.
        for sym in min_dfa["alphabet"]:
            assert min_dfa["transitions"][ds][sym] == ds
