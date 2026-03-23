"""Deterministic symbolic dynamics — minimal DFA from basin sequences.

Transforms basin sequences into a deterministic finite automaton (DFA)
that captures the formal language of the system's dynamics: allowed
transitions, forbidden transitions, and cycle structure.

Pure deterministic algorithms only. No randomness, no mutation of inputs.
"""

from typing import Any, Dict, List, Tuple


# -- A1: extract basin sequence --------------------------------------------


def extract_basin_sequence(
    trajectory_states: List[Any],
    basin_mapping: Dict[Any, int],
) -> List[int]:
    """Map each trajectory state to its basin ID."""
    return [basin_mapping[s] for s in trajectory_states]


# -- A2: compress consecutive duplicates -----------------------------------


def compress_sequence(seq: List[int]) -> List[int]:
    """Collapse consecutive duplicate basin IDs.

    [3, 3, 3, 5, 5, 2] → [3, 5, 2]
    """
    if not seq:
        return []
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            result.append(seq[i])
    return result


# -- B1: trie construction ------------------------------------------------


def _new_node(node_id: int) -> Dict[str, Any]:
    return {"id": node_id, "transitions": {}}


def build_trie(sequences: List[List[int]]) -> Dict[str, Any]:
    """Build a prefix tree from basin sequences.

    Nodes are created in deterministic order: sequences are processed
    in input order, symbols within each sequence left-to-right,
    and new children are assigned incrementing IDs.
    """
    root = _new_node(0)
    next_id = 1
    for seq in sequences:
        node = root
        for symbol in seq:
            if symbol not in node["transitions"]:
                node["transitions"][symbol] = _new_node(next_id)
                next_id += 1
            node = node["transitions"][symbol]
    return root


# -- C1: trie → DFA -------------------------------------------------------


def trie_to_dfa(root: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a prefix trie into a canonical DFA.

    States are numbered by creation order (BFS). Alphabet and
    transitions are sorted for canonical output.
    """
    alphabet_set: set = set()
    transitions: Dict[int, Dict[int, int]] = {}
    states: List[int] = []

    # BFS in deterministic order (sorted keys at each level).
    queue = [root]
    visited_ids: set = set()
    while queue:
        node = queue.pop(0)
        nid = node["id"]
        if nid in visited_ids:
            continue
        visited_ids.add(nid)
        states.append(nid)
        trans: Dict[int, int] = {}
        for symbol in sorted(node["transitions"]):
            child = node["transitions"][symbol]
            trans[symbol] = child["id"]
            alphabet_set.add(symbol)
        transitions[nid] = trans
        for symbol in sorted(node["transitions"]):
            queue.append(node["transitions"][symbol])

    alphabet = sorted(alphabet_set)
    return {
        "states": states,
        "alphabet": alphabet,
        "transitions": transitions,
        "initial_state": 0,
    }


# -- D1: rule extraction --------------------------------------------------


def extract_rules(dfa: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """Extract allowed transition rules from DFA.

    Returns list of (state, symbol, next_state) tuples, sorted by
    (state, symbol) for canonical ordering.
    """
    rules: List[Tuple[int, int, int]] = []
    for state in dfa["states"]:
        trans = dfa["transitions"].get(state, {})
        for symbol in sorted(trans):
            rules.append((state, symbol, trans[symbol]))
    return rules


# -- D2: forbidden transitions --------------------------------------------


def extract_forbidden(dfa: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Identify forbidden transitions (state, symbol) pairs.

    For each state, any alphabet symbol absent from its transitions
    is a forbidden transition.
    """
    forbidden: List[Tuple[int, int]] = []
    alphabet = dfa["alphabet"]
    for state in dfa["states"]:
        trans = dfa["transitions"].get(state, {})
        for symbol in alphabet:
            if symbol not in trans:
                forbidden.append((state, symbol))
    return forbidden


# -- E1: pipeline entry ----------------------------------------------------


def run_symbolic_dynamics(
    trajectory_states: Dict[int, List[Any]],
    basin_mapping: Dict[Any, int],
) -> Dict[str, Any]:
    """Full symbolic dynamics pipeline.

    Steps:
      1. Extract basin sequences from trajectories.
      2. Compress consecutive duplicates.
      3. Build prefix trie.
      4. Convert to canonical DFA.
      5. Extract transition rules.

    Returns canonical DFA with metadata.
    """
    # 1–2: extract and compress sequences.
    sequences: List[List[int]] = []
    for tid in sorted(trajectory_states):
        raw = extract_basin_sequence(trajectory_states[tid], basin_mapping)
        sequences.append(compress_sequence(raw))

    # 3: build trie.
    root = build_trie(sequences)

    # 4: convert to DFA.
    dfa = trie_to_dfa(root)

    # 5: extract rules.
    rules = extract_rules(dfa)

    return {
        "alphabet": dfa["alphabet"],
        "states": dfa["states"],
        "transitions": dfa["transitions"],
        "rules": rules,
        "initial_state": dfa["initial_state"],
        "n_states": len(dfa["states"]),
        "n_rules": len(rules),
    }
