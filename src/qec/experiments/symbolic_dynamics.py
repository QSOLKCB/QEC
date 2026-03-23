"""Deterministic symbolic dynamics — minimal DFA from basin sequences.

Transforms basin sequences into a deterministic finite automaton (DFA)
that captures the formal language of the system's dynamics: allowed
transitions, forbidden transitions, and cycle structure.

Pure deterministic algorithms only. No randomness, no mutation of inputs.
"""

from typing import Any, Dict, List, Set, Tuple


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


# -- F1: collect states ----------------------------------------------------


def collect_states(dfa: Dict[str, Any]) -> List[int]:
    """Return sorted list of states in the DFA."""
    return sorted(dfa["states"])


# -- F2: make total transition function ------------------------------------


def make_total(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new DFA with a total transition function.

    For each (state, symbol) pair where a transition is missing,
    add a transition to a deterministic dead state. If no missing
    transitions exist, return a copy without a dead state.

    Does not mutate the input DFA.
    """
    states = list(dfa["states"])
    alphabet = list(dfa["alphabet"])
    transitions: Dict[int, Dict[int, int]] = {
        s: dict(dfa["transitions"].get(s, {})) for s in states
    }

    # Check if any transitions are missing.
    needs_dead = False
    for s in states:
        for sym in alphabet:
            if sym not in transitions[s]:
                needs_dead = True
                break
        if needs_dead:
            break

    if needs_dead:
        dead = max(states) + 1 if states else 0
        states.append(dead)
        transitions[dead] = {sym: dead for sym in alphabet}
        for s in states:
            for sym in alphabet:
                if sym not in transitions[s]:
                    transitions[s][sym] = dead

    return {
        "states": states,
        "alphabet": alphabet,
        "transitions": transitions,
        "initial_state": dfa["initial_state"],
    }


# -- F3: Hopcroft DFA minimization ----------------------------------------


def minimize_dfa(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize a DFA using partition refinement (Hopcroft's algorithm).

    Produces a canonical minimal automaton by merging states with
    identical transition signatures. Pure deterministic algorithm.

    Does not mutate the input DFA.
    """
    if not dfa["states"] or not dfa["alphabet"]:
        return {
            "states": list(dfa["states"]),
            "alphabet": list(dfa["alphabet"]),
            "transitions": {
                s: dict(dfa["transitions"].get(s, {}))
                for s in dfa["states"]
            },
            "initial_state": dfa["initial_state"],
            "n_states": len(dfa["states"]),
        }

    # Step 1: make transitions total (Hopcroft requires it).
    total = make_total(dfa)
    states = total["states"]
    alphabet = sorted(total["alphabet"])
    transitions = total["transitions"]

    # Step 2: initial partition — all states in one block.
    partition: List[Set[int]] = [set(states)]

    # Step 3: refine until stable.
    changed = True
    while changed:
        changed = False
        new_partition: List[Set[int]] = []
        for block in partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue
            # Build signature for each state: tuple of partition indices
            # for each symbol transition.
            state_to_block = {}
            for idx, blk in enumerate(partition):
                for s in blk:
                    state_to_block[s] = idx

            groups: Dict[Tuple[int, ...], Set[int]] = {}
            for s in sorted(block):
                sig = tuple(
                    state_to_block[transitions[s][sym]] for sym in alphabet
                )
                if sig not in groups:
                    groups[sig] = set()
                groups[sig].add(s)

            if len(groups) > 1:
                changed = True
            # Sort groups deterministically for canonical ordering.
            for sig in sorted(groups):
                new_partition.append(groups[sig])
        partition = new_partition

    # Step 4: assign new state IDs.
    # Sort partitions by (min original state, block size) for canonical IDs.
    partition.sort(key=lambda blk: (min(blk), len(blk)))

    state_to_new: Dict[int, int] = {}
    for new_id, block in enumerate(partition):
        for s in block:
            state_to_new[s] = new_id

    new_initial = state_to_new[total["initial_state"]]

    # Step 5: build minimized transitions.
    new_transitions: Dict[int, Dict[int, int]] = {}
    new_states: List[int] = []
    for new_id, block in enumerate(partition):
        new_states.append(new_id)
        rep = min(block)  # deterministic representative
        new_transitions[new_id] = {
            sym: state_to_new[transitions[rep][sym]] for sym in alphabet
        }

    return {
        "states": new_states,
        "alphabet": list(alphabet),
        "transitions": new_transitions,
        "initial_state": new_initial,
        "n_states": len(new_states),
    }


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

    # 6: minimize DFA.
    min_dfa = minimize_dfa(dfa)

    return {
        "alphabet": dfa["alphabet"],
        "states": dfa["states"],
        "transitions": dfa["transitions"],
        "rules": rules,
        "initial_state": dfa["initial_state"],
        "n_states": len(dfa["states"]),
        "n_rules": len(rules),
        "dfa": dfa,
        "minimized_dfa": min_dfa,
        "compression_ratio": len(dfa["states"]) / max(1, len(min_dfa["states"])),
    }
