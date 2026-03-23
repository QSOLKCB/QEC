"""Deterministic symbolic dynamics — minimal DFA from basin sequences.

Transforms basin sequences into a deterministic finite automaton (DFA)
that captures the formal language of the system's dynamics: allowed
transitions, forbidden transitions, and cycle structure.

Pure deterministic algorithms only. No randomness, no mutation of inputs.
"""

from typing import Any, Dict, List, Set, Tuple

from qec.experiments.dfa_analysis import analyze_dfa
from qec.experiments.dfa_engine import run_dfa_engine


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

    The dead state ID (if created) is stored under ``"dead_state"``.

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

    dead_state = None
    if needs_dead:
        dead_state = max(states) + 1 if states else 0
        states.append(dead_state)
        transitions[dead_state] = {sym: dead_state for sym in alphabet}
        for s in states:
            for sym in alphabet:
                if sym not in transitions[s]:
                    transitions[s][sym] = dead_state

    return {
        "states": states,
        "alphabet": alphabet,
        "transitions": transitions,
        "initial_state": dfa["initial_state"],
        "dead_state": dead_state,
    }


# -- F3: Hopcroft DFA minimization ----------------------------------------


def _build_inverse(
    states: List[int],
    alphabet: List[int],
    transitions: Dict[int, Dict[int, int]],
) -> Dict[int, Dict[int, Set[int]]]:
    """Precompute inverse transition map: inv[a][t] = {states reaching t via a}."""
    inv: Dict[int, Dict[int, Set[int]]] = {a: {} for a in alphabet}
    for s in states:
        for a in alphabet:
            t = transitions[s][a]
            inv[a].setdefault(t, set()).add(s)
    return inv


def minimize_dfa(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize a DFA using Hopcroft's worklist algorithm.

    True Hopcroft refinement with inverse transition map and worklist.
    Produces a canonical minimal automaton. Pure deterministic algorithm.

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
            "dead_state": None,
        }

    # Step 1: make transitions total (Hopcroft requires it).
    total = make_total(dfa)
    states = sorted(total["states"])
    alphabet = sorted(total["alphabet"])
    transitions = total["transitions"]

    # Step 2: precompute inverse transitions.
    inv = _build_inverse(states, alphabet, transitions)

    # Step 3: initial partition — all states in one block (no accept states).
    initial_block = frozenset(states)
    # partition stored as list for deterministic indexing
    partition: List[Set[int]] = [set(states)]
    # Map each state to its block index.
    state_to_block: Dict[int, int] = {s: 0 for s in states}

    # Step 4: worklist — set of block indices to process as splitters.
    worklist: List[int] = [0]
    worklist_set: Set[int] = {0}

    # Step 5: Hopcroft refinement loop.
    while worklist:
        # Pop from worklist (deterministic: always pop smallest index).
        worklist.sort()
        a_idx = worklist.pop(0)
        worklist_set.discard(a_idx)

        if a_idx >= len(partition):
            continue
        splitter = partition[a_idx]

        for sym in alphabet:
            # X = states that transition via sym into splitter.
            x: Set[int] = set()
            for t in splitter:
                if t in inv[sym]:
                    x.update(inv[sym][t])

            if not x:
                continue

            # Find all blocks that intersect with X and may split.
            blocks_to_check = sorted(set(state_to_block[s] for s in x))

            for b_idx in blocks_to_check:
                block_y = partition[b_idx]
                intersect = block_y & x
                diff = block_y - x

                if not intersect or not diff:
                    continue

                # Split block_y into intersect and diff.
                partition[b_idx] = intersect
                new_idx = len(partition)
                partition.append(diff)

                # Update state_to_block mapping.
                for s in intersect:
                    state_to_block[s] = b_idx
                for s in diff:
                    state_to_block[s] = new_idx

                # Classic Hopcroft rule: add smaller split to worklist.
                if b_idx in worklist_set:
                    # Both halves need to be in worklist.
                    worklist.append(new_idx)
                    worklist_set.add(new_idx)
                else:
                    # Add the smaller half.
                    if len(intersect) <= len(diff):
                        worklist.append(b_idx)
                        worklist_set.add(b_idx)
                    else:
                        worklist.append(new_idx)
                        worklist_set.add(new_idx)

    # Step 6: filter out empty blocks and canonicalize.
    final_partition = [blk for blk in partition if blk]
    final_partition.sort(key=lambda blk: (min(blk), len(blk)))

    state_to_new: Dict[int, int] = {}
    for new_id, block in enumerate(final_partition):
        for s in block:
            state_to_new[s] = new_id

    new_initial = state_to_new[total["initial_state"]]

    # Step 7: build minimized transitions.
    new_transitions: Dict[int, Dict[int, int]] = {}
    new_states: List[int] = []
    for new_id, block in enumerate(final_partition):
        new_states.append(new_id)
        rep = min(block)  # deterministic representative
        new_transitions[new_id] = {
            sym: state_to_new[transitions[rep][sym]] for sym in alphabet
        }

    # Step 8: map dead state through minimization.
    raw_dead = total.get("dead_state")
    min_dead = state_to_new[raw_dead] if raw_dead is not None else None

    return {
        "states": new_states,
        "alphabet": list(alphabet),
        "transitions": new_transitions,
        "initial_state": new_initial,
        "n_states": len(new_states),
        "dead_state": min_dead,
    }


# -- B4: dead state helper -------------------------------------------------


def is_dead_state(state: int, dfa: Dict[str, Any]) -> bool:
    """Check whether *state* is the dead (sink) state of *dfa*."""
    return state == dfa.get("dead_state")


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

    # 7: compute compression metrics.
    n_raw_states = len(dfa["states"])
    n_min_states = len(min_dfa["states"])
    n_raw_transitions = sum(len(v) for v in dfa["transitions"].values())
    n_min_transitions = sum(len(v) for v in min_dfa["transitions"].values())
    state_ratio = n_raw_states / max(1, n_min_states)
    transition_ratio = n_raw_transitions / max(1, n_min_transitions)

    # 8: structural analysis of minimized DFA.
    dfa_analysis = analyze_dfa(min_dfa)

    # 9: DFA engine — prediction, validation, control.
    dfa_engine = run_dfa_engine(min_dfa)

    return {
        "alphabet": dfa["alphabet"],
        "states": dfa["states"],
        "transitions": dfa["transitions"],
        "rules": rules,
        "initial_state": dfa["initial_state"],
        "n_states": n_raw_states,
        "n_rules": len(rules),
        "dfa": dfa,
        "minimized_dfa": min_dfa,
        "compression_ratio": state_ratio,
        "metrics": {
            "state_ratio": state_ratio,
            "transition_ratio": transition_ratio,
            "raw_states": n_raw_states,
            "min_states": n_min_states,
        },
        "dfa_analysis": dfa_analysis,
        "dfa_engine": dfa_engine,
    }
