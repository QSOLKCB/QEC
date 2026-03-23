"""Multi-DFA benchmark suite with efficiency metrics (v92.3.0).

Generates multiple deterministic DFAs, runs correction modes on each,
computes normalized efficiency metrics, and summarizes results as a
taxonomy of system behaviors.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from qec.experiments.correction_layer import compute_metrics, safe_div
from qec.experiments.dfa_engine import step


# ---------------------------------------------------------------------------
# PART 1 — DETERMINISTIC DFA GENERATORS
# ---------------------------------------------------------------------------


def build_chain_dfa(n: int) -> Dict[str, Any]:
    """Build a linear chain DFA: 0 → 1 → … → n-1 (terminal).

    Single alphabet symbol 0 drives the chain forward.
    State n-1 is absorbing (self-loop).
    """
    states = list(range(n))
    transitions: Dict[int, Dict[int, int]] = {}
    for s in states:
        if s < n - 1:
            transitions[s] = {0: s + 1}
        else:
            transitions[s] = {0: s}  # absorbing terminal
    return {
        "states": states,
        "alphabet": [0],
        "transitions": transitions,
        "initial_state": 0,
    }


def build_cycle_dfa(n: int) -> Dict[str, Any]:
    """Build a cycle DFA: 0 → 1 → … → n-1 → 0.

    Single alphabet symbol 0 drives around the cycle.
    """
    states = list(range(n))
    transitions: Dict[int, Dict[int, int]] = {}
    for s in states:
        transitions[s] = {0: (s + 1) % n}
    return {
        "states": states,
        "alphabet": [0],
        "transitions": transitions,
        "initial_state": 0,
    }


def build_branching_dfa(n: int) -> Dict[str, Any]:
    """Build a branching DFA: state 0 branches, paths merge at n-1.

    Symbols 0 and 1 available.  State 0 branches to 1 (sym 0) and
    n//2 (sym 1).  All other states chain forward on symbol 0.
    State n-1 is absorbing.
    """
    states = list(range(n))
    mid = max(1, n // 2)
    transitions: Dict[int, Dict[int, int]] = {}
    transitions[0] = {0: 1, 1: mid}
    for s in range(1, n):
        if s < n - 1:
            transitions[s] = {0: s + 1}
        else:
            transitions[s] = {0: s}  # absorbing
    return {
        "states": states,
        "alphabet": [0, 1],
        "transitions": transitions,
        "initial_state": 0,
    }


def build_two_basin_dfa() -> Dict[str, Any]:
    """Build a DFA with two attractors (basins).

    States 0-2 form basin A (0→1→2→2), states 3-4 form basin B (3→4→4).
    State 0 branches: sym 0 → 1, sym 1 → 3.
    """
    return {
        "states": [0, 1, 2, 3, 4],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 3},
            1: {0: 2},
            2: {0: 2},
            3: {0: 4},
            4: {0: 4},
        },
        "initial_state": 0,
    }


def build_dead_state_dfa() -> Dict[str, Any]:
    """Build a DFA with an absorbing dead state.

    States 0-3.  State 3 is absorbing (all transitions self-loop).
    State 0 → 1 (sym 0), 0 → 3 (sym 1).
    State 1 → 2 (sym 0).
    State 2 → 3 (sym 0).
    """
    return {
        "states": [0, 1, 2, 3],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 3},
            1: {0: 2},
            2: {0: 3},
            3: {0: 3, 1: 3},
        },
        "initial_state": 0,
    }


# ---------------------------------------------------------------------------
# PART 2 — REGISTRY AND MODES
# ---------------------------------------------------------------------------


DFA_REGISTRY: Dict[str, Callable] = {
    "chain": build_chain_dfa,
    "cycle": build_cycle_dfa,
    "branching": build_branching_dfa,
    "two_basin": build_two_basin_dfa,
    "dead_state": build_dead_state_dfa,
}


# (mode_name, correction_mode, use_invariants)
MODES: List[Tuple[str, Optional[str], bool]] = [
    ("none", None, False),
    ("square", "square", False),
    ("d4", "d4", False),
    ("d4+inv", "d4", True),
]


# ---------------------------------------------------------------------------
# PART 3 — TRAJECTORY AND MEASUREMENT HELPERS
# ---------------------------------------------------------------------------


def _trajectory_from_dfa(
    dfa: Dict[str, Any], start: int, steps: int
) -> List[int]:
    """Generate a deterministic trajectory through a DFA.

    Uses the smallest available symbol at each step.
    """
    alphabet = sorted(dfa.get("alphabet", []))
    trajectory = [start]
    current = start
    for _ in range(steps):
        next_state = None
        for sym in alphabet:
            ns = step(dfa, current, sym)
            if ns is not None:
                next_state = ns
                break
        if next_state is None:
            trajectory.append(current)
        else:
            trajectory.append(next_state)
            current = next_state
    return trajectory


def _build_stabilizer_code(d: int, num_states: int) -> Any:
    """Build a minimal qudit stabilizer code for benchmarking.

    Creates a code with k qudits (where d^k >= num_states) and
    one Z generator on the first qudit.  Deterministic, no randomness.
    """
    from qudit_stabilizer import QuditStabilizerCode

    k = 1
    while d ** k < num_states:
        k += 1
    a = np.zeros(k, dtype=int)
    b = np.zeros(k, dtype=int)
    b[0] = 1  # Z on first qudit
    generators = [(a.copy(), b.copy())]
    return QuditStabilizerCode(d, k, generators)


def _state_to_vec(state_id: int, num_states: int) -> np.ndarray:
    """One-hot basis vector for a DFA state."""
    vec = np.zeros(num_states, dtype=np.float64)
    if 0 <= state_id < num_states:
        vec[state_id] = 1.0
    return vec


def _embed_to_qudit(state_vec: np.ndarray, d: int) -> np.ndarray:
    """Embed a state vector into qudit Hilbert space."""
    num_states = len(state_vec)
    nonzero = np.nonzero(state_vec)[0]
    if len(nonzero) != 1:
        raise ValueError("state_vec must be a one-hot vector")
    state_idx = int(nonzero[0])
    k = 1
    while d ** k < num_states:
        k += 1
    dim = d ** k
    qudit_vec = np.zeros(dim, dtype=np.complex128)
    qudit_vec[state_idx] = 1.0 + 0.0j
    return qudit_vec


# ---------------------------------------------------------------------------
# PART 4 — SINGLE BENCHMARK RUN
# ---------------------------------------------------------------------------


def run_single_mode(
    dfa: Dict[str, Any],
    dfa_name: str,
    n: int,
    mode_name: str,
    correction_mode: Optional[str],
    use_invariants: bool,
) -> Dict[str, Any]:
    """Run a single DFA through one correction mode and extract metrics.

    Steps:
      1. Generate deterministic trajectory.
      2. Embed states into qudit space.
      3. Measure stabilizer syndromes before and after correction.
      4. Compute efficiency metrics.

    Returns dict with dfa_name, n, mode, and all metrics.
    """
    from qec.experiments.correction_layer import apply_correction

    num_states = len(dfa["states"])
    d = 2  # qubit dimension
    # Minimal stabilizer code matching qudit embedding size.
    stab_code = _build_stabilizer_code(d, num_states)

    steps = min(num_states * 2, 20)
    trajectory = _trajectory_from_dfa(dfa, dfa["initial_state"], steps)

    # Embed trajectory and measure syndromes.
    qudit_states: List[np.ndarray] = []
    for sid in trajectory:
        basis = _state_to_vec(sid, num_states)
        qudit_state = _embed_to_qudit(basis, d)
        qudit_states.append(qudit_state)

    before_syn = [stab_code.syndromes(s) for s in qudit_states]

    # Apply correction.
    allowed_states = None
    if use_invariants:
        # Derive allowed states from reachable states in DFA.
        reachable = set(trajectory)
        allowed_states = reachable

    corrected_states: List[np.ndarray] = []
    deltas: List[float] = []
    for s in qudit_states:
        c, delta = apply_correction(s, correction_mode, allowed_states)
        corrected_states.append(c)
        deltas.append(delta)

    # Measure syndromes on corrected states; fall back to original
    # if correction produced a zero vector (cannot measure).
    after_syn = []
    for i, cs in enumerate(corrected_states):
        if np.linalg.norm(cs) > 0:
            after_syn.append(stab_code.syndromes(cs))
        else:
            after_syn.append(before_syn[i])

    metrics = compute_metrics(before_syn, after_syn, deltas)

    # Build per-step alignment data.
    alignment: List[Dict[str, Any]] = []
    for i, (sid, bsyn, asyn) in enumerate(
        zip(trajectory, before_syn, after_syn)
    ):
        alignment.append({
            "step": i,
            "state": int(sid),
            "before": [int(v) for v in bsyn],
            "after": [int(v) for v in asyn],
        })

    return {
        "dfa_name": dfa_name,
        "n": n,
        "mode": mode_name,
        "metrics": metrics,
        "alignment": alignment,
    }


# ---------------------------------------------------------------------------
# PART 5 — SUITE RUNNER
# ---------------------------------------------------------------------------


def run_suite() -> List[Dict[str, Any]]:
    """Run the full multi-DFA benchmark suite.

    Iterates over all DFA types, sizes, and correction modes.
    Returns a list of result dicts.
    """
    results: List[Dict[str, Any]] = []
    for name, builder in sorted(DFA_REGISTRY.items()):
        for n in [5, 10]:
            if "n" in builder.__code__.co_varnames:
                dfa = builder(n)
            else:
                dfa = builder()
            for mode_name, mode, use_inv in MODES:
                results.append(
                    run_single_mode(dfa, name, n, mode_name, mode, use_inv)
                )
    return results


# ---------------------------------------------------------------------------
# PART 6 — RESULT SUMMARY (TAXONOMY)
# ---------------------------------------------------------------------------


def summarize(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Group results by dfa_name → mode → averaged metrics.

    Returns nested dict: {dfa_name: {mode: {metric: value}}}.
    """
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in results:
        key = r["dfa_name"]
        mode = r["mode"]
        grouped.setdefault(key, {}).setdefault(mode, []).append(r["metrics"])

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dfa_name in sorted(grouped):
        summary[dfa_name] = {}
        for mode in sorted(grouped[dfa_name]):
            metric_lists = grouped[dfa_name][mode]
            avg: Dict[str, float] = {}
            for key in ["compression_efficiency", "stability_efficiency"]:
                vals = [m[key] for m in metric_lists]
                avg[key] = safe_div(sum(vals), len(vals))
            summary[dfa_name][mode] = avg
    return summary


def print_summary(
    summary: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """Format summary as a readable text table.

    Returns the formatted string.
    """
    lines: List[str] = []
    for dfa_name in sorted(summary):
        lines.append(f"DFA: {dfa_name}")
        lines.append(f"{'mode':<12} {'comp_eff':>10} {'stab_eff':>10}")
        lines.append("-" * 34)
        for mode in sorted(summary[dfa_name]):
            m = summary[dfa_name][mode]
            lines.append(
                f"{mode:<12} {m['compression_efficiency']:>10.4f}"
                f" {m['stability_efficiency']:>10.4f}"
            )
        lines.append("")
    return "\n".join(lines)
