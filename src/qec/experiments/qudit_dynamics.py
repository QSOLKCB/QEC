"""Deterministic qudit measurement layer and experiment pipeline (v91.1.0).

DFA → trajectory → qudit state → stabilizers → syndrome evolution.
Optional: geometric correction, syndrome compression, stabilizer metadata.

All algorithms are pure, deterministic, and use only stdlib + numpy.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from qec.experiments.dfa_engine import step


# ---------------------------------------------------------------------------
# PART A — DFA → STATE VECTOR
# ---------------------------------------------------------------------------


def state_to_basis_vector(state_id: int, num_states: int) -> np.ndarray:
    """Map DFA state to basis vector |s> in R^n (one-hot).

    Deterministic ordering by sorted state IDs.

    Args:
        state_id: index of the state (0-based).
        num_states: total number of states n.

    Returns:
        One-hot vector of shape (num_states,).
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if state_id < 0 or state_id >= num_states:
        raise ValueError(
            f"state_id {state_id} out of range [0, {num_states})"
        )
    vec = np.zeros(num_states, dtype=np.float64)
    vec[state_id] = 1.0
    return vec


def trajectory_to_states(
    dfa: Dict[str, Any], start_state: int, steps: int
) -> List[int]:
    """Deterministic trajectory using existing DFA step logic.

    At each step the smallest available symbol (sorted alphabet) that
    yields a valid transition is used.  This guarantees a single
    deterministic path with no randomness.

    Args:
        dfa: DFA dict with "alphabet", "transitions".
        start_state: initial state id.
        steps: number of transitions to take.

    Returns:
        List of state ids of length ``steps + 1`` (includes start).
    """
    alphabet = sorted(dfa.get("alphabet", []))
    trajectory: List[int] = [start_state]
    current = start_state
    for _ in range(steps):
        next_state: Optional[int] = None
        for sym in alphabet:
            ns = step(dfa, current, sym)
            if ns is not None:
                next_state = ns
                break
        if next_state is None:
            # No outgoing transition — stay (absorbing).
            trajectory.append(current)
        else:
            trajectory.append(next_state)
            current = next_state
    return trajectory


# ---------------------------------------------------------------------------
# PART B — QUDIT EMBEDDING
# ---------------------------------------------------------------------------


def embed_state_to_qudit(state_vec: np.ndarray, d: int) -> np.ndarray:
    """Embed DFA basis vector into qudit Hilbert space.

    Strategy:
      - Find the state index from the one-hot vector.
      - Compute minimal k such that d^k >= num_states.
      - Map state index to computational basis of dimension d^k.
      - Zero-pad deterministically.

    Args:
        state_vec: one-hot basis vector of length num_states.
        d: local qudit dimension.

    Returns:
        Normalized state vector in C^(d^k).
    """
    if d < 2:
        raise ValueError("qudit dimension d must be >= 2")

    num_states = len(state_vec)
    # Find state index from one-hot.
    nonzero = np.nonzero(state_vec)[0]
    if len(nonzero) != 1:
        raise ValueError("state_vec must be a one-hot vector")
    state_idx = int(nonzero[0])

    # Minimal k such that d^k >= num_states.
    k = 1
    while d ** k < num_states:
        k += 1
    dim = d ** k

    # Computational basis vector in C^(d^k).
    qudit_vec = np.zeros(dim, dtype=np.complex128)
    qudit_vec[state_idx] = 1.0 + 0.0j
    return qudit_vec


# ---------------------------------------------------------------------------
# PART B2 — STABILIZER METADATA
# ---------------------------------------------------------------------------


def build_stabilizer_metadata(stabilizer_code: Any) -> Dict[str, Any]:
    """Extract deterministic metadata from a stabilizer code instance.

    Returns:
        {"d": int, "n_qudits": int, "generator_count": int}
    """
    return {
        "d": int(stabilizer_code.d),
        "n_qudits": int(stabilizer_code.n),
        "generator_count": len(stabilizer_code.generators),
    }


# ---------------------------------------------------------------------------
# PART C — STABILIZER MEASUREMENT
# ---------------------------------------------------------------------------


def measure_trajectory(
    trajectory: List[int],
    d: int,
    stabilizer_code: Any,
) -> Dict[str, Any]:
    """Measure stabilizers along a DFA trajectory.

    For each state in trajectory:
      - convert to basis vector
      - embed into qudit space
      - measure stabilizers
      - extract syndrome

    Args:
        trajectory: list of DFA state ids.
        d: qudit dimension.
        stabilizer_code: a QuditStabilizerCode instance with
            ``measure_stabilizers(state)`` and ``syndromes(state)``
            methods.

    Returns:
        Dict with keys "states", "stabilizer_values", "syndromes".
    """
    if not trajectory:
        return {
            "states": [],
            "stabilizer_values": [],
            "syndromes": [],
            "state_syndrome_pairs": [],
        }

    num_states = max(trajectory) + 1

    states_out: List[np.ndarray] = []
    stab_vals: List[List[complex]] = []
    syndromes: List[np.ndarray] = []
    state_syndrome_pairs: List[Dict[str, Any]] = []

    for sid in trajectory:
        basis = state_to_basis_vector(sid, num_states)
        qudit_state = embed_state_to_qudit(basis, d)
        vals = stabilizer_code.measure_stabilizers(qudit_state)
        synd = stabilizer_code.syndromes(qudit_state)

        states_out.append(qudit_state.copy())
        stab_vals.append(list(vals))
        syndromes.append(synd.copy())
        state_syndrome_pairs.append({
            "state": int(sid),
            "syndrome": [int(x) for x in synd],
        })

    return {
        "states": states_out,
        "stabilizer_values": stab_vals,
        "syndromes": syndromes,
        "state_syndrome_pairs": state_syndrome_pairs,
    }


# ---------------------------------------------------------------------------
# PART D — SYNDROME EVOLUTION METRICS
# ---------------------------------------------------------------------------


def analyze_syndrome_evolution(
    syndromes: List[np.ndarray],
) -> Dict[str, Any]:
    """Compute deterministic metrics from a syndrome sequence.

    Metrics:
      - unique_syndromes: sorted list of unique syndrome tuples.
      - transition_count: number of steps where syndrome changes.
      - stable_regions: list of (start, length) for runs of identical
        syndromes.
      - change_points: indices where syndrome differs from previous.

    Args:
        syndromes: list of 1-d integer arrays.

    Returns:
        Dict with all metrics.
    """
    if not syndromes:
        return {
            "unique_syndromes": [],
            "transition_count": 0,
            "stable_regions": [],
            "change_points": [],
        }

    # Unique syndromes — convert to tuples for hashability, sorted.
    seen: Dict[tuple, None] = {}
    for s in syndromes:
        key = tuple(int(x) for x in s)
        if key not in seen:
            seen[key] = None
    unique = sorted(seen.keys())

    # Change points and transition count.
    change_points: List[int] = []
    for i in range(1, len(syndromes)):
        if not np.array_equal(syndromes[i], syndromes[i - 1]):
            change_points.append(i)
    transition_count = len(change_points)

    # Stable regions — runs of identical consecutive syndromes.
    stable_regions: List[tuple] = []
    run_start = 0
    for i in range(1, len(syndromes)):
        if not np.array_equal(syndromes[i], syndromes[run_start]):
            stable_regions.append((run_start, i - run_start))
            run_start = i
    stable_regions.append((run_start, len(syndromes) - run_start))

    return {
        "unique_syndromes": unique,
        "transition_count": transition_count,
        "stable_regions": stable_regions,
        "change_points": change_points,
    }


# ---------------------------------------------------------------------------
# PART D2 — SYNDROME COMPRESSION
# ---------------------------------------------------------------------------


def measure_corrected_states(
    states: List[np.ndarray], stabilizer_code: Any
) -> List[Dict[str, Any]]:
    """Re-measure stabilizers on corrected state vectors.

    Args:
        states: corrected qudit state vectors.
        stabilizer_code: QuditStabilizerCode instance.

    Returns:
        List of dicts with "values" and "syndrome" keys.
    """
    corrected_measurements: List[Dict[str, Any]] = []
    for s in states:
        vals = stabilizer_code.measure_stabilizers(s)
        synd = stabilizer_code.syndromes(s)
        corrected_measurements.append({
            "values": vals,
            "syndrome": synd,
        })
    return corrected_measurements


def compress_syndrome(syndrome: np.ndarray) -> str:
    """Deterministic string signature for a syndrome vector.

    Converts each element to int and joins with underscore.
    Example: np.array([0, 1, 0, 2]) → "0_1_0_2"

    Args:
        syndrome: 1-d integer array.

    Returns:
        Deterministic string signature.
    """
    return "_".join(str(int(x)) for x in syndrome)


# ---------------------------------------------------------------------------
# PART E — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_qudit_dynamics(
    dfa: Dict[str, Any],
    start_state: int,
    steps: int,
    d: int,
    stabilizer_code: Any,
    correction_mode: Optional[str] = None,
    run_correction: bool = False,
    compress: bool = False,
) -> Dict[str, Any]:
    """Full deterministic qudit dynamics pipeline (v91.1.0).

    1. Generate DFA trajectory.
    2. Embed into qudit space.
    3. Measure stabilizers.
    4. Extract syndromes.
    5. Analyze evolution.
    6. (Optional) Apply lattice-projection correction.
    7. (Optional) Run correction experiments for all modes.
    3. (Optional) Apply geometric correction.
    4. Measure stabilizers.
    5. Extract syndromes.
    6. Analyze evolution.

    Args:
        dfa: DFA dict.
        start_state: initial DFA state.
        steps: number of DFA transitions.
        d: qudit local dimension.
        stabilizer_code: QuditStabilizerCode instance.
        correction_mode: if set, apply this projection to states
            and record the mean correction delta.
        run_correction: if True, run correction experiments for
            all projection modes (None, "square", "d4").

    Returns:
        Dict with "trajectory", "qudit", "syndrome_analysis",
        and optionally "correction" and/or "experiments".
        correction_mode: optional geometric correction mode ("square", "d4").
            None means no correction.
        compress: if True, include syndrome_signatures in output.

    Returns:
        Dict with "trajectory", "qudit", "syndrome_analysis",
        "stabilizer_metadata", and optionally "corrected",
        "correction_effect", "syndrome_signatures".
    """
    from qec.experiments.correction_layer import (
        apply_correction,
        run_correction_experiment,
    )

    trajectory = trajectory_to_states(dfa, start_state, steps)

    qudit_result = measure_trajectory(trajectory, d, stabilizer_code)

    syndrome_analysis = analyze_syndrome_evolution(qudit_result["syndromes"])

    stabilizer_metadata = build_stabilizer_metadata(stabilizer_code)

    result: Dict[str, Any] = {
        "trajectory": trajectory,
        "qudit": qudit_result,
        "syndrome_analysis": syndrome_analysis,
        "stabilizer_metadata": stabilizer_metadata,
    }

    # v91.1.0 — syndrome compression.
    if compress:
        result["syndrome_signatures"] = [
            compress_syndrome(s) for s in qudit_result["syndromes"]
        ]

    # v92.1.0 — lattice-projection correction with re-measurement.
    if correction_mode is not None:
        corrected_states: List[np.ndarray] = []
        deltas: List[float] = []
        for s in qudit_result["states"]:
            c, delta = apply_correction(s, correction_mode)
            corrected_states.append(c)
            deltas.append(delta)

        corrected_measurements = measure_corrected_states(
            corrected_states, stabilizer_code
        )
        corrected_syndromes = [m["syndrome"] for m in corrected_measurements]

        result["correction"] = {
            "mode": correction_mode,
            "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
            "corrected_syndromes": corrected_syndromes,
        }

    return result
