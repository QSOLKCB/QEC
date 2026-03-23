"""Hierarchical correction stacks with E8-like projection (v96.0.0).

Upgrades from:
    single-step correction + invariant effectiveness analysis
to:
    hierarchical correction stacks + cross-class invariant promotion

Supports multi-stage correction pipelines: square, d4, e8_like,
and their sequential compositions (e.g. square>d4>e8_like).

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec.experiments.correction_layer import (
    apply_correction,
    compute_metrics,
    project_d4,
    project_square,
    safe_div,
)


# ---------------------------------------------------------------------------
# PART 1 — E8-LIKE PROJECTION
# ---------------------------------------------------------------------------

# E8 lattice is an 8-dimensional even unimodular lattice.
# We implement a lightweight deterministic approximation:
#   - chunk input into blocks of 8 (pad/truncate as needed)
#   - round each coordinate to nearest integer
#   - enforce even-sum parity per chunk (D8-like constraint)
#   - enforce even sum-of-squares parity per chunk (extra E8 constraint)
# This gives stronger symmetry than D4 without heavy math.

_E8_CHUNK_SIZE = 8


def _pad_to_multiple(x: np.ndarray, chunk_size: int) -> np.ndarray:
    """Pad vector with zeros to make length a multiple of chunk_size."""
    n = len(x)
    remainder = n % chunk_size
    if remainder == 0:
        return x.copy()
    pad_len = chunk_size - remainder
    return np.concatenate([x, np.zeros(pad_len)])


def _truncate_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Truncate vector back to original length."""
    return x[:length].copy()


def project_e8_like(vec: np.ndarray) -> np.ndarray:
    """Deterministic lightweight 8D-inspired projection.

    Stronger symmetry than D4, simple chunked rule system.
    No symbolic heavy math. No external libraries.

    Rules per 8-element chunk:
      1. Round each coordinate to nearest integer.
      2. Enforce even-sum parity (D8-like).
      3. Enforce even sum-of-squares parity (E8-like extra constraint).

    For both parity fixes, adjust the coordinate closest to a
    half-integer boundary. If both parities need fixing simultaneously,
    adjust the two coordinates with smallest rounding residuals.

    Args:
        vec: input real-valued vector (not mutated).

    Returns:
        Projected vector (same length as input).
    """
    original_len = len(vec)
    x = _pad_to_multiple(vec.astype(float), _E8_CHUNK_SIZE)
    n_chunks = len(x) // _E8_CHUNK_SIZE
    result = np.empty_like(x)

    for c in range(n_chunks):
        start = c * _E8_CHUNK_SIZE
        end = start + _E8_CHUNK_SIZE
        chunk = x[start:end]

        # Step 1: round to nearest integer.
        y = np.round(chunk).astype(int)

        # Residuals for tie-breaking: how close each was to half-integer.
        residuals = np.abs(chunk - y.astype(float))

        # Step 2: check even-sum parity.
        sum_parity = int(np.sum(y)) % 2

        # Step 3: check even sum-of-squares parity.
        sos_parity = int(np.sum(y * y)) % 2

        if sum_parity == 0 and sos_parity == 0:
            # Both parities satisfied — no adjustment needed.
            pass
        elif sum_parity != 0 and sos_parity == 0:
            # Fix sum parity only: flip coordinate closest to half-integer.
            idx = int(np.argmax(residuals))
            y[idx] += 1 if chunk[idx] > y[idx] else -1
        elif sum_parity == 0 and sos_parity != 0:
            # Fix sum-of-squares parity: need to flip TWO coordinates
            # (to keep sum even) choosing the two closest to half-integer.
            order = np.argsort(-residuals)  # largest residual first
            idx_a = int(order[0])
            idx_b = int(order[1])
            y[idx_a] += 1 if chunk[idx_a] > y[idx_a] else -1
            y[idx_b] += 1 if chunk[idx_b] > y[idx_b] else -1
        else:
            # Both parities wrong: flip one coordinate fixes sum parity,
            # then check if sos parity is also fixed.
            idx = int(np.argmax(residuals))
            y[idx] += 1 if chunk[idx] > y[idx] else -1
            # Recheck sos parity after fixing sum.
            if int(np.sum(y * y)) % 2 != 0:
                # Flip two more to fix sos while keeping sum even.
                residuals2 = np.abs(chunk - y.astype(float))
                residuals2[idx] = -1.0  # exclude already-adjusted
                order2 = np.argsort(-residuals2)
                idx_a = int(order2[0])
                idx_b = int(order2[1])
                y[idx_a] += 1 if chunk[idx_a] > y[idx_a] else -1
                y[idx_b] += 1 if chunk[idx_b] > y[idx_b] else -1

        result[start:end] = y.astype(float)

    return _truncate_to_length(result, original_len)


# ---------------------------------------------------------------------------
# PART 2 — PROJECTION DISPATCH & HIERARCHICAL MODES
# ---------------------------------------------------------------------------

HIERARCHICAL_MODES = [
    "square",
    "d4",
    "e8_like",
    "square>d4",
    "d4>e8_like",
    "square>d4>e8_like",
]

# Map single-stage names to projection functions.
_SINGLE_PROJECTIONS = {
    "square": project_square,
    "d4": project_d4,
    "e8_like": project_e8_like,
}


def _parse_stages(mode: str) -> List[str]:
    """Parse a hierarchical mode string into ordered stage names."""
    return mode.split(">")


def project_hierarchical(
    x: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply hierarchical projection (one or more stages).

    Args:
        x: real-valued input vector (not mutated).
        mode: hierarchical mode name (e.g. "square>d4>e8_like").

    Returns:
        Tuple of (projected vector, list of per-stage metadata dicts).
        Each metadata dict has: stage, projection_distance.
    """
    stages = _parse_stages(mode)
    current = x.copy().astype(float)
    stage_metadata: List[Dict[str, Any]] = []

    for stage_name in stages:
        proj_fn = _SINGLE_PROJECTIONS.get(stage_name)
        if proj_fn is None:
            raise ValueError(f"unknown projection stage: {stage_name!r}")
        projected = proj_fn(current)
        dist = float(np.linalg.norm(current - projected))
        stage_metadata.append({
            "stage": stage_name,
            "projection_distance": round(dist, 10),
        })
        current = projected

    return current, stage_metadata


# ---------------------------------------------------------------------------
# PART 3 — HIERARCHICAL CORRECTION RUNNER
# ---------------------------------------------------------------------------


def run_hierarchical_correction(
    dfa: Dict[str, Any],
    mode: str,
    use_invariants: bool = False,
) -> Dict[str, Any]:
    """Run hierarchical correction on a DFA.

    Generates a deterministic trajectory, embeds into qudit space,
    applies hierarchical projection stages, measures syndromes
    before/after, and computes metrics.

    Args:
        dfa: DFA dict with states, alphabet, transitions, initial_state.
        mode: hierarchical mode string.
        use_invariants: if True, use reachable states for damping.

    Returns:
        Dict with mode, stages, projection_distances,
        total_projection_distance, metrics, and efficiency scores.
    """
    from qec.experiments.dfa_benchmark import (
        _build_stabilizer_code,
        _embed_to_qudit,
        _state_to_vec,
        _trajectory_from_dfa,
    )

    num_states = len(dfa["states"])
    d = 2
    stab_code = _build_stabilizer_code(d, num_states)

    steps = min(num_states * 2, 20)
    trajectory = _trajectory_from_dfa(dfa, dfa["initial_state"], steps)

    # Embed trajectory into qudit space.
    qudit_states: List[np.ndarray] = []
    for sid in trajectory:
        basis = _state_to_vec(sid, num_states)
        qudit_state = _embed_to_qudit(basis, d)
        qudit_states.append(qudit_state)

    before_syn = [stab_code.syndromes(s) for s in qudit_states]

    # Determine allowed states for invariant damping.
    allowed_states = None
    if use_invariants:
        allowed_states = set(trajectory)

    # Apply hierarchical correction to each state.
    corrected_states: List[np.ndarray] = []
    deltas: List[float] = []
    all_stage_metadata: List[List[Dict[str, Any]]] = []

    for s in qudit_states:
        # Pre-process: real part + optional damping.
        x = np.real(s).astype(float)
        if allowed_states is not None:
            masked = x.copy()
            for i in range(len(masked)):
                if i not in allowed_states:
                    masked[i] *= 0.5
            x = masked

        projected, stage_meta = project_hierarchical(x, mode)

        # Normalize.
        corrected = projected.astype(float)
        norm = np.linalg.norm(corrected)
        if norm > 0:
            corrected = corrected / norm

        delta = float(np.linalg.norm(s - corrected))
        corrected_states.append(corrected)
        deltas.append(delta)
        all_stage_metadata.append(stage_meta)

    # Measure syndromes on corrected states.
    after_syn = []
    for i, cs in enumerate(corrected_states):
        if np.linalg.norm(cs) > 0:
            after_syn.append(stab_code.syndromes(cs))
        else:
            after_syn.append(before_syn[i])

    metrics = compute_metrics(before_syn, after_syn, deltas)

    # Aggregate stage-level projection distances.
    stages = _parse_stages(mode)
    per_stage_distances: List[float] = []
    for stage_name in stages:
        dists = [
            meta["projection_distance"]
            for step_meta in all_stage_metadata
            for meta in step_meta
            if meta["stage"] == stage_name
        ]
        avg_dist = round(safe_div(sum(dists), len(dists)), 10) if dists else 0.0
        per_stage_distances.append(avg_dist)

    total_projection_distance = round(sum(per_stage_distances), 10)

    return {
        "mode": mode,
        "stages": list(stages),
        "projection_distances": per_stage_distances,
        "total_projection_distance": total_projection_distance,
        "compression_efficiency": metrics["compression_efficiency"],
        "stability_efficiency": metrics["stability_efficiency"],
        "stability_gain": metrics["stability_gain"],
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# PART 4 — COMPARISON AND RANKING
# ---------------------------------------------------------------------------


def compare_hierarchical_modes(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank hierarchical correction results deterministically.

    Sort by:
      1. stability_efficiency descending
      2. compression_efficiency descending
      3. total_projection_distance ascending (lower = less work)
      4. mode name ascending (lexicographic tiebreak)

    Args:
        results: list of run_hierarchical_correction outputs.

    Returns:
        Ranked list with added "rank" field.
    """
    sorted_results = sorted(
        results,
        key=lambda r: (
            -r["stability_efficiency"],
            -r["compression_efficiency"],
            r["total_projection_distance"],
            r["mode"],
        ),
    )

    ranked: List[Dict[str, Any]] = []
    for i, r in enumerate(sorted_results):
        ranked.append({
            "rank": i + 1,
            **r,
        })
    return ranked


def run_all_hierarchical_modes(
    dfa: Dict[str, Any],
    use_invariants: bool = False,
) -> List[Dict[str, Any]]:
    """Run all hierarchical modes on a single DFA.

    Returns list of results, one per mode, sorted deterministically.
    """
    results: List[Dict[str, Any]] = []
    for mode in HIERARCHICAL_MODES:
        result = run_hierarchical_correction(dfa, mode, use_invariants)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_hierarchical_report(report: Dict[str, Any]) -> str:
    """Format hierarchical correction report as human-readable text.

    Args:
        report: output from run_hierarchical_invariant_pipeline or
                a dict with hierarchical_results and comparisons.

    Returns:
        Deterministic, sorted, text-only report string.
    """
    lines: List[str] = []
    lines.append("=== Hierarchical Comparison ===")
    lines.append("")

    comparisons = report.get("comparisons", [])
    if not comparisons:
        lines.append("No comparisons available.")
        return "\n".join(lines)

    for comp in comparisons:
        dfa_name = comp.get("dfa_name", "unknown")
        n = comp.get("n")
        lines.append(f"DFA: {dfa_name} (n={n})")

        baseline = comp.get("baseline_mode", "none")
        hier = comp.get("hierarchical_mode", "none")
        core = comp.get("core_overlay_mode", "none")
        winner = comp.get("best_variant", "unknown")

        lines.append(f"  baseline_best: {baseline}")
        lines.append(f"  hierarchical_best: {hier}")
        lines.append(f"  core_overlay_best: {core}")
        lines.append(f"  winner: {winner}")
        lines.append("")

    # Global best modes.
    global_best = report.get("global_best_modes", {})
    if global_best:
        lines.append("=== Global Best Modes ===")
        for key in sorted(global_best.keys()):
            lines.append(f"  {key}: {global_best[key]}")
        lines.append("")

    return "\n".join(lines)
