"""v87.1.0 — Trajectory Motifs & Periodicity.

Analyses repeating structure inside ternary syndrome trajectories:
  - Loop detection (first repeated state)
  - Periodicity detection (exact periodic repetition)
  - Motif extraction (repeating subsequences)
  - Motif complexity score

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def detect_loops(
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Detect if any state reappears after a gap.

    The first repeated state defines loop start; loop_length is the
    distance between the two occurrences.
    """
    seen: Dict[Tuple[int, ...], int] = {}
    for i, state in enumerate(series):
        if state in seen:
            return {
                "has_loop": True,
                "first_loop_index": seen[state],
                "loop_length": i - seen[state],
            }
        seen[state] = i
    return {"has_loop": False, "first_loop_index": None, "loop_length": None}


def detect_periodicity(
    series: List[Tuple[int, ...]],
    max_period: int = 10,
) -> Dict[str, Any]:
    """Check whether *series* is exactly periodic with period p.

    Tests candidate periods 1..min(max_period, len(series)//2).
    A period p is valid iff series[i] == series[i % p] for all i.
    """
    n = len(series)
    if n == 0:
        return {"is_periodic": False, "period": None}
    for p in range(1, min(max_period, n // 2) + 1):
        if all(series[i] == series[i % p] for i in range(n)):
            return {"is_periodic": True, "period": p}
    return {"is_periodic": False, "period": None}


def extract_motifs(
    series: List[Tuple[int, ...]],
    max_length: int = 5,
) -> Dict[str, Any]:
    """Find repeating subsequences of length 2..max_length.

    Only motifs with count >= 2 are returned.
    Ordering: by length ascending, then lexicographic on pattern tuples.
    """
    n = len(series)
    counts: Dict[Tuple[Tuple[int, ...], ...], int] = {}
    for length in range(2, min(max_length, n) + 1):
        for start in range(n - length + 1):
            key = tuple(series[start : start + length])
            counts[key] = counts.get(key, 0) + 1

    motifs = [
        {"pattern": [list(t) for t in pat], "count": cnt}
        for pat, cnt in counts.items()
        if cnt >= 2
    ]
    motifs.sort(key=lambda m: (len(m["pattern"]), [tuple(t) for t in m["pattern"]]))
    return {"motifs": motifs}


def compute_motif_complexity(
    series: List[Tuple[int, ...]],
    motifs: List[Dict[str, Any]],
) -> float:
    """Complexity = unique_states / len(series), boosted if no motifs."""
    if len(series) == 0:
        return 0.0
    unique = len(set(series))
    complexity = unique / len(series)
    if len(motifs) == 0:
        complexity = min(1.0, complexity * 1.5)
    return complexity


def run_trajectory_motif_analysis(
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Full motif & periodicity analysis on a ternary trajectory."""
    loop = detect_loops(series)
    periodicity = detect_periodicity(series)
    motif_result = extract_motifs(series)
    complexity = compute_motif_complexity(series, motif_result["motifs"])
    return {
        "loop": loop,
        "periodicity": periodicity,
        "motifs": motif_result["motifs"],
        "complexity": complexity,
    }
