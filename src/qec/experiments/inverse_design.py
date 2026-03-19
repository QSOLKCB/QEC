"""v82.5.0 — Inverse Design Engine (Targeted Sequence Search).

Bounded deterministic search over candidate note sequences to find
inputs that most reliably produce a target behavior class.

Given a desired behavior (e.g. ``"stable"``, ``"fragile"``, ``"chaotic"``,
``"boundary_rider"``), generates a deterministic candidate set and returns
the best-matching sequences ranked by a deterministic scoring function.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from qec.experiments.sequence_landscape import (
    _extract_sequence_summary,
    classify_sequence,
)


# ---------------------------------------------------------------------------
# Valid target classes
# ---------------------------------------------------------------------------

_VALID_TARGETS = frozenset({"stable", "fragile", "chaotic", "boundary_rider"})

_WORST_SCORE = 1e9


# ---------------------------------------------------------------------------
# Step 1 — Candidate Sequence Generator
# ---------------------------------------------------------------------------

def generate_candidate_sequences(
    notes: List[int],
    lengths: List[int],
    *,
    velocity: int = 100,
) -> List[List[Dict[str, Any]]]:
    """Generate a deterministic set of simple note sequences.

    Produces four pattern families for each ``(note, length)`` pair:

    - **repeated**: the same note repeated *length* times
    - **ascending**: ascending slice of *notes* starting at *note*
    - **descending**: descending slice of *notes* starting at *note*
    - **alternating**: alternates between *note* and the next note

    Parameters
    ----------
    notes : list[int]
        MIDI note values to build patterns from.
    lengths : list[int]
        Sequence lengths to generate for each pattern.
    velocity : int, optional
        MIDI velocity for all events (default ``100``).

    Returns
    -------
    list[list[dict]]
        Each inner list is a sequence of note events with keys
        ``note``, ``velocity``, ``time``.
    """
    if not notes or not lengths:
        return []

    candidates: List[List[Dict[str, Any]]] = []
    seen: set = set()

    for note in notes:
        note_idx = notes.index(note)
        for length in lengths:
            patterns = _build_patterns(notes, note_idx, length, velocity)
            for pat in patterns:
                key = tuple((e["note"], e["time"]) for e in pat)
                if key not in seen:
                    seen.add(key)
                    candidates.append(pat)

    return candidates


def _build_patterns(
    notes: List[int],
    note_idx: int,
    length: int,
    velocity: int,
) -> List[List[Dict[str, Any]]]:
    """Build the four pattern families for a given note index and length."""
    note = notes[note_idx]
    n = len(notes)
    patterns: List[List[Dict[str, Any]]] = []

    # Repeated
    patterns.append([
        {"note": note, "velocity": velocity, "time": float(i)}
        for i in range(length)
    ])

    # Ascending — wrap around notes list
    asc = [notes[(note_idx + i) % n] for i in range(length)]
    patterns.append([
        {"note": asc[i], "velocity": velocity, "time": float(i)}
        for i in range(length)
    ])

    # Descending — wrap around notes list
    desc = [notes[(note_idx - i) % n] for i in range(length)]
    patterns.append([
        {"note": desc[i], "velocity": velocity, "time": float(i)}
        for i in range(length)
    ])

    # Alternating — between current note and next note
    alt_note = notes[(note_idx + 1) % n]
    alt = [note if i % 2 == 0 else alt_note for i in range(length)]
    patterns.append([
        {"note": alt[i], "velocity": velocity, "time": float(i)}
        for i in range(length)
    ])

    return patterns


# ---------------------------------------------------------------------------
# Step 2 — Sequence Evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    seq: List[Dict[str, Any]],
    pipeline_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a single candidate sequence through the pipeline.

    Parameters
    ----------
    seq : list[dict]
        Note event sequence.
    pipeline_fn : callable
        Pipeline function that accepts a sequence and returns a full
        pipeline result dict.

    Returns
    -------
    dict
        Compact summary with keys: ``stability_score``, ``phase``,
        ``trajectory_class``, ``sequence_class``, ``consensus``,
        ``verified``, ``invariant_strength``.
    """
    result = pipeline_fn(seq)
    summary = _extract_sequence_summary(seq, result)
    summary["sequence_class"] = classify_sequence(summary)
    return summary


# ---------------------------------------------------------------------------
# Step 3 — Target Match Scoring
# ---------------------------------------------------------------------------

def score_candidate(summary: Dict[str, Any], target: str) -> float:
    """Score a candidate summary against a target behavior.

    Lower score is better.  Candidates that fail consensus or
    verification receive ``_WORST_SCORE``.

    Parameters
    ----------
    summary : dict
        Output of :func:`evaluate_sequence`.  Not mutated.
    target : str
        One of ``"stable"``, ``"fragile"``, ``"chaotic"``,
        ``"boundary_rider"``.

    Returns
    -------
    float
        Scalar score (lower is better).
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; valid: {sorted(_VALID_TARGETS)}"
        )

    # Hard filter: consensus and verification must pass
    if not summary.get("consensus", False) or not summary.get("verified", False):
        return _WORST_SCORE

    score_fn = _TARGET_SCORERS[target]
    return score_fn(summary)


def _score_stable(s: Dict[str, Any]) -> float:
    """Prefer low stability_score, stable class, non-divergent trajectory."""
    score = 0.0
    score += s.get("stability_score", 0.0)
    if s.get("sequence_class") != "stable":
        score += 10.0
    if s.get("trajectory_class") in ("divergent", "chaotic"):
        score += 5.0
    if s.get("phase") in ("chaotic_transition", "unstable_region"):
        score += 5.0
    return score


def _score_fragile(s: Dict[str, Any]) -> float:
    """Prefer moderate stability_score near 0.5–1.5, fragile class."""
    score = 0.0
    stability = s.get("stability_score", 0.0)
    # Distance from ideal range center (1.0)
    score += abs(stability - 1.0)
    if s.get("sequence_class") != "fragile":
        score += 10.0
    if s.get("phase") == "near_boundary":
        score -= 1.0  # bonus
    return max(score, 0.0)


def _score_chaotic(s: Dict[str, Any]) -> float:
    """Prefer high stability_score, chaotic class, unstable trajectory."""
    score = 0.0
    stability = s.get("stability_score", 0.0)
    # Invert: higher stability_score → lower (better) score
    score += max(0.0, 5.0 - stability)
    if s.get("sequence_class") != "chaotic":
        score += 10.0
    if s.get("trajectory_class") not in ("chaotic", "divergent", "oscillating"):
        score += 5.0
    return score


def _score_boundary_rider(s: Dict[str, Any]) -> float:
    """Prefer near-boundary phase, boundary_rider class."""
    score = 0.0
    if s.get("sequence_class") != "boundary_rider":
        score += 10.0
    if s.get("phase") not in ("near_boundary", "unstable_region"):
        score += 5.0
    stability = s.get("stability_score", 0.0)
    # Prefer moderate stability near boundary
    score += abs(stability - 1.5)
    return score


_TARGET_SCORERS = {
    "stable": _score_stable,
    "fragile": _score_fragile,
    "chaotic": _score_chaotic,
    "boundary_rider": _score_boundary_rider,
}


# ---------------------------------------------------------------------------
# Step 4 — Search Engine
# ---------------------------------------------------------------------------

def run_inverse_design(
    target: str,
    notes: List[int],
    lengths: List[int],
    pipeline_fn: Callable[..., Dict[str, Any]],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run bounded inverse design search.

    Generates deterministic candidate sequences, evaluates each through
    *pipeline_fn*, scores against *target*, and returns the best matches.

    Parameters
    ----------
    target : str
        Target behavior class.
    notes : list[int]
        MIDI notes for candidate generation.
    lengths : list[int]
        Sequence lengths for candidate generation.
    pipeline_fn : callable
        Pipeline function accepting a sequence and returning a result dict.
    top_k : int, optional
        Number of best candidates to return (default ``5``).

    Returns
    -------
    dict
        Result with keys: ``target``, ``n_candidates``, ``best``,
        ``all_scores``.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; valid: {sorted(_VALID_TARGETS)}"
        )

    candidates = generate_candidate_sequences(notes, lengths)

    scored: List[Dict[str, Any]] = []
    for seq in candidates:
        summary = evaluate_sequence(seq, pipeline_fn)
        s = score_candidate(summary, target)
        scored.append({
            "sequence": seq,
            "score": s,
            "summary": summary,
        })

    # Sort by score ascending (lower is better), stable sort
    scored.sort(key=lambda x: x["score"])

    return {
        "target": target,
        "n_candidates": len(candidates),
        "best": scored[:top_k],
        "all_scores": [entry["score"] for entry in scored],
    }
