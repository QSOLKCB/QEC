"""v14.4.0 — Deterministic two-step spectral beam search planner."""

from __future__ import annotations

import math
from typing import Any, Callable


def _swap_key(candidate: dict[str, Any]) -> tuple:
    removed = tuple((int(ci), int(vi)) for ci, vi in candidate.get("remove", ()))
    added = tuple((int(ci), int(vi)) for ci, vi in candidate.get("add", ()))
    return removed + added


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    return (
        float(candidate.get("score", 0.0)),
        int(candidate.get("swap_index", 0)),
        _swap_key(candidate),
    )


def adaptive_beam_width(
    *,
    basin_depth: float,
    beam_min: int,
    beam_max: int,
    depth_scale: float,
) -> int:
    """Compute deterministic sigmoid-scaled beam width."""
    lo = max(1, int(beam_min))
    hi = max(lo, int(beam_max))
    if hi == lo:
        return lo
    scaled = 1.0 / (1.0 + math.exp(-float(depth_scale) * float(basin_depth)))
    width = lo + int(math.floor((hi - lo) * scaled))
    return max(lo, min(hi, int(width)))


def _swap_edges(candidate: dict[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset((int(ci), int(vi)) for ci, vi in candidate.get("remove", ()))


def select_beam_candidates(
    candidates: list[dict[str, Any]],
    *,
    beam_width: int,
    beam_diversity: bool,
) -> list[dict[str, Any]]:
    """Select first-step beam candidates with deterministic ordering and diversity guard."""
    width = max(1, int(beam_width))
    ranked = sorted(candidates, key=_candidate_sort_key)
    if not beam_diversity:
        return ranked[:width]

    chosen: list[dict[str, Any]] = []
    used_edges: set[tuple[int, int]] = set()
    for candidate in ranked:
        edges = _swap_edges(candidate)
        if edges.intersection(used_edges):
            continue
        chosen.append(candidate)
        used_edges.update(edges)
        if len(chosen) >= width:
            break
    return chosen


def plan_two_step_swap(
    H: Any,
    *,
    enumerate_candidates: Callable[[Any], list[dict[str, Any]]],
    apply_swap: Callable[[Any, dict[str, Any]], Any],
    beam_width: int = 5,
    beam_diversity: bool = False,
    second_step_limit: int = 5,
    beam_score_weight: float = 1.0,
) -> dict[str, Any] | None:
    """Plan a deterministic two-step swap sequence and return first swap.

    Evaluates only a bounded search tree (top-K first-step swaps and
    top-L second-step continuations), then returns the best sequence while
    applying only the first step externally.
    """
    first_candidates = enumerate_candidates(H)
    if not first_candidates:
        return None

    first_ranked = select_beam_candidates(
        first_candidates,
        beam_width=max(1, int(beam_width)),
        beam_diversity=bool(beam_diversity),
    )

    best_plan: dict[str, Any] | None = None
    for first in first_ranked:
        H_next = apply_swap(H, first)
        second_candidates = enumerate_candidates(H_next)
        if not second_candidates:
            continue
        second_ranked = sorted(second_candidates, key=_candidate_sort_key)[: max(1, int(second_step_limit))]
        second = second_ranked[0]

        score_1 = float(first.get("score", 0.0))
        score_2 = float(second.get("score", 0.0))
        total = score_1 + float(beam_score_weight) * score_2
        candidate_plan = {
            "first_swap": first,
            "second_swap": second,
            "planned_sequence_score": float(total),
            "first_score": score_1,
            "second_score": score_2,
        }
        if best_plan is None:
            best_plan = candidate_plan
            continue

        incumbent = (
            float(best_plan["planned_sequence_score"]),
            _candidate_sort_key(best_plan["first_swap"]),
            _candidate_sort_key(best_plan["second_swap"]),
        )
        challenger = (
            float(candidate_plan["planned_sequence_score"]),
            _candidate_sort_key(candidate_plan["first_swap"]),
            _candidate_sort_key(candidate_plan["second_swap"]),
        )
        if challenger < incumbent:
            best_plan = candidate_plan

    return best_plan
