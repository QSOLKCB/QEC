"""v14.4.0 — Deterministic two-step spectral beam search planner."""

from __future__ import annotations

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


def plan_two_step_swap(
    H: Any,
    *,
    enumerate_candidates: Callable[[Any], list[dict[str, Any]]],
    apply_swap: Callable[[Any, dict[str, Any]], Any],
    beam_width: int = 5,
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

    first_ranked = sorted(first_candidates, key=_candidate_sort_key)[: max(1, int(beam_width))]

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

