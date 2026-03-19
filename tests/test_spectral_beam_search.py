from __future__ import annotations

import numpy as np
import pytest

from qec.discovery.mutation_nb_gradient import NBGradientMutator
from qec.discovery.spectral_beam_search import plan_two_step_swap


def test_plan_two_step_swap_lookahead_beats_greedy() -> None:
    transitions = {
        "root": [
            {"name": "a", "score": -5.0, "swap_index": 0, "remove": ((0, 0), (1, 1)), "add": ((0, 1), (1, 0))},
            {"name": "b", "score": -3.0, "swap_index": 1, "remove": ((0, 1), (1, 0)), "add": ((0, 0), (1, 1))},
        ],
        "root->a": [
            {"name": "a2", "score": +10.0, "swap_index": 0, "remove": ((0, 0), (1, 1)), "add": ((0, 1), (1, 0))},
        ],
        "root->b": [
            {"name": "b2", "score": -8.0, "swap_index": 0, "remove": ((0, 1), (1, 0)), "add": ((0, 0), (1, 1))},
        ],
    }

    def _enumerate(state: str) -> list[dict]:
        return [dict(c) for c in transitions.get(state, [])]

    def _apply(state: str, swap: dict) -> str:
        return f"{state}->{swap['name']}"

    plan = plan_two_step_swap(
        "root",
        enumerate_candidates=_enumerate,
        apply_swap=_apply,
        beam_width=2,
        second_step_limit=1,
    )

    assert plan is not None
    assert plan["first_swap"]["name"] == "b"
    assert plan["second_swap"]["name"] == "b2"


def test_plan_two_step_swap_is_deterministic_on_ties() -> None:
    candidates = [
        {"name": "x", "score": -1.0, "swap_index": 0, "remove": ((0, 0), (1, 1)), "add": ((0, 1), (1, 0))},
        {"name": "y", "score": -1.0, "swap_index": 1, "remove": ((0, 0), (1, 2)), "add": ((0, 2), (1, 0))},
    ]

    def _enumerate(state: str) -> list[dict]:
        if state == "root":
            return [dict(c) for c in candidates]
        return [{"name": "z", "score": 0.0, "swap_index": 0, "remove": ((0, 0), (1, 1)), "add": ((0, 1), (1, 0))}]

    def _apply(state: str, swap: dict) -> str:
        return f"{state}->{swap['name']}"

    p1 = plan_two_step_swap("root", enumerate_candidates=_enumerate, apply_swap=_apply)
    p2 = plan_two_step_swap("root", enumerate_candidates=_enumerate, apply_swap=_apply)

    assert p1 == p2
    assert p1 is not None
    assert p1["first_swap"]["name"] == "x"


def test_mutator_beam_search_opt_in_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    H = np.array([
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.float64)

    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        enable_spectral_beam_search=True,
        beam_width=2,
        second_step_limit=2,
    )

    gradient = {
        "edge_scores": {(0, 0): 2.0, (1, 1): 1.0},
        "node_instability": {0: 6.0, 1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0},
        "gradient_direction": {(0, 0): 7.0, (1, 1): 6.0},
    }

    mut._analyzer.compute_gradient = lambda _H: gradient  # type: ignore[assignment]
    mut._trapping_predictor.predict_trapping_regions = lambda _H: {"ipr": 0.3, "risk_score": 0.2}  # type: ignore[assignment]

    def _partner(ci: int, vi: int, vj: int, _check_neighbors: dict, _var_neighbors: dict) -> int | None:
        if ci == 0:
            return 1
        if ci == 1:
            return 0
        return None

    mut._find_partner_check = _partner  # type: ignore[assignment]

    def _fake_plan(_H, **kwargs):
        candidates = kwargs["enumerate_candidates"](_H)
        assert candidates
        return {
            "first_swap": candidates[0],
            "second_swap": candidates[0],
            "planned_sequence_score": -1.0,
        }

    monkeypatch.setattr("src.qec.discovery.mutation_nb_gradient.plan_two_step_swap", _fake_plan)

    H1, log1 = mut.mutate_flow(H, iterations=1)
    H2, log2 = mut.mutate_flow(H, iterations=1)
    np.testing.assert_array_equal(H1, H2)
    assert log1 == log2
    assert log1
    assert log1[0]["beam_search_used"] is True
    assert log1[0]["beam_width"] == 2
    assert isinstance(log1[0]["planned_sequence_score"], float)
