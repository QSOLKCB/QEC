from __future__ import annotations

import numpy as np

from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def _gradient() -> dict:
    return {
        "edge_scores": {
            (0, 0): 2.0,
            (1, 1): 1.0,
        },
        "node_instability": {
            0: 6.0,
            1: 5.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
        },
        "gradient_direction": {
            (0, 0): 7.0,
            (1, 1): 6.0,
        },
    }




def _partner(ci: int, vi: int, vj: int, _check_neighbors: dict, _var_neighbors: dict) -> int | None:
    if ci == 0:
        return 1
    if ci == 1:
        return 0
    return None

def _matrix() -> np.ndarray:
    return np.array([
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.float64)


def test_no_candidate_sets_no_penalty_steering() -> None:
    mut = NBGradientMutator(enabled=True, avoid_4cycles=False, avoid_predicted_trapping_sets=True)
    mut._trapping_predictor.predict_trapping_regions = lambda _H: {  # type: ignore[assignment]
        "candidate_sets": [],
        "node_scores": {0: 100.0},
        "risk_score": 0.0,
    }

    mut._find_partner_check = _partner  # type: ignore[assignment]
    step = mut._apply_single_gradient_step(_matrix().copy(), _gradient())
    assert step is not None
    assert step["removed_edge"] == (0, 0)


def test_penalty_applies_only_for_nodes_in_candidate_sets() -> None:
    mut = NBGradientMutator(enabled=True, avoid_4cycles=False, avoid_predicted_trapping_sets=True)
    mut._trapping_predictor.predict_trapping_regions = lambda _H: {  # type: ignore[assignment]
        "candidate_sets": [[0]],
        "node_scores": {0: 100.0},
        "risk_score": 1.0,
    }

    mut._find_partner_check = _partner  # type: ignore[assignment]
    step = mut._apply_single_gradient_step(_matrix().copy(), _gradient())
    assert step is not None
    assert step["removed_edge"] == (1, 1)


def test_default_behavior_unchanged_when_avoidance_disabled() -> None:
    H = _matrix().copy()
    gradient = _gradient()

    mut_default = NBGradientMutator(enabled=True, avoid_4cycles=False, avoid_predicted_trapping_sets=False)
    mut_with_prediction_ignored = NBGradientMutator(enabled=True, avoid_4cycles=False, avoid_predicted_trapping_sets=False)

    mut_default._find_partner_check = _partner  # type: ignore[assignment]
    mut_with_prediction_ignored._find_partner_check = _partner  # type: ignore[assignment]

    mut_with_prediction_ignored._trapping_predictor.predict_trapping_regions = lambda _H: {  # type: ignore[assignment]
        "candidate_sets": [[0]],
        "node_scores": {0: 9999.0, 1: 9999.0, 2: 9999.0},
        "risk_score": 1.0,
    }

    step_a = mut_default._apply_single_gradient_step(H.copy(), gradient)
    step_b = mut_with_prediction_ignored._apply_single_gradient_step(H.copy(), gradient)

    assert step_a is not None and step_b is not None
    assert step_a == step_b
