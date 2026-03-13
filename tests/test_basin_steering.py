from __future__ import annotations

import numpy as np

from src.qec.analysis.nb_spectral_basin_steering import NBSpectralBasinSteering
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


def test_steering_score_formula_and_rounding() -> None:
    steering = NBSpectralBasinSteering()
    steering._predictor.predict_trapping_regions = lambda _H: {  # type: ignore[assignment]
        "spectral_radius": 1.234567890123,
        "ipr": 0.333333333333,
        "risk_score": 0.125,
    }

    result = steering.compute_steering(_matrix())
    expected = round(1.234567890123 + 0.333333333333 + 2.0 * 0.125, 12)

    assert result["steering_score"] == expected
    assert round(result["steering_score"], 12) == result["steering_score"]


def test_basin_steering_is_opt_in_for_mutation() -> None:
    H = _matrix()
    gradient = _gradient()

    mut_default = NBGradientMutator(enabled=True, avoid_4cycles=False, steer_spectral_basins=False)
    mut_steered = NBGradientMutator(enabled=True, avoid_4cycles=False, steer_spectral_basins=True)

    def _raise_if_called(_H: np.ndarray) -> dict:
        raise AssertionError("steering should not be called when disabled")

    called = {"count": 0}

    def _steering_score(_H: np.ndarray) -> dict:
        called["count"] += 1
        return {"steering_score": 1000.0}

    mut_default._basin_steering.compute_steering = _raise_if_called  # type: ignore[assignment]
    mut_steered._basin_steering.compute_steering = _steering_score  # type: ignore[assignment]

    mut_default._find_partner_check = _partner  # type: ignore[assignment]
    mut_steered._find_partner_check = _partner  # type: ignore[assignment]

    step_default = mut_default._apply_single_gradient_step(H.copy(), gradient)
    step_steered = mut_steered._apply_single_gradient_step(H.copy(), gradient)

    assert step_default is not None and step_steered is not None
    assert step_default == step_steered
    assert called["count"] == 1
