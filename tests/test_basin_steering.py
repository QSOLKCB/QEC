from __future__ import annotations

import numpy as np

from qec.analysis.nb_spectral_basin_steering import NBSpectralBasinSteering
from qec.discovery.mutation_nb_gradient import NBGradientMutator


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

    def _steering_score(_H: np.ndarray, prediction: dict | None = None) -> dict:
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


def test_compute_steering_reuses_precomputed_prediction() -> None:
    steering = NBSpectralBasinSteering()

    def _raise_if_called(_H: np.ndarray) -> dict:
        raise AssertionError("predictor should not be called when prediction is supplied")

    steering._predictor.predict_trapping_regions = _raise_if_called  # type: ignore[assignment]
    pred = {"spectral_radius": 0.4, "ipr": 0.2, "risk_score": 0.1, "candidate_sets": [[0]]}

    result = steering.compute_steering(_matrix(), prediction=pred)

    assert result["risk_score"] == 0.1
    assert result["trapping_risk"] == result["risk_score"]
    assert result["steering_score"] == round(0.4 + 0.2 + 2.0 * 0.1, 12)


def test_gradient_mutator_shares_prediction_between_penalty_and_steering() -> None:
    H = _matrix()
    gradient = _gradient()
    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        avoid_predicted_trapping_sets=True,
        steer_spectral_basins=True,
    )

    calls = {"predict": 0, "steer": 0}

    def _predict(_H: np.ndarray) -> dict:
        calls["predict"] += 1
        return {
            "candidate_sets": [[0]],
            "node_scores": {0: 1.5},
            "risk_score": 0.25,
            "spectral_radius": 0.5,
            "ipr": 0.2,
        }

    def _steer(_H: np.ndarray, prediction: dict | None = None) -> dict:
        calls["steer"] += 1
        assert prediction is not None
        assert prediction["risk_score"] == 0.25
        return {"steering_score": 0.5}

    mut._trapping_predictor.predict_trapping_regions = _predict  # type: ignore[assignment]
    mut._basin_steering.compute_steering = _steer  # type: ignore[assignment]
    mut._find_partner_check = _partner  # type: ignore[assignment]

    step = mut._apply_single_gradient_step(H.copy(), gradient)
    assert step is not None
    assert calls == {"predict": 1, "steer": 1}


def test_edge_ranking_uses_raw_scores_before_rounding() -> None:
    H = _matrix()
    gradient = {
        "edge_scores": {(0, 0): 1.00000000000049, (1, 1): 1.00000000000048},
        "node_instability": _gradient()["node_instability"],
        "gradient_direction": {(0, 0): 9.0, (1, 1): 8.0},
    }
    mut = NBGradientMutator(enabled=True, avoid_4cycles=False, steer_spectral_basins=True)
    mut._basin_steering.compute_steering = lambda _H, prediction=None: {"steering_score": 0.0}  # type: ignore[assignment]
    mut._find_partner_check = _partner  # type: ignore[assignment]

    step = mut._apply_single_gradient_step(H.copy(), gradient)
    assert step is not None
    assert step["removed_edge"] == (0, 0)
