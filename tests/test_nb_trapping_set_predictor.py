"""Tests for v12.8.0 NB trapping-set predictor."""

from __future__ import annotations

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _dense_matrix() -> np.ndarray:
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
    ], dtype=np.float64)


class TestNBTrappingSetPredictor:
    def test_determinism(self) -> None:
        H = _matrix()
        pred = NBTrappingSetPredictor()
        r1 = pred.predict_trapping_regions(H)
        r2 = pred.predict_trapping_regions(H)
        assert r1 == r2

    def test_sparse_dense_equivalence(self) -> None:
        H = _matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        pred = NBTrappingSetPredictor()
        assert pred.predict_trapping_regions(H) == pred.predict_trapping_regions(H_sp)

    def test_clustering_behavior_returns_sorted_components(self) -> None:
        H = _dense_matrix()
        pred = NBTrappingSetPredictor()
        result = pred.predict_trapping_regions(H)
        for comp in result["candidate_sets"]:
            assert comp == sorted(comp)

    def test_empty_graph_edge_case(self) -> None:
        H = np.zeros((0, 0), dtype=np.float64)
        pred = NBTrappingSetPredictor()
        result = pred.predict_trapping_regions(H)
        assert result == {
            "node_scores": {},
            "edge_scores": {},
            "candidate_sets": [],
            "candidate_scores": [],
            "ipr": 0.0,
            "spectral_radius": 0.0,
            "risk_score": 0.0,
            "trapping_risk": 0.0,
        }

    def test_rounding_precision(self) -> None:
        H = _matrix()
        pred = NBTrappingSetPredictor()
        result = pred.predict_trapping_regions(H)
        assert round(result["ipr"], 12) == result["ipr"]
        assert round(result["spectral_radius"], 12) == result["spectral_radius"]
        assert round(result["risk_score"], 12) == result["risk_score"]
        for val in result["node_scores"].values():
            assert round(val, 12) == val
        for val in result["edge_scores"].values():
            assert round(val, 12) == val

    def test_risk_score_computation(self) -> None:
        H = _matrix()
        pred = NBTrappingSetPredictor()
        result = pred.predict_trapping_regions(H)
        candidate_nodes = [v for comp in result["candidate_sets"] for v in comp]
        if candidate_nodes:
            mean_score = float(np.mean([result["node_scores"][v] for v in candidate_nodes]))
            expected = round(mean_score * result["ipr"], 12)
            assert result["risk_score"] == expected
        else:
            assert result["risk_score"] == 0.0


def test_risk_score_alias_for_backward_compatibility() -> None:
    result = NBTrappingSetPredictor().predict_trapping_regions(_matrix())
    assert result["trapping_risk"] == result["risk_score"]
