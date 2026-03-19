from __future__ import annotations

import numpy as np
import scipy.sparse

from qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _rounded_12(v: float) -> bool:
    return abs(v - round(v, 12)) <= 1e-15


def test_edge_scores_and_hot_edges_are_deterministic() -> None:
    analyzer = NBEigenmodeFlowAnalyzer()
    r1 = analyzer.analyze(_H())
    r2 = analyzer.analyze(_H())
    assert r1["edge_scores"] == r2["edge_scores"]
    assert r1["hot_edges"] == r2["hot_edges"]


def test_required_signature_fields_exist_and_are_rounded() -> None:
    result = NBEigenmodeFlowAnalyzer().analyze(_H())
    for key in ["spectral_radius", "mode_ipr", "support_fraction", "topk_mass_fraction"]:
        assert key in result
        assert key in result["signature"]
        assert _rounded_12(float(result[key]))
        assert _rounded_12(float(result["signature"][key]))


def test_sparse_input_supported() -> None:
    dense = _H()
    sparse = scipy.sparse.csr_matrix(dense)
    analyzer = NBEigenmodeFlowAnalyzer()
    r_dense = analyzer.analyze(dense)
    r_sparse = analyzer.analyze(sparse)
    assert r_dense["signature"] == r_sparse["signature"]
    assert r_dense["hot_edges"] == r_sparse["hot_edges"]


def test_zero_signal_returns_zero_support_and_topk_mass() -> None:
    H = np.array([[1.0]], dtype=np.float64)
    result = NBEigenmodeFlowAnalyzer().analyze(H)
    assert result["support_fraction"] == 0.0
    assert result["topk_mass_fraction"] == 0.0


def test_precision_parameter_is_applied() -> None:
    result = NBEigenmodeFlowAnalyzer(precision=6).analyze(_H())
    assert all(abs(v - round(v, 6)) <= 1e-12 for v in result["signature"].values())
    assert all(abs(v - round(v, 6)) <= 1e-12 for v in result["edge_scores"].values())
