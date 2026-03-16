import numpy as np

from src.qec.analysis.ihara_bass_gradient import (
    compute_ihara_bass_gradient,
    compute_ipr,
    compute_severity,
)
from src.qec.discovery.eigenmode_guided_swap import score_2x2_swap


def test_ihara_gradient_is_deterministic() -> None:
    eigvals = np.array([-0.4, -0.1], dtype=np.float64)
    eigvecs = np.array(
        [
            [0.7, 0.1],
            [0.2, 0.8],
            [0.6, -0.1],
            [0.3, -0.5],
        ],
        dtype=np.float64,
    )
    iprs = np.array([compute_ipr(eigvecs[:, 0]), compute_ipr(eigvecs[:, 1])], dtype=np.float64)
    adj = [(0, 2), (1, 3), (0, 3)]

    g1 = compute_ihara_bass_gradient(eigvals, eigvecs, iprs, adj, r=1.5)
    g2 = compute_ihara_bass_gradient(eigvals, eigvecs, iprs, adj, r=1.5)

    assert g1 == g2
    assert list(g1.keys()) == sorted(g1.keys())
    assert compute_severity(-0.4, iprs[0]) > 0.0


def test_swap_scoring_formula() -> None:
    G = {
        (0, 0): 2.0,
        (1, 1): 3.0,
        (0, 1): 1.0,
        (1, 0): -2.0,
    }
    delta = score_2x2_swap(G, 0, 0, 1, 1)
    assert np.isclose(delta, 6.0)


def test_ihara_dual_operator_gradient_is_deterministic() -> None:
    eigvals = np.array([-0.4], dtype=np.float64)
    eigvecs = np.array([[0.7], [0.2], [0.6], [0.3]], dtype=np.float64)
    iprs = np.array([compute_ipr(eigvecs[:, 0])], dtype=np.float64)
    adj = [(0, 2), (0, 3), (1, 3)]

    g1 = compute_ihara_bass_gradient(
        eigvals,
        eigvecs,
        iprs,
        adj,
        r=1.5,
        dual_operator=True,
        bh_eigenvectors=eigvecs,
        w_nb=1.0,
        w_bh=0.5,
    )
    g2 = compute_ihara_bass_gradient(
        eigvals,
        eigvecs,
        iprs,
        adj,
        r=1.5,
        dual_operator=True,
        bh_eigenvectors=eigvecs,
        w_nb=1.0,
        w_bh=0.5,
    )
    assert g1 == g2
