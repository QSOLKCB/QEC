"""v13.2.0 — Bethe Hessian and localization diagnostics experiment."""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.analysis.api import BetheHessianAnalyzer, compute_bh_spectrum
from qec.analysis.localization_metrics import (
    IPR,
    ParticipationEntropy,
    SpectralInstabilityScore,
    compute_edge_energy_map,
)


def _sample_H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
    ], dtype=np.float64)


def run() -> dict[str, float | int]:
    H = _sample_H()
    analyzer = BetheHessianAnalyzer()
    stability = analyzer.compute_bethe_hessian_stability(H)

    bh = compute_bh_spectrum(H)
    H_B = scipy.sparse.csr_matrix(bh["bethe_hessian"], dtype=np.float64).toarray()
    H_sparse = scipy.sparse.csr_matrix(H, dtype=np.float64)
    A = H_sparse.T.dot(H_sparse).tocsr()
    A.setdiag(0.0)
    A = ((A != 0).astype(np.float64)).tocsr()

    eigvals, eigvecs = np.linalg.eigh(H_B)
    idx = int(np.argmin(eigvals))
    v = eigvecs[:, idx]

    ipr = IPR.compute(v)
    entropy = ParticipationEntropy.compute(v)
    score = SpectralInstabilityScore.compute(stability["bethe_hessian_min_eigenvalue"], ipr, entropy)

    edge_energy_map = compute_edge_energy_map(A, v)
    localized_edges = sum(1 for _, _, e in edge_energy_map if e >= 2.0 * ipr)

    return {
        "lambda_min": score["lambda_min"],
        "IPR": score["ipr"],
        "entropy": score["entropy"],
        "spectral_instability_score": score["spectral_instability_score"],
        "localized_edges": int(localized_edges),
    }


def main() -> None:
    result = run()
    print(f"lambda_min: {result['lambda_min']:.4f}")
    print(f"IPR: {result['IPR']:.3f}")
    print(f"entropy: {result['entropy']:.2f}")
    print(f"spectral_instability_score: {result['spectral_instability_score']:.3f}")
    print(f"localized_edges: {result['localized_edges']}")


if __name__ == "__main__":
    main()
