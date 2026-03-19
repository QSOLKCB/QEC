from __future__ import annotations

import numpy as np

from qec.analysis.eigenmode_mutation import build_bethe_hessian
from qec.analysis.spectral_energy import compute_bethe_hessian_spectrum


def test_spectral_energy_matches_dense_reference() -> None:
    H = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float64)
    B, r = build_bethe_hessian(H)
    vals_ref, _ = np.linalg.eigh(B.toarray())
    vals_ref = np.asarray(vals_ref, dtype=np.float64)
    energy_ref = float(np.sum((vals_ref[vals_ref < 0.0]) ** 2))

    result = compute_bethe_hessian_spectrum(H, num_eigenvalues=B.shape[0], r=r)

    assert np.allclose(result.eigenvalues, vals_ref)
    assert np.isclose(result.energy, energy_ref)
