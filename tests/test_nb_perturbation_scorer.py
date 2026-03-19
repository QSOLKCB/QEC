from __future__ import annotations

import numpy as np
import scipy.sparse

from qec.analysis.nb_perturbation_scorer import NBPerturbationScorer


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_fohpe_denominator_present_and_nonzero() -> None:
    spectrum = NBPerturbationScorer().compute_nb_spectrum(_H())
    assert spectrum["valid_first_order"] is True
    assert "fohpe_denominator" in spectrum
    assert abs(float(spectrum["fohpe_denominator"])) > 1e-15


def test_eigenvectors_are_l2_normalized() -> None:
    spectrum = NBPerturbationScorer().compute_nb_spectrum(_H())
    u = np.asarray(spectrum["u"], dtype=np.float64)
    v = np.asarray(spectrum["v"], dtype=np.float64)
    assert np.isclose(float(np.linalg.norm(u)), 1.0, atol=1e-12, rtol=0.0)
    assert np.isclose(float(np.linalg.norm(v)), 1.0, atol=1e-12, rtol=0.0)


def test_dense_conversion_avoids_copy_for_float64_ndarray() -> None:
    scorer = NBPerturbationScorer()
    H = _H()
    out = scorer._to_dense_copy(H)
    assert out is H


def test_dense_conversion_uses_toarray_for_sparse() -> None:
    scorer = NBPerturbationScorer()
    H = scipy.sparse.csr_matrix(_H())
    out = scorer._to_dense_copy(H)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_invalid_perturbation_returns_none() -> None:
    scorer = NBPerturbationScorer()
    H = _H()
    spectrum = scorer.compute_nb_spectrum(H)
    assert scorer.predict_swap_delta(H, (0, 0, 0, 1), spectrum) is None


def test_predict_swap_delta_is_deterministic() -> None:
    scorer = NBPerturbationScorer()
    H = _H()
    spectrum = scorer.compute_nb_spectrum(H)
    swap = (0, 0, 1, 2)
    p1 = scorer.predict_swap_delta(H, swap, spectrum)
    p2 = scorer.predict_swap_delta(H, swap, spectrum)
    assert p1 == p2
