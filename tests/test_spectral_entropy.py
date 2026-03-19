from __future__ import annotations

import numpy as np

from qec.analysis.spectral_entropy import spectral_entropy


def test_entropy_determinism_same_input_same_output() -> None:
    eigvals = np.array([3.0 + 0.0j, -1.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
    e1 = spectral_entropy(eigvals)
    e2 = spectral_entropy(eigvals)
    assert e1 == e2


def test_uniform_spectrum_has_higher_entropy_than_peaked() -> None:
    uniform = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    peaked = np.array([4.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert spectral_entropy(uniform) > spectral_entropy(peaked)


def test_zero_spectrum_entropy_is_zero() -> None:
    eigvals = np.zeros(3, dtype=np.float64)
    assert spectral_entropy(eigvals) == 0.0


def test_rounding_determinism_precision_control() -> None:
    eigvals = np.array([np.pi, np.e, np.sqrt(2.0)], dtype=np.float64)
    vals = [spectral_entropy(eigvals, precision=8) for _ in range(5)]
    assert all(v == vals[0] for v in vals)
    assert vals[0] == round(vals[0], 8)
