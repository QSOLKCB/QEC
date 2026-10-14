"""Control restart randomness without changing the raw spectral boundary."""

import numpy as np
import pytest
import scipy.sparse.linalg

from qec.analysis.nonbacktracking_flow import (
    NBFlowConfig,
    NonBacktrackingEigenvectorFlowAnalyzer,
    _eigs_with_local_rng,
)


def test_older_eigs_signature_receives_no_rng_keyword(monkeypatch):
    calls = []
    values = np.array([1j])
    vectors = np.ones((2, 1))

    def legacy_eigs(B, *, k, v0):
        calls.append((B, k, v0))
        return values, vectors

    monkeypatch.setattr(scipy.sparse.linalg, "eigs", legacy_eigs)
    B = scipy.sparse.eye(2)
    v0 = np.ones(2)
    actual_values, actual_vectors = _eigs_with_local_rng(B, k=1, v0=v0)
    assert len(calls) == 1
    assert calls[0][0] is B and calls[0][1] == 1 and calls[0][2] is v0
    assert actual_values is values and actual_vectors is vectors


def test_modern_eigs_receives_fresh_pcg64_seeded_state(monkeypatch):
    samples = []
    generators = []

    def modern_eigs(B, *, k, v0, rng=None):
        generators.append(rng)
        samples.append(rng.uniform(size=12))
        return np.array([1j]), np.ones((2, 1))

    monkeypatch.setattr(scipy.sparse.linalg, "eigs", modern_eigs)
    state = np.random.get_state()
    _eigs_with_local_rng(scipy.sparse.eye(2), k=1, v0=np.ones(2))
    _eigs_with_local_rng(scipy.sparse.eye(2), k=1, v0=np.ones(2))
    after = np.random.get_state()
    assert generators[0] is not generators[1]
    assert type(generators[0].bit_generator) is np.random.PCG64
    expected = np.random.Generator(np.random.PCG64(0)).uniform(size=12)
    np.testing.assert_array_equal(samples[0], expected)
    np.testing.assert_array_equal(samples[1], expected)
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("supports_rng", [False, True])
def test_all_three_sparse_solve_sites_use_compatible_policy(monkeypatch, supports_rng):
    calls = []
    generators = []
    samples = []

    def result(B, k, v0):
        calls.append(B.copy())
        np.testing.assert_array_equal(v0, np.ones(B.shape[0]))
        return np.full(k, 2.0, dtype=np.complex128), np.ones((B.shape[0], k), dtype=np.complex128)

    def legacy_eigs(B, *, k, which, v0, maxiter, tol):
        return result(B, k, v0)

    def modern_eigs(B, *, k, which, v0, maxiter, tol, rng=None):
        assert isinstance(rng, np.random.Generator)
        generators.append(rng)
        samples.append(rng.uniform(size=12))
        return result(B, k, v0)

    monkeypatch.setattr(scipy.sparse.linalg, "eigs", modern_eigs if supports_rng else legacy_eigs)
    analyzer = NonBacktrackingEigenvectorFlowAnalyzer(NBFlowConfig(
        num_nb_eigenvalues=1, bulk_radius_mode="fixed", bulk_radius_value=1.0,
        use_left_right_pairing=True,
    ))
    H = np.ones((2, 2), dtype=np.float64)
    assert analyzer.compute_modes(H)[0].size == 1
    assert len(analyzer.build_flow_field(H).selected_modes) == 1
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0].toarray(), calls[1].toarray())
    np.testing.assert_array_equal(calls[1].T.toarray(), calls[2].toarray())
    if supports_rng:
        assert len({id(rng) for rng in generators}) == 3
        expected = np.random.Generator(np.random.PCG64(0)).uniform(size=12)
        for sample in samples:
            np.testing.assert_array_equal(sample, expected)


@pytest.mark.parametrize("magnitude, expected", [
    (np.nextafter(1.0, 0.0), 0), (1.0, 0), (np.nextafter(1.0, np.inf), 1),
])
def test_raw_strict_boundary_comparison_is_not_redefined(monkeypatch, magnitude, expected):
    analyzer = NonBacktrackingEigenvectorFlowAnalyzer(NBFlowConfig(
        num_nb_eigenvalues=1, bulk_radius_mode="fixed", bulk_radius_value=1.0,
        use_left_right_pairing=False,
    ))
    H = np.ones((2, 2), dtype=np.float64)

    def solver(B, *, rng=None, **kwargs):
        return np.array([complex(magnitude)]), np.eye(8, dtype=np.complex128)[:, :1]

    monkeypatch.setattr(scipy.sparse.linalg, "eigs", solver)
    values, _ = analyzer.compute_modes(H)
    assert values.size == expected
    assert len(analyzer.build_flow_field(H).selected_modes) == expected
