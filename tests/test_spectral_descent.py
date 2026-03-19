import numpy as np

from qec.analysis.bethe_hessian import estimate_nishimori_temperature
from qec.analysis.eigenmode_mutation import build_bethe_hessian, extract_unstable_modes
from qec.discovery.spectral_descent_loop import spectral_descent


def _max_severity(H: np.ndarray) -> float:
    B, _ = build_bethe_hessian(H)
    modes = extract_unstable_modes(B, num_modes=10)
    if not modes:
        return 0.0
    return float(max(mode["severity"] for mode in modes))


def test_spectral_descent_nonincreasing_defect_severity() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    prev = _max_severity(H)
    for iter_cap in range(1, 9):
        H_after = spectral_descent(H, max_iter=iter_cap, scheduler="aggregate")
        new_severity = _max_severity(H_after.toarray())
        assert new_severity <= prev + 1e-12
        prev = new_severity



def test_spectral_descent_is_deterministic() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out1 = spectral_descent(H, max_iter=5, scheduler="strongest_first")
    out2 = spectral_descent(H, max_iter=5, scheduler="strongest_first")

    assert np.array_equal(out1.toarray(), out2.toarray())


def test_spectral_descent_dual_operator_is_deterministic() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out1 = spectral_descent(H, max_iter=5, scheduler="aggregate", dual_operator=True)
    out2 = spectral_descent(H, max_iter=5, scheduler="aggregate", dual_operator=True)

    assert np.array_equal(out1.toarray(), out2.toarray())


def test_nishimori_estimation_linear_fallback(monkeypatch) -> None:
    H = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    values = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _fake_lambda(_self, _A, _r):
        return float(next(values))

    monkeypatch.setattr("src.qec.analysis.bethe_hessian.BetheHessianAnalyzer._lambda_min_from_adjacency", _fake_lambda)
    r = estimate_nishimori_temperature(H)
    assert r > 0.0


def test_spectral_descent_dual_mode_dedup_is_deterministic() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out1 = spectral_descent(H, max_iter=4, scheduler="aggregate", dual_operator=True)
    out2 = spectral_descent(H, max_iter=4, scheduler="aggregate", dual_operator=True)
    assert np.array_equal(out1.toarray(), out2.toarray())
