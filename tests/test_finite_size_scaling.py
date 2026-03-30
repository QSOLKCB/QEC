"""Deterministic tests for v110.1.0 finite-size scaling analysis."""

from __future__ import annotations

import pytest

from qec.analysis.finite_size_scaling import run_finite_size_scaling


REQUIRED_KEYS = {
    "chain_lengths",
    "surface_result",
    "logical_error_scaling_curve",
    "pseudo_threshold_estimate",
    "critical_threshold_estimate",
    "scaling_exponent",
    "fit_quality_score",
    "scaling_class",
}


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.40, 0.55, 0.70],
            "surface_stability_score": 0.55,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    r1 = run_finite_size_scaling(diffusion_steps=4)
    r2 = run_finite_size_scaling(diffusion_steps=4)

    assert r1 == r2


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.25, 1.20, -0.5],
            "surface_stability_score": 0.40,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert all(0.0 <= value <= 1.0 for value in out["logical_error_scaling_curve"])
    assert out["pseudo_threshold_estimate"] is None or 0.0 <= out["pseudo_threshold_estimate"] <= 1.0
    assert out["critical_threshold_estimate"] is None or 0.0 <= out["critical_threshold_estimate"] <= 1.0
    assert 0.0 <= out["scaling_exponent"] <= 10.0
    assert 0.0 <= out["fit_quality_score"] <= 1.0


def test_pseudo_threshold_crossing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.20, 0.55, 0.80],
            "surface_stability_score": 0.50,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert out["logical_error_scaling_curve"] == [0.8, 0.44999999999999996, 0.19999999999999996]
    assert out["pseudo_threshold_estimate"] == pytest.approx((9.0 - 5.0) / (17.0 - 5.0))


def test_critical_threshold_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.10, 0.30, 0.55],
            "surface_stability_score": 0.35,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert out["critical_threshold_estimate"] is not None
    assert out["scaling_exponent"] > 0.0


def test_loglog_fit_executes_without_typeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.10, 0.30, 0.55],
            "surface_stability_score": 0.35,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert out["critical_threshold_estimate"] is not None


def test_surface_chain_length_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 11, 17),
            "chain_stability_curve": [0.10, 0.30, 0.55],
            "surface_stability_score": 0.35,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    with pytest.raises(ValueError, match="surface_result chain_lengths must match scaling input"):
        run_finite_size_scaling(chain_lengths=[5, 9, 17], diffusion_steps=4)


def test_fit_quality_score_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.42, 0.58, 0.73],
            "surface_stability_score": 0.57,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert 0.0 <= out["fit_quality_score"] <= 1.0


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (3,),
            "chain_stability_curve": [0.75],
            "surface_stability_score": 0.75,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(chain_lengths=[3], diffusion_steps=4)

    assert out["chain_lengths"] == (3,)
    assert out["critical_threshold_estimate"] == out["pseudo_threshold_estimate"]
    assert out["scaling_exponent"] == 0.0


def test_no_crossing_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5, 9, 17),
            "chain_stability_curve": [0.10, 0.20, 0.30],
            "surface_stability_score": 0.20,
        }

    monkeypatch.setattr("qec.analysis.finite_size_scaling.run_phase_surface_analysis", _stub)

    out = run_finite_size_scaling(diffusion_steps=4)

    assert out["pseudo_threshold_estimate"] is None
    assert out["scaling_class"] == "under_threshold"
