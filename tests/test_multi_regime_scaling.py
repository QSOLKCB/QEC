"""Deterministic tests for v110.2.0 multi-regime finite-size scaling analysis."""

from __future__ import annotations

import pytest

from qec.analysis.multi_regime_scaling import run_multi_regime_scaling


REQUIRED_KEYS = {
    "chain_lengths",
    "scaling_result",
    "regime_boundaries",
    "small_regime_exponent",
    "medium_regime_exponent",
    "large_regime_exponent",
    "regime_transition_score",
    "scaling_regime_class",
}


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16, 32, 64, 128),
            "logical_error_scaling_curve": [0.5, 0.25, 0.125, 0.07, 0.04, 0.02],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    r1 = run_multi_regime_scaling(diffusion_steps=4)
    r2 = run_multi_regime_scaling(diffusion_steps=4)

    assert r1 == r2


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16, 32, 64, 128),
            "logical_error_scaling_curve": [0.9, 0.9, 0.01, 0.01, 1e-9, 1e-9],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    out = run_multi_regime_scaling(diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert 0.0 <= out["small_regime_exponent"] <= 10.0
    assert 0.0 <= out["medium_regime_exponent"] <= 10.0
    assert 0.0 <= out["large_regime_exponent"] <= 10.0
    assert 0.0 <= out["regime_transition_score"] <= 1.0


def test_exponent_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16, 32, 64, 128),
            "logical_error_scaling_curve": [0.25, 0.125, 0.01, 0.01, 0.02, 0.01],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    out = run_multi_regime_scaling(diffusion_steps=4)

    assert out["regime_boundaries"] == (2, 4)
    assert out["small_regime_exponent"] == pytest.approx(1.0)
    assert out["medium_regime_exponent"] == pytest.approx(0.0)
    assert out["large_regime_exponent"] == pytest.approx(1.0)


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (3,),
            "logical_error_scaling_curve": [0.2],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    out = run_multi_regime_scaling(chain_lengths=[3], diffusion_steps=4)

    assert out["chain_lengths"] == (3,)
    assert out["regime_boundaries"] == (1, 1)
    assert out["small_regime_exponent"] == 0.0
    assert out["medium_regime_exponent"] == 0.0
    assert out["large_regime_exponent"] == 0.0


def test_regime_transition_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16, 32, 64, 128),
            "logical_error_scaling_curve": [0.25, 0.125, 0.01, 0.01, 0.02, 0.01],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    out = run_multi_regime_scaling(diffusion_steps=4)

    assert out["regime_transition_score"] == pytest.approx(0.1)


def test_scaling_curve_length_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16),
            "logical_error_scaling_curve": [0.25, 0.125],
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    with pytest.raises(ValueError, match="logical_error_scaling_curve must match chain_lengths length"):
        run_multi_regime_scaling(diffusion_steps=4)


@pytest.mark.parametrize(
    ("curve", "expected_class"),
    [
        ([0.25, 0.125, 0.01, 0.01, 0.02, 0.01], "uniform_scaling"),
        ([0.25, 0.125, 0.01, 0.00125, 0.02, 0.01], "transition_scaling"),
        ([0.25, 0.125, 0.9, 1e-12, 0.2, 0.2], "regime_shifted_scaling"),
    ],
)
def test_class_threshold_behavior(
    monkeypatch: pytest.MonkeyPatch,
    curve: list[float],
    expected_class: str,
) -> None:
    def _stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (4, 8, 16, 32, 64, 128),
            "logical_error_scaling_curve": curve,
        }

    monkeypatch.setattr("qec.analysis.multi_regime_scaling.run_finite_size_scaling", _stub)

    out = run_multi_regime_scaling(diffusion_steps=4)

    assert out["scaling_regime_class"] == expected_class
