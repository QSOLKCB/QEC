"""Deterministic tests for v111.0.0 spectral phase-boundary tracking."""

from __future__ import annotations

import pytest

from qec.analysis.spectral_phase_boundary import run_spectral_phase_boundary


REQUIRED_KEYS = {
    "chain_lengths",
    "multi_regime_result",
    "spectral_gap_curve",
    "spectral_gap_drift",
    "boundary_shift_score",
    "spectral_stability_score",
    "phase_boundary_class",
}

COMPONENT_KEYS = {
    "onset_component",
    "spectral_component",
    "component_balance_score",
    "dominant_component",
}


def _stub_multi_regime(**_: object) -> dict[str, object]:
    return {
        "chain_lengths": (4, 8, 16, 32, 64, 128),
        "scaling_result": {
            "logical_error_scaling_curve": [0.25, 0.125, 0.1, 0.05, 0.025, 0.0125],
            "scaling_exponent": 0.5,
            "surface_result": {"onset_drift_index": 0.3},
        },
        "regime_boundaries": (2, 4),
        "small_regime_exponent": 1.0,
        "medium_regime_exponent": 0.25,
        "large_regime_exponent": 0.75,
        "regime_transition_score": 0.2,
    }


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _stub_multi_regime)

    r1 = run_spectral_phase_boundary(diffusion_steps=4)
    r2 = run_spectral_phase_boundary(diffusion_steps=4)

    assert r1 == r2


def test_default_bit_stability_with_component_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _stub_multi_regime)

    default_out = run_spectral_phase_boundary(diffusion_steps=4)
    explicit_default_out = run_spectral_phase_boundary(diffusion_steps=4, return_components=False)

    assert default_out == explicit_default_out
    assert set(default_out.keys()) == REQUIRED_KEYS


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _stub_multi_regime)

    out = run_spectral_phase_boundary(diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert all(0.0 <= value <= 1.0 for value in out["spectral_gap_curve"])
    assert 0.0 <= out["spectral_gap_drift"] <= 1.0
    assert 0.0 <= out["boundary_shift_score"] <= 1.0
    assert 0.0 <= out["spectral_stability_score"] <= 1.0


def test_component_return_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _stub_multi_regime)

    out = run_spectral_phase_boundary(diffusion_steps=4, return_components=True)

    assert set(out.keys()) == REQUIRED_KEYS | COMPONENT_KEYS
    assert out["onset_component"] == pytest.approx(0.15)
    assert out["spectral_component"] == pytest.approx(0.007499999999999995)
    assert out["component_balance_score"] == pytest.approx(0.8575)
    assert out["dominant_component"] == "onset_dominant"


def test_component_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    def _high_values_stub(**_: object) -> dict[str, object]:
        out = _stub_multi_regime()
        out["scaling_result"]["surface_result"]["onset_drift_index"] = 2.0
        out["small_regime_exponent"] = 100.0
        out["medium_regime_exponent"] = 0.0
        out["large_regime_exponent"] = 100.0
        return out

    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _high_values_stub)

    out = run_spectral_phase_boundary(diffusion_steps=4, return_components=True)

    assert 0.0 <= out["onset_component"] <= 1.0
    assert 0.0 <= out["spectral_component"] <= 1.0
    assert 0.0 <= out["component_balance_score"] <= 1.0


def test_spectral_gap_extraction_uses_regime_exponents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _stub_multi_regime)

    out = run_spectral_phase_boundary(diffusion_steps=4)

    assert out["spectral_gap_curve"] == pytest.approx([
        0.8,
        0.8,
        0.8333333333333334,
        0.8333333333333334,
        0.875,
        0.875,
    ])


def test_drift_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _drift_stub(**_: object) -> dict[str, object]:
        out = _stub_multi_regime()
        out["small_regime_exponent"] = 0.0
        out["medium_regime_exponent"] = 1.0
        out["large_regime_exponent"] = 1.0
        return out

    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _drift_stub)

    out = run_spectral_phase_boundary(diffusion_steps=4)

    assert out["spectral_gap_curve"] == pytest.approx([0.6666666666666666] * 2 + [0.8] * 4)
    assert out["spectral_gap_drift"] == pytest.approx((0.8 - (2.0 / 3.0)) / 5.0)


def test_class_threshold_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    stable_payload = {
        "chain_lengths": (4, 8, 16),
        "scaling_result": {
            "logical_error_scaling_curve": [0.2, 0.1],
            "scaling_exponent": 0.0,
            "surface_result": {"onset_drift_index": 0.0},
        },
        "regime_boundaries": (1, 2),
        "small_regime_exponent": 0.0,
        "medium_regime_exponent": 100.0,
        "large_regime_exponent": 0.0,
        "regime_transition_score": 0.2,
    }
    monkeypatch.setattr(
        "qec.analysis.spectral_phase_boundary.run_multi_regime_scaling",
        lambda **_: stable_payload,
    )
    stable = run_spectral_phase_boundary(diffusion_steps=4)

    drifting_payload = {
        **stable_payload,
        "scaling_result": {
            **stable_payload["scaling_result"],
            "surface_result": {"onset_drift_index": 0.5},
        },
    }
    monkeypatch.setattr(
        "qec.analysis.spectral_phase_boundary.run_multi_regime_scaling",
        lambda **_: drifting_payload,
    )
    drifting = run_spectral_phase_boundary(diffusion_steps=4)

    critical_payload = {
        "chain_lengths": (4, 8, 16),
        "scaling_result": {
            "logical_error_scaling_curve": [0.2, 0.1, 0.05],
            "scaling_exponent": 0.0,
            "surface_result": {"onset_drift_index": 1.0},
        },
        "regime_boundaries": (1, 2),
        "small_regime_exponent": 0.0,
        "medium_regime_exponent": 100.0,
        "large_regime_exponent": 0.0,
        "regime_transition_score": 0.2,
    }
    monkeypatch.setattr(
        "qec.analysis.spectral_phase_boundary.run_multi_regime_scaling",
        lambda **_: critical_payload,
    )
    critical = run_spectral_phase_boundary(diffusion_steps=4)

    assert stable["phase_boundary_class"] == "stable_boundary"
    assert drifting["phase_boundary_class"] == "drifting_boundary"
    assert critical["phase_boundary_class"] == "critical_boundary"


def test_non_monotonic_spectral_gap_drift_regression() -> None:
    from qec.analysis.spectral_phase_boundary import _spectral_gap_drift

    drift = _spectral_gap_drift([0.7, 0.9, 0.6, 0.6, 0.8])

    assert drift == pytest.approx(0.175)


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def _short_stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5,),
            "scaling_result": {
                "logical_error_scaling_curve": [0.1],
                "scaling_exponent": 0.5,
                "surface_result": {"onset_drift_index": 0.9},
            },
            "regime_boundaries": (1, 1),
            "small_regime_exponent": 0.5,
            "medium_regime_exponent": 0.0,
            "large_regime_exponent": 0.0,
            "regime_transition_score": 0.9,
        }

    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _short_stub)

    out = run_spectral_phase_boundary(chain_lengths=[5], diffusion_steps=4)

    assert out["chain_lengths"] == (5,)
    assert out["spectral_gap_curve"] == pytest.approx([1.0])
    assert out["spectral_gap_drift"] == 0.0
    assert out["boundary_shift_score"] == pytest.approx(0.45)


def test_balanced_component_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def _balanced_stub(**_: object) -> dict[str, object]:
        out = _stub_multi_regime()
        out["scaling_result"]["surface_result"]["onset_drift_index"] = 0.2
        out["small_regime_exponent"] = 0.0
        out["medium_regime_exponent"] = 1.0
        out["large_regime_exponent"] = 1.0
        return out

    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _balanced_stub)

    out = run_spectral_phase_boundary(diffusion_steps=4, return_components=True)

    assert out["onset_component"] == pytest.approx(0.1)
    assert out["spectral_component"] == pytest.approx(0.01333333333333333)
    assert out["dominant_component"] == "balanced_components"


def test_short_chain_edge_case_components(monkeypatch: pytest.MonkeyPatch) -> None:
    def _short_stub(**_: object) -> dict[str, object]:
        return {
            "chain_lengths": (5,),
            "scaling_result": {
                "logical_error_scaling_curve": [0.1],
                "scaling_exponent": 0.5,
                "surface_result": {"onset_drift_index": 0.9},
            },
            "regime_boundaries": (1, 1),
            "small_regime_exponent": 0.5,
            "medium_regime_exponent": 0.0,
            "large_regime_exponent": 0.0,
            "regime_transition_score": 0.9,
        }

    monkeypatch.setattr("qec.analysis.spectral_phase_boundary.run_multi_regime_scaling", _short_stub)

    out = run_spectral_phase_boundary(chain_lengths=[5], diffusion_steps=4, return_components=True)

    assert out["onset_component"] == pytest.approx(0.45)
    assert out["spectral_component"] == 0.0
    assert out["component_balance_score"] == pytest.approx(0.55)
    assert out["dominant_component"] == "onset_dominant"
