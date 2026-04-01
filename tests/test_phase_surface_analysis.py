"""Deterministic tests for v110.0.0 phase-surface analysis."""

from __future__ import annotations

import pytest

from qec.analysis.phase_surface_analysis import (
    DEFAULT_CHAIN_LENGTH_SWEEP,
    run_phase_surface_analysis,
)


REQUIRED_KEYS = {
    "chain_lengths",
    "threshold_values",
    "surface_results",
    "onset_drift_index",
    "chain_stability_curve",
    "surface_stability_score",
    "surface_class",
}


def _phase_map_stub(*, chain_length: int, threshold_values: list[float] | None = None, **_: object) -> dict[str, object]:
    ordered_threshold_values = [0.25, 0.5, 0.75] if threshold_values is None else [float(value) for value in threshold_values]
    onset_by_chain = {5: 0, 9: 2, 17: 4}
    onset_index = onset_by_chain.get(int(chain_length), 1)
    phase_results = [
        {
            "onset_index": onset_index,
            "sweep_result": {"perturbation_values": [0.25, 0.5, 1.0, 1.5, 2.0]},
        }
        for _ in ordered_threshold_values
    ]
    return {
        "chain_length": int(chain_length),
        "threshold_values": tuple(ordered_threshold_values),
        "phase_results": phase_results,
        "threshold_stability_score": 0.8,
    }


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _phase_map_stub)

    r1 = run_phase_surface_analysis(diffusion_steps=4)
    r2 = run_phase_surface_analysis(diffusion_steps=4)

    assert r1 == r2


def test_chain_ordering_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_phase_surface_analysis(chain_lengths=[], diffusion_steps=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_phase_surface_analysis(chain_lengths=[9, 5, 17], diffusion_steps=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_phase_surface_analysis(chain_lengths=[5, 5, 17], diffusion_steps=4)


def test_threshold_values_must_match_across_chain_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mismatch_stub(*, chain_length: int, **_: object) -> dict[str, object]:
        threshold_values = (0.25, 0.5, 0.75) if int(chain_length) == 5 else (0.3, 0.6, 0.9)
        return {
            "chain_length": int(chain_length),
            "threshold_values": threshold_values,
            "phase_results": [{"onset_index": 0, "sweep_result": {"perturbation_values": [0.25, 0.5]}}],
            "threshold_stability_score": 1.0,
        }

    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _mismatch_stub)
    with pytest.raises(ValueError, match="all threshold_values must match across chain lengths"):
        run_phase_surface_analysis(chain_lengths=[5, 9], diffusion_steps=4)


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _phase_map_stub)

    out = run_phase_surface_analysis(diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert out["chain_lengths"] == DEFAULT_CHAIN_LENGTH_SWEEP
    assert 0.0 <= out["onset_drift_index"] <= 1.0
    assert 0.0 <= out["surface_stability_score"] <= 1.0


def test_onset_drift_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _phase_map_stub)

    out = run_phase_surface_analysis(diffusion_steps=4)

    assert out["onset_drift_index"] == pytest.approx(0.5)


def test_stable_surface_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stable_stub(*, chain_length: int, threshold_values: list[float] | None = None, **_: object) -> dict[str, object]:
        ordered_threshold_values = [0.25, 0.5, 0.75] if threshold_values is None else [float(value) for value in threshold_values]
        phase_results = [
            {
                "onset_index": 1,
                "sweep_result": {"perturbation_values": [0.25, 0.5, 1.0]},
            }
            for _ in ordered_threshold_values
        ]
        return {
            "chain_length": int(chain_length),
            "threshold_values": tuple(ordered_threshold_values),
            "phase_results": phase_results,
            "threshold_stability_score": 1.0,
        }

    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _stable_stub)

    out = run_phase_surface_analysis(diffusion_steps=4)

    assert out["onset_drift_index"] == 0.0
    assert out["surface_stability_score"] == 1.0
    assert out["surface_class"] == "stable_surface"


def test_mixed_and_unstable_surface_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mixed_stub(*, chain_length: int, threshold_values: list[float] | None = None, **_: object) -> dict[str, object]:
        ordered_threshold_values = [0.25, 0.5, 0.75] if threshold_values is None else [float(value) for value in threshold_values]
        onset_map = {5: 0, 9: 0, 17: 1}
        phase_results = [
            {
                "onset_index": onset_map[int(chain_length)],
                "sweep_result": {"perturbation_values": [0.25, 0.5, 1.0]},
            }
            for _ in ordered_threshold_values
        ]
        return {
            "chain_length": int(chain_length),
            "threshold_values": tuple(ordered_threshold_values),
            "phase_results": phase_results,
            "threshold_stability_score": 0.6,
        }

    def _unstable_stub(*, chain_length: int, threshold_values: list[float] | None = None, **_: object) -> dict[str, object]:
        ordered_threshold_values = [0.25, 0.5, 0.75] if threshold_values is None else [float(value) for value in threshold_values]
        onset_map = {5: 0, 9: 2, 17: 0}
        phase_results = [
            {
                "onset_index": onset_map[int(chain_length)],
                "sweep_result": {"perturbation_values": [0.25, 0.5, 1.0]},
            }
            for _ in ordered_threshold_values
        ]
        return {
            "chain_length": int(chain_length),
            "threshold_values": tuple(ordered_threshold_values),
            "phase_results": phase_results,
            "threshold_stability_score": 0.4,
        }

    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _mixed_stub)
    mixed_out = run_phase_surface_analysis(diffusion_steps=4)
    assert mixed_out["surface_class"] == "mixed_surface"

    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _unstable_stub)
    unstable_out = run_phase_surface_analysis(diffusion_steps=4)
    assert unstable_out["surface_class"] == "unstable_surface"


def test_short_chain_edge_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    def _short_stub(*, chain_length: int, threshold_values: list[float] | None = None, **_: object) -> dict[str, object]:
        ordered_threshold_values = [0.25, 0.5, 0.75] if threshold_values is None else [float(value) for value in threshold_values]
        phase_results = [
            {
                "onset_index": None,
                "sweep_result": {"perturbation_values": [0.25]},
            }
            for _ in ordered_threshold_values
        ]
        return {
            "chain_length": int(chain_length),
            "threshold_values": tuple(ordered_threshold_values),
            "phase_results": phase_results,
            "threshold_stability_score": 1.0,
        }

    monkeypatch.setattr("qec.analysis.phase_surface_analysis.run_threshold_phase_map", _short_stub)

    out = run_phase_surface_analysis(chain_lengths=[3], diffusion_steps=4)

    assert out["chain_lengths"] == (3,)
    assert out["onset_drift_index"] == 0.0
    assert out["chain_stability_curve"] == [1.0]
