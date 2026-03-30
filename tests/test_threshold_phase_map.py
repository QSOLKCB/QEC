"""Deterministic tests for v109.2.0 threshold phase-map analysis."""

from __future__ import annotations

import pytest

from qec.analysis.threshold_phase_map import (
    DEFAULT_ONSET_THRESHOLD_SWEEP,
    _phase_map_class,
    run_threshold_phase_map,
)


REQUIRED_KEYS = {
    "chain_length",
    "threshold_values",
    "phase_results",
    "onset_curve",
    "threshold_stability_score",
    "transition_consistency_score",
    "phase_map_class",
}


def test_exact_determinism() -> None:
    r1 = run_threshold_phase_map(
        chain_length=9,
        threshold_values=[0.25, 0.5, 0.75],
        perturbation_values=[0.25, 0.5, 1.0, 2.0],
        diffusion_steps=4,
    )
    r2 = run_threshold_phase_map(
        chain_length=9,
        threshold_values=[0.25, 0.5, 0.75],
        perturbation_values=[0.25, 0.5, 1.0, 2.0],
        diffusion_steps=4,
    )
    assert r1 == r2


def test_threshold_ordering_validation() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        run_threshold_phase_map(chain_length=7, threshold_values=[0.5, 0.25, 0.75], diffusion_steps=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_threshold_phase_map(chain_length=7, threshold_values=[0.5, 0.5, 0.75], diffusion_steps=4)


def test_boundedness() -> None:
    out = run_threshold_phase_map(chain_length=7, diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert out["threshold_values"] == DEFAULT_ONSET_THRESHOLD_SWEEP
    assert 0.0 <= out["threshold_stability_score"] <= 1.0
    assert 0.0 <= out["transition_consistency_score"] <= 1.0


def test_stable_threshold_region_case(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _stub(**_: object) -> dict[str, object]:
        calls["count"] += 1
        return {
            "onset_index": 2,
            "onset_perturbation": 1.0,
            "phase_transition_class": "sharp_transition",
        }

    monkeypatch.setattr("qec.analysis.threshold_phase_map.run_phase_transition_analysis", _stub)

    out = run_threshold_phase_map(chain_length=11, threshold_values=[0.25, 0.5, 0.75], diffusion_steps=4)

    assert calls["count"] == 3
    assert out["threshold_stability_score"] == 1.0
    assert out["transition_consistency_score"] == 1.0
    assert out["phase_map_class"] == "stable_threshold_region"


def test_mixed_transition_case(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = [
        {"onset_index": 1, "onset_perturbation": 0.5, "phase_transition_class": "sharp_transition"},
        {"onset_index": 1, "onset_perturbation": 0.5, "phase_transition_class": "gradual_transition"},
        {"onset_index": 1, "onset_perturbation": 0.5, "phase_transition_class": "sharp_transition"},
    ]
    calls = {"count": 0}

    def _stub(**_: object) -> dict[str, object]:
        index = calls["count"]
        calls["count"] += 1
        return sequence[index]

    monkeypatch.setattr("qec.analysis.threshold_phase_map.run_phase_transition_analysis", _stub)

    out = run_threshold_phase_map(chain_length=11, threshold_values=[0.25, 0.5, 0.75], diffusion_steps=4)

    assert out["threshold_stability_score"] == 1.0
    assert out["transition_consistency_score"] == 0.0
    assert out["phase_map_class"] == "mixed_transition_region"


def test_short_chain_edge_case() -> None:
    out = run_threshold_phase_map(chain_length=3, diffusion_steps=4)

    assert out["chain_length"] == 3
    assert len(out["threshold_values"]) == 3
    assert len(out["phase_results"]) == 3
    assert len(out["onset_curve"]) == 3


def test_phase_map_class_unstable_threshold_region() -> None:
    assert _phase_map_class(threshold_stability_score=0.49, transition_consistency_score=1.0) == "unstable_threshold_region"
