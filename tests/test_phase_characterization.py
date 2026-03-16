from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.phase_characterization import (
    build_phase_profile,
    classify_phase,
    compute_phase_metrics,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_compute_phase_metrics_deterministic_float64_values() -> None:
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    spectrum = np.array([1.2, -0.3, 0.7], dtype=np.float64)
    decoder_stats = {"instability_score": 0.25, "trapping_set_count": 2}

    m1 = compute_phase_metrics(H, spectrum, decoder_stats)
    m2 = compute_phase_metrics(H, spectrum, decoder_stats)

    assert m1 == m2
    assert sorted(m1.keys()) == sorted(
        [
            "bp_stability_score",
            "trapping_density",
            "estimated_threshold",
            "spectral_radius",
            "bethe_hessian_min_eigenvalue",
        ]
    )
    for value in m1.values():
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_classify_phase_deterministic_thresholds() -> None:
    stable = classify_phase(
        {
            "bp_stability_score": 0.9,
            "trapping_density": 0.1,
            "estimated_threshold": 0.6,
            "spectral_radius": 1.05,
            "bethe_hessian_min_eigenvalue": 0.05,
        }
    )
    trapping = classify_phase(
        {
            "bp_stability_score": 0.95,
            "trapping_density": 0.5,
            "estimated_threshold": 0.8,
            "spectral_radius": 0.8,
            "bethe_hessian_min_eigenvalue": 0.2,
        }
    )

    assert stable == {"phase_label": "stable_bp_phase"}
    assert trapping == {"phase_label": "trapping_dominated_phase"}


def test_build_phase_profile_deterministic_order() -> None:
    metrics = {
        "bp_stability_score": 0.7,
        "trapping_density": 0.2,
        "estimated_threshold": 0.5,
        "spectral_radius": 1.1,
        "bethe_hessian_min_eigenvalue": -0.1,
    }
    profile = build_phase_profile(3, metrics, {"phase_label": "fragile_bp_phase"})

    assert list(profile.keys()) == [
        "phase_id",
        "phase_label",
        "bp_stability_score",
        "trapping_density",
        "estimated_threshold",
        "spectral_radius",
        "bethe_min_eigenvalue",
    ]
    assert profile["phase_id"] == 3
    assert profile["phase_label"] == "fragile_bp_phase"


def test_engine_phase_characterization_reproducible_opt_in() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_basin_hopping=True,
        basin_detection_interval=1,
        enable_phase_novelty_discovery=True,
        phase_novelty_interval=1,
        phase_novelty_threshold=-1.0,
        enable_phase_characterization=True,
        phase_characterization_interval=1,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_basin_hopping=True,
        basin_detection_interval=1,
        enable_phase_novelty_discovery=True,
        phase_novelty_interval=1,
        phase_novelty_threshold=-1.0,
        enable_phase_characterization=True,
        phase_characterization_interval=1,
    )

    assert json.dumps(r1.get("phase_profiles", []), sort_keys=True) == json.dumps(r2.get("phase_profiles", []), sort_keys=True)
    assert json.dumps(r1.get("phase_characterization_metrics", []), sort_keys=True) == json.dumps(r2.get("phase_characterization_metrics", []), sort_keys=True)
    assert "phase_profiles" in r1
    assert "phase_characterization_metrics" in r1
    summary = r1["generation_summaries"][-1]
    assert "phase_characterized" in summary
    if summary["phase_characterized"]:
        assert "phase_label" in summary
