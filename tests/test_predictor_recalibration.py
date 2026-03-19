from __future__ import annotations

import json

import numpy as np

from qec.analysis.predictor_recalibration import apply_recalibration, compute_recalibration_bias
from qec.discovery.threshold_search import PhaseDiagramOrchestrator, SpectralSearchConfig, run_spectral_threshold_search
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_compute_recalibration_bias_is_deterministic() -> None:
    predictions = [0.1, 0.2, 0.3]
    actuals = [0.11, 0.21, 0.32]

    first = compute_recalibration_bias(predictions, actuals)
    second = compute_recalibration_bias(predictions, actuals)

    assert first == second
    assert first == 0.013333333333


def test_apply_recalibration_rounding_stability() -> None:
    corrected = apply_recalibration(0.1234567891234, 0.0000000000006)
    assert corrected == 0.123456789124


def test_recalibration_artifact_serialization_and_correction(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    monkeypatch.setattr(
        PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.2}
        },
    )

    cfg = SpectralSearchConfig(
        iterations=3,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_predictor_recalibration=True,
        recalibration_interval=1,
        enable_beam_mutations=False,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "predictor_recalibration.json").read_text(encoding="utf-8"))
    assert payload == {"bias_correction": 0.1, "samples": 3}

    metrics_payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    candidates = [c for c in metrics_payload["candidates"] if not c["rejected"]]

    assert candidates[0]["predicted_threshold"] == 0.0
    assert candidates[1]["predicted_threshold"] == 0.2
    assert candidates[2]["predicted_threshold"] == 0.1
    assert candidates[1]["predictor_bias"] == 0.2
    assert candidates[2]["predictor_bias"] == 0.1
