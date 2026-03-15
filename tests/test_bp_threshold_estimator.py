from __future__ import annotations

import json

from src.qec.experiments.bp_threshold_estimator import BPThresholdEstimator


_PHASE_DIAGRAM = {
    "experiment": "bp-threshold",
    "x_axis": "physical_error_rate",
    "x_values": [0.02, 0.03, 0.04, 0.05],
    "y_axis": "decoder_iterations",
    "grid": [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, -1],
        [1, 1, 1, -1, -1],
        [1, 1, -1, -1, -1],
    ],
}


def test_deterministic_estimation() -> None:
    estimator = BPThresholdEstimator()
    rates_a = estimator.compute_success_rates(_PHASE_DIAGRAM)
    rates_b = estimator.compute_success_rates(_PHASE_DIAGRAM)
    assert rates_a == rates_b

    threshold_a = estimator.estimate_threshold(rates_a)
    threshold_b = estimator.estimate_threshold(rates_b)
    assert threshold_a == threshold_b


def test_expected_threshold_detection() -> None:
    estimator = BPThresholdEstimator()
    rates = estimator.compute_success_rates(_PHASE_DIAGRAM)
    threshold = estimator.estimate_threshold(rates)

    assert threshold["method"] == "50_percent_crossing"
    assert abs(float(threshold["threshold"]) - 0.045) < 1e-12


def test_artifact_serialization_is_deterministic(tmp_path) -> None:
    estimator = BPThresholdEstimator()
    rates = estimator.compute_success_rates(_PHASE_DIAGRAM)
    threshold = estimator.estimate_threshold(rates)

    path = tmp_path / "threshold_estimate.json"
    payload_a = estimator.write_threshold_artifact(
        path,
        experiment="bp-threshold",
        success_rates=rates,
        threshold=threshold,
    )
    text_a = path.read_text(encoding="utf-8")

    payload_b = estimator.write_threshold_artifact(
        path,
        experiment="bp-threshold",
        success_rates=rates,
        threshold=threshold,
    )
    text_b = path.read_text(encoding="utf-8")

    assert payload_a == payload_b
    assert text_a == text_b

    loaded = json.loads(text_a)
    assert loaded["experiment"] == "bp-threshold"
    assert loaded["method"] == "50_percent_crossing"
    assert abs(float(loaded["estimated_threshold"]) - 0.045) < 1e-12
