from __future__ import annotations

from qec.analysis.threshold_predictor import ThresholdPrediction, predict_threshold_quality


def test_predictor_is_deterministic() -> None:
    a = predict_threshold_quality(0.8, 2.0, 0.2, 0.3, 0.1)
    b = predict_threshold_quality(0.8, 2.0, 0.2, 0.3, 0.1)
    assert a == b


def test_predictor_monotonic_in_spectral_radius() -> None:
    low = predict_threshold_quality(0.5, 2.0, 0.2, 0.3, 0.1)
    high = predict_threshold_quality(1.0, 2.0, 0.2, 0.3, 0.1)
    assert low.predicted_threshold >= high.predicted_threshold
    assert low.score >= high.score


def test_predictor_rounding_is_stable() -> None:
    out = predict_threshold_quality(0.123456789123456, 0.0, 0.0, 0.0, 0.0)
    assert out.score == round(out.score, 12)
    assert out.predicted_threshold == round(out.predicted_threshold, 12)


def test_prediction_dataclass_equality() -> None:
    a = ThresholdPrediction(predicted_threshold=0.1, score=0.2)
    b = ThresholdPrediction(predicted_threshold=0.1, score=0.2)
    assert a == b
