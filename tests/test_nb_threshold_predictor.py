from __future__ import annotations

from qec.analysis.nb_threshold_predictor import predict_threshold_from_spectrum


def test_predictor_is_deterministic_and_bounded() -> None:
    metrics = {
        "nb_spectral_radius": 0.81,
        "bethe_negative_mass": 0.2,
        "flow_ipr": 0.1,
        "spectral_entropy": 0.7,
        "trap_similarity": 0.3,
    }

    p1 = predict_threshold_from_spectrum(metrics)
    p2 = predict_threshold_from_spectrum(metrics)

    assert p1 == p2
    assert 0.0 <= p1["predicted_threshold"] <= 0.1
    assert isinstance(p1["prediction_score"], float)
