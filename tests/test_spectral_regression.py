from __future__ import annotations

import json

from src.qec.analysis.spectral_regression import SpectralThresholdModel, load_training_dataset


def _dataset() -> list[dict[str, float]]:
    return [
        {
            "nb_spectral_radius": 0.7,
            "bethe_negative_mass": 2.0,
            "flow_ipr": 0.1,
            "spectral_entropy": 0.6,
            "trap_similarity": 0.2,
            "threshold": 0.045,
        },
        {
            "nb_spectral_radius": 0.8,
            "bethe_negative_mass": 1.0,
            "flow_ipr": 0.2,
            "spectral_entropy": 0.5,
            "trap_similarity": 0.3,
            "threshold": 0.035,
        },
    ]


def test_regression_fit_predict_is_deterministic() -> None:
    model_a = SpectralThresholdModel()
    model_b = SpectralThresholdModel()

    model_a.fit(_dataset())
    model_b.fit(_dataset())

    assert model_a.coefficients.tolist() == model_b.coefficients.tolist()

    features = _dataset()[0]
    assert model_a.predict(features) == model_b.predict(features)


def test_model_artifact_serialization_and_loading(tmp_path) -> None:
    model = SpectralThresholdModel()
    model.fit(_dataset())

    path = tmp_path / "spectral_threshold_model.json"
    model.save_model(path)
    first = path.read_text(encoding="utf-8")

    loaded = SpectralThresholdModel.load_model(path)
    loaded.save_model(path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first) == json.loads(second)


def test_load_training_dataset(tmp_path) -> None:
    exp1 = tmp_path / "exp_a"
    exp2 = tmp_path / "exp_b"
    exp1.mkdir()
    exp2.mkdir()

    (exp1 / "results.json").write_text(
        json.dumps(
            {
                "threshold": 0.04,
                "spectral_metrics": {
                    "nb_spectral_radius": 0.9,
                    "bethe_hessian_negative_modes": 3,
                    "ipr_localization": 0.2,
                    "spectral_entropy": 0.5,
                    "trap_similarity": 0.1,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (exp2 / "results.json").write_text(
        json.dumps(
            {
                "threshold": 0.03,
                "spectral_metrics": {
                    "nb_spectral_radius": 1.1,
                    "bethe_negative_mass": 2.0,
                    "flow_ipr": 0.3,
                    "spectral_entropy": 0.4,
                    "trap_similarity": 0.2,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_training_dataset(tmp_path)
    assert len(rows) == 2
    assert rows[0]["bethe_negative_mass"] == 3.0
    assert rows[0]["flow_ipr"] == 0.2
