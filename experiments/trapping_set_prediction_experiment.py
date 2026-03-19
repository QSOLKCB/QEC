"""Deterministic trapping-set risk correlation experiment for v12.8."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor
from qec.experiments.spectral_phase_diagram import (
    _derive_seed,
    _run_decoder_trial,
)


_ROUND = 12


def _make_graphs() -> list[np.ndarray]:
    return [
        np.array([
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
        ], dtype=np.float64),
        np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1],
        ], dtype=np.float64),
    ]


def run_trapping_set_prediction_experiment(
    *,
    error_rate: float = 0.05,
    trials_per_graph: int = 8,
    base_seed: int = 42,
) -> dict[str, Any]:
    predictor = NBTrappingSetPredictor()
    graphs = _make_graphs()

    records: list[dict[str, Any]] = []
    positives = 0
    predicted_positive = 0
    true_positive = 0

    for gi, H in enumerate(graphs):
        pred = predictor.predict_trapping_regions(H)
        risk = float(pred["risk_score"])
        pred_fail = risk > 0.0
        if pred_fail:
            predicted_positive += 1

        failures = 0
        for ti in range(trials_per_graph):
            seed = _derive_seed(base_seed, f"trapping_exp_g{gi}_t{ti}_er{error_rate}")
            trial = _run_decoder_trial(H, error_rate, seed)
            if not trial["success"]:
                failures += 1

        fer = failures / trials_per_graph if trials_per_graph > 0 else 0.0
        actual_fail = fer > 0.0
        if actual_fail:
            positives += 1
        if pred_fail and actual_fail:
            true_positive += 1

        records.append({
            "graph_index": gi,
            "fer": round(float(fer), _ROUND),
            "spectral_radius": round(float(pred["spectral_radius"]), _ROUND),
            "ipr": round(float(pred["ipr"]), _ROUND),
            "trapping_risk": round(float(risk), _ROUND),
            "predicted_failure": pred_fail,
            "observed_failure": actual_fail,
        })

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / positives if positives > 0 else 0.0

    return {
        "prediction_precision": round(float(precision), _ROUND),
        "prediction_recall": round(float(recall), _ROUND),
        "error_rate": round(float(error_rate), _ROUND),
        "trials_per_graph": int(trials_per_graph),
        "records": records,
    }


if __name__ == "__main__":
    result = run_trapping_set_prediction_experiment()
    print(json.dumps(result, sort_keys=True, indent=2))
