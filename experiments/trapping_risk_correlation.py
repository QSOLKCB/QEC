"""v12.9.0 — Deterministic trapping-risk vs FER correlation experiment."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor
from qec.experiments.spectral_phase_diagram import _derive_seed, _run_decoder_trial


_ROUND = 12


def _generate_graphs(num_graphs: int, base_seed: int) -> list[np.ndarray]:
    graphs: list[np.ndarray] = []
    for gi in range(num_graphs):
        seed = _derive_seed(base_seed, f"trapping_risk_graph_{gi}")
        rng = np.random.RandomState(seed)

        m = 3 + (gi % 4)
        n = m + 2 + (gi % 3)
        density = 0.35 + 0.05 * (gi % 6)
        H = (rng.random((m, n)) < density).astype(np.float64)

        for row in range(m):
            if H[row].sum() == 0:
                H[row, rng.randint(n)] = 1.0
        for col in range(n):
            if H[:, col].sum() == 0:
                H[rng.randint(m), col] = 1.0

        graphs.append(H)
    return graphs


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xc = x - x_mean
    yc = y - y_mean
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(xc * yc) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _ascii_histogram(values: np.ndarray, bins: int = 10, width: int = 40) -> list[str]:
    if values.size == 0:
        return ["(empty)"]
    counts, edges = np.histogram(values, bins=bins)
    max_count = int(max(counts)) if counts.size > 0 else 1
    if max_count <= 0:
        max_count = 1

    lines: list[str] = []
    for i in range(bins):
        lo = round(float(edges[i]), _ROUND)
        hi = round(float(edges[i + 1]), _ROUND)
        count = int(counts[i])
        bar_len = int(round((count / max_count) * width)) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"[{lo:.3f}, {hi:.3f}) | {bar} ({count})")
    return lines


def run_trapping_risk_correlation_experiment(
    *,
    num_graphs: int = 200,
    error_rate: float = 0.05,
    trials_per_graph: int = 4,
    base_seed: int = 42,
) -> dict[str, Any]:
    predictor = NBTrappingSetPredictor()
    graphs = _generate_graphs(num_graphs, base_seed)

    risks: list[float] = []
    fers: list[float] = []
    records: list[dict[str, Any]] = []

    for gi, H in enumerate(graphs):
        prediction = predictor.predict_trapping_regions(H)
        trapping_risk = round(float(prediction["risk_score"]), _ROUND)

        failures = 0
        for ti in range(trials_per_graph):
            seed = _derive_seed(base_seed, f"trapping_risk_fer_g{gi}_t{ti}_er{error_rate}")
            trial = _run_decoder_trial(H, error_rate, seed)
            if not trial["success"]:
                failures += 1

        fer = round(float(failures / trials_per_graph if trials_per_graph > 0 else 0.0), _ROUND)
        risks.append(trapping_risk)
        fers.append(fer)
        records.append({
            "graph_index": int(gi),
            "trapping_risk": trapping_risk,
            "fer": fer,
        })

    risk_arr = np.asarray(risks, dtype=np.float64)
    fer_arr = np.asarray(fers, dtype=np.float64)

    pearson_corr = round(_pearson(risk_arr, fer_arr), _ROUND)
    spearman_corr = round(_spearman(risk_arr, fer_arr), _ROUND)

    return {
        "num_graphs": int(num_graphs),
        "error_rate": round(float(error_rate), _ROUND),
        "trials_per_graph": int(trials_per_graph),
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "fer_histogram_ascii": _ascii_histogram(fer_arr),
        "records": records,
    }


if __name__ == "__main__":
    result = run_trapping_risk_correlation_experiment()
    print(json.dumps(result, sort_keys=True, indent=2))
