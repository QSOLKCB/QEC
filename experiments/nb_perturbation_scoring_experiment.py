"""v13.1.0 — Deterministic perturbation-vs-exact NB scoring experiment."""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer
from src.qec.analysis.nb_perturbation_scorer import NBPerturbationScorer
from src.qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation


_ROUND = 12
_TOP_K = 8


def _graphs() -> list[np.ndarray]:
    return [
        np.array([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float64),
    ]


def run() -> dict[str, float]:
    analyzer = NBEigenmodeFlowAnalyzer()
    scorer = NBPerturbationScorer()
    mutator = NBEigenmodeMutation(enabled=True)

    top1_match = 0
    topk_overlap = 0.0
    corr_values: list[float] = []
    n_graphs = 0
    candidate_counts: list[int] = []

    for H in _graphs():
        baseline = analyzer.analyze(H)
        swaps = mutator._enumerate_swaps(H, baseline.get("hot_edges", [])[:8])
        if not swaps:
            continue

        state = scorer.baseline_state(H)
        if not state.get("valid_first_order", False):
            continue

        pred = []
        exact = []
        for idx, swap in enumerate(swaps):
            p = scorer.score_swap(H, swap, state)
            if not p["valid_first_order"]:
                continue
            ci, vi, cj, vj = swap
            Hc = H.copy()
            Hc[ci, vi] = 0.0
            Hc[cj, vj] = 0.0
            Hc[ci, vj] = 1.0
            Hc[cj, vi] = 1.0
            sig = analyzer.analyze(Hc)["signature"]
            e = NBEigenmodeMutation._candidate_key(baseline["signature"], sig)
            pred.append((idx, p["predicted_delta"]))
            exact.append((idx, e[0]))

        if len(pred) < 2:
            continue

        n_graphs += 1
        candidate_counts.append(len(pred))
        pred_sorted = sorted(pred, key=lambda x: (x[1], x[0]))
        exact_sorted = sorted(exact, key=lambda x: (x[1], x[0]))

        top1_match += int(pred_sorted[0][0] == exact_sorted[0][0])

        k = min(_TOP_K, len(pred_sorted))
        pset = {idx for idx, _ in pred_sorted[:k]}
        eset = {idx for idx, _ in exact_sorted[:k]}
        topk_overlap += len(pset & eset) / float(k)

        p_rank = {idx: rank for rank, (idx, _) in enumerate(pred_sorted)}
        e_rank = {idx: rank for rank, (idx, _) in enumerate(exact_sorted)}
        keys = sorted(p_rank)
        p_vec = np.asarray([p_rank[k_] for k_ in keys], dtype=np.float64)
        e_vec = np.asarray([e_rank[k_] for k_ in keys], dtype=np.float64)
        if len(keys) > 1:
            corr = np.corrcoef(p_vec, e_vec)[0, 1]
            corr_values.append(float(0.0 if np.isnan(corr) else corr))

    if n_graphs == 0:
        return {
            "top1_agreement": 0.0,
            "topk_overlap": 0.0,
            "spearman_proxy": 0.0,
            "avg_candidate_count": 0.0,
            "avg_exact_rechecks": 0.0,
            "graphs_evaluated": 0.0,
        }

    avg_candidates = float(np.mean(candidate_counts))
    return {
        "top1_agreement": round(top1_match / n_graphs, _ROUND),
        "topk_overlap": round(topk_overlap / n_graphs, _ROUND),
        "spearman_proxy": round(float(np.mean(corr_values)) if corr_values else 0.0, _ROUND),
        "avg_candidate_count": round(avg_candidates, _ROUND),
        "avg_exact_rechecks": round(float(min(_TOP_K, int(round(avg_candidates)))), _ROUND),
        "graphs_evaluated": float(n_graphs),
    }


def main() -> None:
    result = run()
    print("NB Perturbation Scoring Validation (deterministic)")
    print("metric                     value")
    print("-" * 40)
    rows = [
        ("top1_agreement", result["top1_agreement"]),
        ("topk_overlap", result["topk_overlap"]),
        ("spearman_proxy", result["spearman_proxy"]),
        ("avg_candidate_count", result["avg_candidate_count"]),
        ("avg_exact_rechecks", result["avg_exact_rechecks"]),
        ("graphs_evaluated", result["graphs_evaluated"]),
    ]
    for name, value in rows:
        print(f"{name:24s} {value: .12f}")
    print(
        "Interpretation: first-order perturbation ranking is used as a deterministic prefilter for exact rechecks."
    )


if __name__ == "__main__":
    main()
