"""
v13.1.0 — Deterministic NB Eigenmode Mutation Operator.

Performs degree-preserving Tanner-graph rewires guided by dominant
non-backtracking eigenmode pressure and compact eigenmode signatures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer
from src.qec.analysis.nb_perturbation_scorer import NBPerturbationScorer


_ROUND = 12


class NBEigenmodeMutation:
    """Opt-in deterministic NB-eigenmode mutation operator."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        hot_edges_limit: int = 8,
        precision: int = _ROUND,
        use_nb_perturbation_scoring: bool = False,
        top_k_exact_recheck: int = 8,
    ) -> None:
        if top_k_exact_recheck < 1:
            raise ValueError("top_k_exact_recheck must be >= 1")
        self.enabled = enabled
        self.hot_edges_limit = hot_edges_limit
        self.precision = precision
        self.use_nb_perturbation_scoring = use_nb_perturbation_scoring
        self.top_k_exact_recheck = top_k_exact_recheck
        self._analyzer = NBEigenmodeFlowAnalyzer()
        self._perturbation = NBPerturbationScorer()

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        H_new = self._to_dense_copy(H)
        if not self.enabled:
            return H_new, []

        baseline = self._analyzer.analyze(H_new)
        hot_edges = baseline.get("hot_edges", [])[: self.hot_edges_limit]
        if not hot_edges:
            return H_new, []

        swaps = self._enumerate_swaps(H_new, hot_edges)
        if not swaps:
            return H_new, []

        if self.use_nb_perturbation_scoring:
            candidates = self._hybrid_candidates(H_new, baseline, swaps)
        else:
            candidates = self._exact_candidates(H_new, baseline, swaps)

        if not candidates:
            return H_new, []

        candidates.sort(key=lambda item: (item[0], item[1]))
        _, (ci, vi, cj, vj), sig, meta = candidates[0]

        H_new[ci, vi] = 0.0
        H_new[cj, vj] = 0.0
        H_new[ci, vj] = 1.0
        H_new[cj, vi] = 1.0

        mutation_log = [{
            "removed_edge": (ci, vi),
            "partner_removed": (cj, vj),
            "added_edge": (ci, vj),
            "partner_added": (cj, vi),
            "signature": sig,
            **meta,
        }]
        return H_new, mutation_log

    def _hybrid_candidates(
        self,
        H: np.ndarray,
        baseline: dict[str, Any],
        swaps: list[tuple[int, int, int, int]],
    ) -> list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any], dict[str, Any]]]:
        base_state = self._perturbation.baseline_state(H)
        if not bool(base_state.get("valid_first_order", False)):
            return self._exact_candidates(H, baseline, swaps, fallback_exact=True)

        predicted: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
        for swap in swaps:
            score = self._perturbation.score_swap(H, swap, base_state)
            if score is None or not score["valid_first_order"]:
                continue
            predicted.append((
                float(score.get("weighted_delta", score["predicted_delta"])),
                swap,
                score,
            ))

        if not predicted:
            return self._exact_candidates(H, baseline, swaps, fallback_exact=True)

        predicted.sort(key=lambda x: (x[0], x[1]))
        top_k = min(self.top_k_exact_recheck, len(predicted))

        out: list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any], dict[str, Any]]] = []
        for _, swap, pscore in predicted[:top_k]:
            ci, vi, cj, vj = swap
            H_candidate = H.copy()
            H_candidate[ci, vi] = 0.0
            H_candidate[cj, vj] = 0.0
            H_candidate[ci, vj] = 1.0
            H_candidate[cj, vi] = 1.0

            cand_sig = self._analyzer.analyze(H_candidate)["signature"]
            key = self._candidate_key(baseline["signature"], cand_sig)
            if key[0] >= 0.0:
                continue
            out.append((key, swap, cand_sig, {
                "score_mode": "perturbation_hybrid",
                "predicted_delta": pscore["predicted_delta"],
                "weighted_delta": pscore.get("weighted_delta", pscore["predicted_delta"]),
                "pressure_weight": pscore.get("pressure_weight", 0.0),
                "predicted_new_value": pscore["predicted_new_value"],
                "tracked_eigenvalue": pscore["tracked_eigenvalue"],
                "perturbation_nnz": int(pscore["perturbation_nnz"]),
                "exact_rechecks": top_k,
                "candidate_count": len(swaps),
            }))
        return out

    def _exact_candidates(
        self,
        H: np.ndarray,
        baseline: dict[str, Any],
        swaps: list[tuple[int, int, int, int]],
        *,
        fallback_exact: bool = False,
    ) -> list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any], dict[str, Any]]]:
        out: list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any], dict[str, Any]]] = []
        for ci, vi, cj, vj in swaps:
            H_candidate = H.copy()
            H_candidate[ci, vi] = 0.0
            H_candidate[cj, vj] = 0.0
            H_candidate[ci, vj] = 1.0
            H_candidate[cj, vi] = 1.0
            cand_sig = self._analyzer.analyze(H_candidate)["signature"]
            key = self._candidate_key(baseline["signature"], cand_sig)
            if key[0] >= 0.0:
                continue
            out.append((key, (ci, vi, cj, vj), cand_sig, {
                "score_mode": "exact",
                "fallback_exact": bool(fallback_exact),
                "exact_rechecks": len(swaps),
                "candidate_count": len(swaps),
            }))
        return out

    @staticmethod
    def _enumerate_swaps(
        H: np.ndarray,
        hot_edges: list[tuple[int, int]],
    ) -> list[tuple[int, int, int, int]]:
        check_neighbors, var_neighbors = NBEigenmodeMutation._build_neighbors(H)
        swaps: list[tuple[int, int, int, int]] = []
        for ci, vi in hot_edges:
            if H[ci, vi] == 0:
                continue
            if len(check_neighbors[ci]) <= 1 or len(var_neighbors[vi]) <= 1:
                continue
            for vj in range(H.shape[1]):
                if vj == vi or H[ci, vj] != 0:
                    continue
                for cj in sorted(var_neighbors[vj]):
                    if cj == ci:
                        continue
                    if vi in check_neighbors[cj]:
                        continue
                    if len(check_neighbors[cj]) <= 1:
                        continue
                    swaps.append((ci, vi, cj, vj))
        return swaps

    @staticmethod
    def _candidate_key(
        before: dict[str, float],
        after: dict[str, float],
    ) -> tuple[float, ...]:
        dr = round(after["spectral_radius"] - before["spectral_radius"], _ROUND)
        di = round(after["mode_ipr"] - before["mode_ipr"], _ROUND)
        ds = round(after["support_fraction"] - before["support_fraction"], _ROUND)
        dt = round(after["topk_mass_fraction"] - before["topk_mass_fraction"], _ROUND)
        total = round(dr + di + ds + dt, _ROUND)
        return (total, dr, di, ds, dt)

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    @staticmethod
    def _build_neighbors(
        H: np.ndarray,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        m, n = H.shape
        check_neighbors: dict[int, set[int]] = {ci: set() for ci in range(m)}
        var_neighbors: dict[int, set[int]] = {vi: set() for vi in range(n)}
        coords = np.argwhere(H != 0)
        for ci, vi in coords:
            cii = int(ci)
            vii = int(vi)
            check_neighbors[cii].add(vii)
            var_neighbors[vii].add(cii)
        return check_neighbors, var_neighbors


def nb_eigenmode_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Operator wrapper matching existing mutation-operator signature."""
    del seed, target_edges
    mutator = NBEigenmodeMutation(enabled=True)
    H_mut, _ = mutator.mutate(H)
    return H_mut
