"""
v13.0.0 — Deterministic NB Eigenmode Mutation Operator.

Performs degree-preserving Tanner-graph rewires guided by dominant
non-backtracking eigenmode pressure and compact eigenmode signatures.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
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
        early_exit_improvement_threshold: float | None = None,
        use_nb_perturbation_scoring: bool = False,
        use_hybrid_perturbation_scoring: bool | None = None,
        top_k_exact_recheck: int = 3,
        use_pressure_weighting: bool = False,
        use_support_aware_heuristic: bool = False,
    ) -> None:
        self.enabled = enabled
        self.hot_edges_limit = hot_edges_limit
        self.precision = precision
        self.early_exit_improvement_threshold = early_exit_improvement_threshold
        if use_hybrid_perturbation_scoring is None:
            self.use_nb_perturbation_scoring = use_nb_perturbation_scoring
        else:
            self.use_nb_perturbation_scoring = bool(use_hybrid_perturbation_scoring)
        self.use_hybrid_perturbation_scoring = self.use_nb_perturbation_scoring
        self.top_k_exact_recheck = int(top_k_exact_recheck)
        self.use_pressure_weighting = use_pressure_weighting
        self.use_support_aware_heuristic = use_support_aware_heuristic

        if self.early_exit_improvement_threshold is not None:
            threshold = float(self.early_exit_improvement_threshold)
            if threshold < 0.0:
                raise ValueError("early_exit_improvement_threshold must be non-negative")
        if self.use_nb_perturbation_scoring and self.top_k_exact_recheck < 1:
            raise ValueError("top_k_exact_recheck must be >= 1")

        self._analyzer = NBEigenmodeFlowAnalyzer(precision=precision)
        self._perturbation_scorer = NBPerturbationScorer(precision=precision)

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

        check_neighbors, var_neighbors = self._build_neighbors(H_new)
        swaps = self._enumerate_swaps(H_new, hot_edges, check_neighbors, var_neighbors)
        if not swaps:
            return H_new, []

        if self.use_nb_perturbation_scoring:
            return self._mutate_hybrid(H_new, baseline["signature"], swaps)
        return self._mutate_exact(H_new, baseline["signature"], swaps)

    def _mutate_exact(
        self,
        H_new: np.ndarray,
        baseline_signature: dict[str, float],
        swaps: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        candidates: list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any]]] = []
        best_candidate: tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any]] | None = None
        threshold = (
            float(self.early_exit_improvement_threshold)
            if self.early_exit_improvement_threshold is not None
            else None
        )

        for swap in swaps:
            ci, vi, cj, vj = swap
            H_candidate = H_new.copy()
            H_candidate[ci, vi] = 0.0
            H_candidate[cj, vj] = 0.0
            H_candidate[ci, vj] = 1.0
            H_candidate[cj, vi] = 1.0

            cand_sig = self._analyzer.analyze(H_candidate)["signature"]
            key = self._candidate_key(baseline_signature, cand_sig)
            if key[0] >= 0.0:
                continue

            candidate = (key, swap, cand_sig)
            candidates.append(candidate)
            if best_candidate is None or (candidate[0], candidate[1]) < (best_candidate[0], best_candidate[1]):
                best_candidate = candidate

            # Sign convention: lower key[0] is better, so improvement magnitude is -key[0].
            # Early exit triggers when that magnitude strictly exceeds the threshold.
            if threshold is not None and key[0] < -threshold:
                return self._apply_candidate(H_new, best_candidate)

        if not candidates:
            return H_new, []

        candidates.sort(key=lambda item: (item[0], item[1]))
        return self._apply_candidate(H_new, candidates[0])

    def _mutate_hybrid(
        self,
        H_new: np.ndarray,
        baseline_signature: dict[str, float],
        swaps: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        # Hybrid pipeline:
        # NB spectrum -> FOHPE ranking -> exact top-k recheck -> deterministic mutate.
        spectrum = self._perturbation_scorer.compute_nb_spectrum(H_new)
        if not bool(spectrum.get("valid_first_order", False)):
            return self._mutate_exact(H_new, baseline_signature, swaps)

        u = np.asarray(spectrum.get("u", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        support_indices: set[int] = set()
        if self.use_support_aware_heuristic and u.size > 0:
            abs_u = np.abs(u)
            median_abs_u = float(np.median(abs_u))
            support_indices = {int(i) for i, val in enumerate(abs_u) if float(val) > median_abs_u}

        ranked: list[tuple[tuple[float, int, int, int, int], tuple[int, int, int, int], float]] = []
        for swap in swaps:
            pred = self._perturbation_scorer.predict_swap_delta(H_new, swap, spectrum)
            if pred is None or not bool(pred.get("valid_first_order", False)):
                continue

            predicted_delta = float(pred.get("predicted_delta", 0.0))
            pressure = float(pred.get("pressure", 0.0))

            weighted_delta = predicted_delta
            if self.use_pressure_weighting:
                weighted_delta = predicted_delta * (1.0 + pressure)

            support_bonus = 0.0
            if self.use_support_aware_heuristic and self._swap_touches_support(swap, spectrum, support_indices, H_new.shape[1]):
                support_bonus = -0.1 * abs(predicted_delta)

            final_score = round(weighted_delta + support_bonus, self.precision)
            ci, vi, cj, vj = swap
            ranked.append(((final_score, ci, vi, cj, vj), swap, predicted_delta))

        if not ranked:
            return self._mutate_exact(H_new, baseline_signature, swaps)

        ranked.sort(key=lambda item: item[0])
        shortlist_size = min(max(self.top_k_exact_recheck, 1), len(ranked))
        recheck = ranked[:shortlist_size]

        best_candidate: tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any]] | None = None
        threshold = (
            float(self.early_exit_improvement_threshold)
            if self.early_exit_improvement_threshold is not None
            else None
        )

        for _, swap, predicted_delta in recheck:
            ci, vi, cj, vj = swap
            H_candidate = H_new.copy()
            H_candidate[ci, vi] = 0.0
            H_candidate[cj, vj] = 0.0
            H_candidate[ci, vj] = 1.0
            H_candidate[cj, vi] = 1.0

            cand_sig = self._analyzer.analyze(H_candidate)["signature"]
            key = self._candidate_key(baseline_signature, cand_sig)
            if key[0] >= 0.0:
                continue

            cand_sig = {
                **cand_sig,
                "predicted_delta": round(predicted_delta, self.precision),
            }
            candidate = (key, swap, cand_sig)
            if best_candidate is None or (candidate[0], candidate[1]) < (best_candidate[0], best_candidate[1]):
                best_candidate = candidate

            if threshold is not None and key[0] < -threshold:
                return self._apply_candidate(H_new, best_candidate)

        if best_candidate is None:
            return H_new, []
        return self._apply_candidate(H_new, best_candidate)

    @staticmethod
    def _swap_touches_support(
        swap: tuple[int, int, int, int],
        spectrum: dict[str, Any],
        support_indices: set[int],
        n: int,
    ) -> bool:
        if not support_indices:
            return False
        ci, vi, cj, vj = swap
        index = spectrum.get("index", {})
        touching = [
            index.get((vi, n + ci)),
            index.get((n + ci, vi)),
            index.get((vj, n + cj)),
            index.get((n + cj, vj)),
        ]
        return any(idx is not None and int(idx) in support_indices for idx in touching)

    @staticmethod
    def _enumerate_swaps(
        H_new: np.ndarray,
        hot_edges: list[tuple[int, int]],
        check_neighbors: dict[int, set[int]],
        var_neighbors: dict[int, set[int]],
    ) -> list[tuple[int, int, int, int]]:
        swaps: list[tuple[int, int, int, int]] = []
        for ci, vi in hot_edges:
            if H_new[ci, vi] == 0:
                continue
            if len(check_neighbors[ci]) <= 1 or len(var_neighbors[vi]) <= 1:
                continue

            for vj in range(H_new.shape[1]):
                if vj == vi or H_new[ci, vj] != 0:
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

    def _apply_candidate(
        self,
        H: np.ndarray,
        candidate: tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any]],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        _, (ci, vi, cj, vj), sig = candidate

        H[ci, vi] = 0.0
        H[cj, vj] = 0.0
        H[ci, vj] = 1.0
        H[cj, vi] = 1.0

        mutation_log = [{
            "removed_edge": (ci, vi),
            "partner_removed": (cj, vj),
            "added_edge": (ci, vj),
            "partner_added": (cj, vi),
            "signature": sig,
        }]
        return H, mutation_log

    def _candidate_key(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> tuple[float, ...]:
        dr = round(after["spectral_radius"] - before["spectral_radius"], self.precision)
        di = round(after["mode_ipr"] - before["mode_ipr"], self.precision)
        ds = round(after["support_fraction"] - before["support_fraction"], self.precision)
        dt = round(after["topk_mass_fraction"] - before["topk_mass_fraction"], self.precision)
        total = round(dr + di + ds + dt, self.precision)
        return (total, dr, di, ds, dt)

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if isinstance(H, np.ndarray) and H.dtype == np.float64:
            return H.copy()
        if scipy.sparse.issparse(H):
            return np.asarray(H.toarray(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64)

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
