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
    ) -> None:
        self.enabled = enabled
        self.hot_edges_limit = hot_edges_limit
        self.precision = precision
        self.early_exit_improvement_threshold = early_exit_improvement_threshold
        self._analyzer = NBEigenmodeFlowAnalyzer(precision=precision)

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
        candidates: list[tuple[tuple[float, ...], tuple[int, int, int, int], dict[str, Any]]] = []

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

                    H_candidate = H_new.copy()
                    H_candidate[ci, vi] = 0.0
                    H_candidate[cj, vj] = 0.0
                    H_candidate[ci, vj] = 1.0
                    H_candidate[cj, vi] = 1.0

                    cand_sig = self._analyzer.analyze(H_candidate)["signature"]
                    key = self._candidate_key(baseline["signature"], cand_sig)
                    if key[0] >= 0.0:
                        continue

                    candidates.append((key, (ci, vi, cj, vj), cand_sig))
                    if (
                        self.early_exit_improvement_threshold is not None
                        and key[0] < -float(self.early_exit_improvement_threshold)
                    ):
                        candidates.sort(key=lambda item: (item[0], item[1]))
                        return self._apply_candidate(H_new, candidates[0])

        if not candidates:
            return H_new, []

        candidates.sort(key=lambda item: (item[0], item[1]))
        return self._apply_candidate(H_new, candidates[0])

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
