"""v16.0.0 — Deterministic non-backtracking eigenvector flow mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nonbacktracking_flow import NBFlowConfig, NonBacktrackingEigenvectorFlowAnalyzer, canonical_directed_edges
from src.qec.analysis.spectral_frustration import spectral_frustration_count


@dataclass(frozen=True)
class NBFlowMutationConfig:
    enabled: bool = False
    max_flow_edges: int = 8
    swap_candidates_per_edge: int = 4
    flow_threshold: float = 0.0
    enable_second_mode: bool = True
    second_mode_weight: float = 0.15
    precision: int = 12


class NonBacktrackingFlowMutator:
    """Opt-in deterministic flow-aligned Tanner graph rewiring."""

    def __init__(
        self,
        *,
        config: NBFlowMutationConfig | None = None,
        eta_nb: float = 1.0,
        eta_frustration: float = 0.0,
    ) -> None:
        self.config = config or NBFlowMutationConfig()
        self.eta_nb = float(eta_nb)
        self.eta_frustration = float(eta_frustration)
        self._analyzer = NonBacktrackingEigenvectorFlowAnalyzer(
            config=NBFlowConfig(precision=self.config.precision),
        )

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        H_new = self._to_dense_copy(H)
        if not self.config.enabled:
            return H_new, []

        eigvals, eigvecs = self._analyzer.compute_modes(H_new)
        if eigvecs.shape[1] == 0:
            return H_new, []

        flow_vector, mode_index, flow_norm = self._build_flow_vector(eigvals, eigvecs)
        if flow_norm <= 1e-15:
            return H_new, []

        undirected, directed, edge_index, _, _, n = canonical_directed_edges(H_new)
        if not undirected:
            return H_new, []

        flow_map = self._edge_flow_map(
            undirected=undirected,
            edge_index=edge_index,
            n=n,
            flow_vector=flow_vector,
        )
        ranked_edges = sorted(
            undirected,
            key=lambda edge: (-abs(flow_map.get(edge, 0.0)), edge),
        )

        top_edges: list[tuple[int, int]] = []
        for edge in ranked_edges:
            if abs(flow_map.get(edge, 0.0)) <= float(self.config.flow_threshold):
                continue
            top_edges.append(edge)
            if len(top_edges) >= int(self.config.max_flow_edges):
                break

        if not top_edges:
            return H_new, []

        base_frustration = self._frustration_score(H_new)
        check_neighbors, var_neighbors = self._build_neighbors(H_new)
        candidates: list[dict[str, Any]] = []
        for edge in top_edges:
            swaps = self._candidate_swaps_for_edge(
                edge=edge,
                H=H_new,
                check_neighbors=check_neighbors,
                var_neighbors=var_neighbors,
            )
            for swap in swaps:
                cand = self._score_swap(
                    H=H_new,
                    swap=swap,
                    flow_map=flow_map,
                    base_frustration=base_frustration,
                )
                candidates.append(cand)

        if not candidates:
            return H_new, []

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item["score"],
                item["swap"],
            ),
        )

        for cand in ranked_candidates:
            if cand["delta_flow"] >= 0.0:
                continue
            swap = cand["swap"]
            ci, vi, cj, vj = swap
            H_new[ci, vi] = 0.0
            H_new[cj, vj] = 0.0
            H_new[ci, vj] = 1.0
            H_new[cj, vi] = 1.0
            meta = {
                "flow_mode_index": int(mode_index),
                "max_flow_edge": tuple(top_edges[0]),
                "swap_selected": tuple(swap),
                "flow_alignment_score": cand["alignment"],
                "flow_norm": round(float(flow_norm), self.config.precision),
                "score": cand["score"],
            }
            return H_new, [meta]

        return H_new, []

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    def _build_flow_vector(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> tuple[np.ndarray, int, float]:
        v1 = np.asarray(eigvecs[:, 0], dtype=np.complex128)
        n1 = float(np.linalg.norm(v1))
        if n1 <= 1e-15:
            return np.zeros(v1.shape[0], dtype=np.float64), 0, 0.0
        v1 = v1 / n1

        mode_index = 0
        degenerate_blend = False
        primary = v1
        if eigvecs.shape[1] > 1 and eigvals.shape[0] > 1:
            lam0 = complex(eigvals[0])
            lam1 = complex(eigvals[1])
            scale = max(abs(lam0), abs(lam1), 1e-15)
            if abs(lam0 - lam1) < 1e-6 * scale:
                v2d = np.asarray(eigvecs[:, 1], dtype=np.complex128)
                n2d = float(np.linalg.norm(v2d))
                if n2d > 1e-15:
                    v2d = v2d / n2d
                    vb = v1 + v2d
                    nb = float(np.linalg.norm(vb))
                    if nb > 1e-15:
                        primary = vb / nb
                        degenerate_blend = True
                        mode_index = 1

        if self.config.enable_second_mode and eigvecs.shape[1] > 1 and not degenerate_blend:
            v2 = np.asarray(eigvecs[:, 1], dtype=np.complex128)
            n2 = float(np.linalg.norm(v2))
            if n2 > 1e-15:
                v2 = v2 / n2
                v = primary + float(self.config.second_mode_weight) * v2
                mode_index = 1
            else:
                v = primary
        else:
            v = primary

        vnorm = float(np.linalg.norm(v))
        if vnorm <= 1e-15:
            return np.zeros(v1.shape[0], dtype=np.float64), mode_index, 0.0
        v = v / vnorm
        out = np.asarray(np.real(v), dtype=np.float64)
        idx = int(np.argmax(np.abs(out))) if out.size else 0
        if out.size and out[idx] < 0.0:
            out = -out
        return out, mode_index, vnorm

    def _edge_flow_map(
        self,
        *,
        undirected: tuple[tuple[int, int], ...],
        edge_index: dict[tuple[int, int], int],
        n: int,
        flow_vector: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        flow: dict[tuple[int, int], float] = {}
        for ci, vi in undirected:
            a = edge_index.get((vi, n + ci))
            b = edge_index.get((n + ci, vi))
            p_uv = float(flow_vector[a]) if a is not None else 0.0
            p_vu = float(flow_vector[b]) if b is not None else 0.0
            strength = round(abs(p_uv - p_vu), self.config.precision)
            flow[(ci, vi)] = float(strength)
        return flow

    @staticmethod
    def _build_neighbors(H: np.ndarray) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        m, n = H.shape
        check_neighbors = {ci: set(np.flatnonzero(H[ci] != 0).tolist()) for ci in range(m)}
        var_neighbors = {vi: set(np.flatnonzero(H[:, vi] != 0).tolist()) for vi in range(n)}
        return check_neighbors, var_neighbors

    def _candidate_swaps_for_edge(
        self,
        *,
        edge: tuple[int, int],
        H: np.ndarray,
        check_neighbors: dict[int, set[int]],
        var_neighbors: dict[int, set[int]],
    ) -> list[tuple[int, int, int, int]]:
        ci, vi = edge
        m, n = H.shape
        _ = m
        ci_neighbors = check_neighbors.get(ci, set())
        swaps: list[tuple[int, int, int, int]] = []
        if vi not in ci_neighbors or len(ci_neighbors) <= 1 or len(var_neighbors.get(vi, set())) <= 1:
            return swaps

        var_set = set(range(n))
        non_neighbor_vars = sorted(var_set - ci_neighbors)
        for vj in non_neighbor_vars:
            if vj == vi:
                continue
            for cj in sorted(var_neighbors.get(vj, set())):
                if cj == ci:
                    continue
                if H[cj, vi] != 0 or len(check_neighbors.get(cj, set())) <= 1:
                    continue
                swap = (ci, vi, cj, vj)
                swap = self._canonical_swap(swap)
                swaps.append(swap)

        swaps = sorted(set(swaps))
        return swaps[: int(self.config.swap_candidates_per_edge)]

    @staticmethod
    def _canonical_swap(swap: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        ci, vi, cj, vj = swap
        a = (ci, vi)
        b = (cj, vj)
        if a <= b:
            return (ci, vi, cj, vj)
        return (cj, vj, ci, vi)

    def _score_swap(
        self,
        *,
        H: np.ndarray,
        swap: tuple[int, int, int, int],
        flow_map: dict[tuple[int, int], float],
        base_frustration: float,
    ) -> dict[str, Any]:
        ci, vi, cj, vj = swap
        before = flow_map.get((ci, vi), 0.0) + flow_map.get((cj, vj), 0.0)
        after = flow_map.get((ci, vj), 0.0) + flow_map.get((cj, vi), 0.0)
        delta_flow = round(float(after - before), self.config.precision)
        alignment = round(float(before - after), self.config.precision)

        H_candidate = H.copy()
        H_candidate[ci, vi] = 0.0
        H_candidate[cj, vj] = 0.0
        H_candidate[ci, vj] = 1.0
        H_candidate[cj, vi] = 1.0
        cand_frustration = self._frustration_score(H_candidate)
        delta_frustration = round(float(cand_frustration - base_frustration), self.config.precision)

        score = round(
            float(delta_flow)
            + float(self.eta_nb) * float(alignment)
            + float(self.eta_frustration) * float(delta_frustration),
            self.config.precision,
        )
        return {
            "swap": swap,
            "delta_flow": float(delta_flow),
            "alignment": float(alignment),
            "delta_frustration": float(delta_frustration),
            "score": float(score),
        }

    def _frustration_score(self, H: np.ndarray) -> float:
        m, n = H.shape
        size = m + n
        A = np.zeros((size, size), dtype=np.float64)
        for ci in range(m):
            for vi in np.flatnonzero(H[ci] != 0):
                u = ci
                v = m + int(vi)
                A[u, v] = 1.0
                A[v, u] = 1.0
        out = spectral_frustration_count(A, r=1.0, candidate_swaps=None)
        return round(float(out.get("baseline_negative_modes", 0.0)), self.config.precision)
