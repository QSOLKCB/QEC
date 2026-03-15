"""Deterministic non-backtracking eigenvector-flow mutation operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.spectral.nb_spectrum import compute_nb_spectral_gap

_ROUND = 12


def spectral_annealing_strength(gap):
    """Deterministic monotonic scaling for spectral annealing."""
    strength = 1.0 / (1.0 + gap)
    return round(float(strength), 12)


def _mutation_size(
    *,
    annealing: bool,
    base_mutation_size: int,
    nb_eigenvalues: np.ndarray | None,
) -> tuple[int, float | None, float | None]:
    if not annealing:
        return 1, None, None
    gap = compute_nb_spectral_gap(np.array([], dtype=np.float64) if nb_eigenvalues is None else nb_eigenvalues)
    strength = spectral_annealing_strength(gap)
    size = max(1, int(int(base_mutation_size) * strength))
    return size, gap, strength
def compute_ipr_localization(eigenvector: np.ndarray) -> float:
    """Compute deterministic inverse participation ratio (IPR) for an eigenvector."""
    v = np.asarray(eigenvector, dtype=np.float64)
    power2 = v ** 2
    power4 = v ** 4
    denom = float(np.sum(power2, dtype=np.float64) ** 2)
    numer = float(np.sum(power4, dtype=np.float64))
    if denom == 0.0:
        return 0.0
    ipr = numer / denom
    return round(float(ipr), _ROUND)


def select_localized_edges(
    edge_flow: np.ndarray,
    eigenvector: np.ndarray,
    top_fraction: float = 0.1,
) -> np.ndarray:
    """Select deterministic top-magnitude localized edge indices."""
    flow = np.asarray(edge_flow, dtype=np.float64)
    vec = np.asarray(eigenvector, dtype=np.float64)
    n = int(min(flow.size, vec.size))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    magnitudes = np.abs(vec[:n]).astype(np.float64)
    order = np.lexsort((np.arange(n, dtype=np.int64), -magnitudes))
    frac = float(np.clip(np.float64(top_fraction), 0.0, 1.0))
    k = max(1, int(n * frac))
    return np.asarray(order[:k], dtype=np.int64)


class NBEigenvectorFlowMutator:
    """Deterministic mutation operator guided by NB eigenvector flow."""

    def __init__(self) -> None:
        pass

    def compute_flow(self, eigenvector: np.ndarray) -> np.ndarray:
        """Compute normalized edge-flow magnitude from NB eigenvector."""
        vec = np.asarray(eigenvector, dtype=np.float64)
        flow = np.abs(vec).astype(np.float64)
        total = float(np.sum(flow, dtype=np.float64))
        if total <= 0.0:
            if flow.size == 0:
                return flow
            flow = np.full(flow.shape, 1.0 / float(flow.size), dtype=np.float64)
        else:
            flow = flow / total
        return flow.astype(np.float64)

    def select_edge(self, flow: np.ndarray, candidate_indices: np.ndarray | None = None) -> int:
        """Deterministically choose the directed-edge index with highest flow."""
        f = np.asarray(flow, dtype=np.float64)
        if f.size == 0:
            return -1
        if candidate_indices is None:
            return int(np.argmax(f))

        cands = np.asarray(candidate_indices, dtype=np.int64)
        if cands.size == 0:
            return -1
        in_range = cands[(cands >= 0) & (cands < int(f.size))]
        if in_range.size == 0:
            return -1
        vals = f[in_range]
        order = np.lexsort((in_range, -vals))
        return int(in_range[int(order[0])])

    def mutate(
        self,
        graph: np.ndarray,
        nb_eigenvector: np.ndarray,
        *,
        nb_eigenvalues: np.ndarray | None = None,
        annealing: bool = False,
        base_mutation_size: int = 4,
        use_ipr_localization: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mutate a binary parity-check matrix using the dominant NB eigenvector."""
        flow = self.compute_flow(nb_eigenvector)
        mutation_size, gap, strength = _mutation_size(
            annealing=bool(annealing),
            base_mutation_size=int(base_mutation_size),
            nb_eigenvalues=None if nb_eigenvalues is None else np.asarray(nb_eigenvalues, dtype=np.complex128),
        )
        _ = bool(use_ipr_localization)
        mutation_size = min(max(1, mutation_size), int(flow.size) if flow.size > 0 else 1)

        order = np.argsort(-flow, kind="stable") if flow.size > 0 else np.array([], dtype=np.int64)
        chosen = order[:mutation_size]
        if chosen.size == 0:
            chosen = np.array([self.select_edge(flow)], dtype=np.int64)
        mutated = np.asarray(graph, dtype=np.float64).copy()
        for edge_index in chosen:
            mutated = self._mutate_edge(mutated, int(edge_index))

        edge_index = int(chosen[0]) if chosen.size > 0 else -1
        flow_strength = 0.0
        if flow.size > 0 and edge_index >= 0:
            flow_strength = round(float(flow[edge_index]), _ROUND)
        meta = {
        use_ipr_localization: bool = True,
        localization_fraction: float = 0.1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mutate a binary parity-check matrix using the dominant NB eigenvector."""
        flow = self.compute_flow(nb_eigenvector)
        cluster = np.zeros(0, dtype=np.int64)
        if use_ipr_localization:
            cluster = select_localized_edges(flow, nb_eigenvector, top_fraction=localization_fraction)
            edge_index = self.select_edge(flow, cluster)
        else:
            edge_index = self.select_edge(flow)

        mutated = self._mutate_edge(graph, edge_index)
        flow_strength = 0.0
        if flow.size > 0 and edge_index >= 0:
            flow_strength = round(float(flow[edge_index]), _ROUND)
        ipr_score = compute_ipr_localization(nb_eigenvector)
        return mutated, {
            "flow_edge_index": int(edge_index),
            "flow_strength": flow_strength,
            "ipr_localization_score": ipr_score,
            "localization_edge_count": int(cluster.size if use_ipr_localization else flow.size),
        }
        if annealing:
            meta.update({
                "nb_spectral_gap": 0.0 if gap is None else round(float(gap), _ROUND),
                "annealing_strength": 1.0 if strength is None else round(float(strength), _ROUND),
                "mutation_size": int(mutation_size),
            })
        return mutated, meta

    def _mutate_edge(self, graph: np.ndarray, edge_index: int) -> np.ndarray:
        """Deterministic edge rewiring for a binary parity-check matrix."""
        H = np.asarray(graph, dtype=np.float64)
        g = H.copy()
        m, n = g.shape
        if m == 0 or n == 0:
            return g

        edges = np.argwhere(g == 1.0)
        if edges.size == 0 or edge_index < 0:
            return g

        idx = int(edge_index) % int(edges.shape[0])
        u = int(edges[idx, 0])
        v = int(edges[idx, 1])

        new_v = (v + 1) % n
        for offset in range(n):
            cand_v = (new_v + offset) % n
            if cand_v == v:
                continue
            if g[u, cand_v] == 1.0:
                continue

            swap_row = -1
            for r in range(m):
                if r == u:
                    continue
                if g[r, cand_v] == 1.0 and g[r, v] == 0.0:
                    swap_row = r
                    break

            if swap_row >= 0:
                g[u, v] = 0.0
                g[u, cand_v] = 1.0
                g[swap_row, cand_v] = 0.0
                g[swap_row, v] = 1.0
                return g

        return g


def nb_flow_mutation(
    H_current: np.ndarray,
    eigenvector: np.ndarray,
    eigenvalues: np.ndarray | None = None,
    *,
    use_ipr_localization: bool = False,
    annealing: bool = False,
    base_mutation_size: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Functional wrapper for deterministic NB flow mutation."""
    mutator = NBEigenvectorFlowMutator()
    return mutator.mutate(
        H_current,
        eigenvector,
        nb_eigenvalues=eigenvalues,
        use_ipr_localization=use_ipr_localization,
        annealing=annealing,
        base_mutation_size=base_mutation_size,
    )
