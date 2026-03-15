"""Deterministic non-backtracking eigenvector-flow mutation operator."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12


class NBEigenvectorFlowMutator:
    """Deterministic mutation operator guided by NB eigenvector flow."""

    def __init__(
        self,
        *,
        enable_spectral_defect_atlas: bool = False,
        defect_atlas: Any | None = None,
    ) -> None:
        self.enable_spectral_defect_atlas = bool(enable_spectral_defect_atlas)
        self.defect_atlas = defect_atlas

    def compute_flow(self, eigenvector: np.ndarray) -> np.ndarray:
        """Compute normalized edge-flow magnitude from NB eigenvector."""
        vec = np.asarray(eigenvector)
        flow = np.abs(vec).astype(np.float64)
        total = float(np.sum(flow, dtype=np.float64))
        if total <= 0.0:
            if flow.size == 0:
                return flow
            flow = np.full(flow.shape, 1.0 / float(flow.size), dtype=np.float64)
        else:
            flow = flow / total
        return flow.astype(np.float64)

    def select_edge(self, flow: np.ndarray) -> int:
        """Deterministically choose the directed-edge index with highest flow."""
        if flow.size == 0:
            return -1
        return int(np.argmax(flow))

    def mutate(
        self,
        graph: np.ndarray,
        nb_eigenvector: np.ndarray,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mutate a binary parity-check matrix using the dominant NB eigenvector."""
        ctx = context or {}
        atlas = ctx.get("spectral_defect_atlas", self.defect_atlas)
        enable_atlas = bool(ctx.get("enable_spectral_defect_atlas", self.enable_spectral_defect_atlas))

        flow = self.compute_flow(nb_eigenvector)
        edge_index = self.select_edge(flow)
        flow_strength = 0.0
        if flow.size > 0 and edge_index >= 0:
            flow_strength = round(float(flow[edge_index]), _ROUND)

        signature = None
        if enable_atlas and atlas is not None and hasattr(atlas, "signature"):
            signature = str(atlas.signature(np.asarray(nb_eigenvector, dtype=np.float64)))

        atlas_hit = False
        atlas_pattern_index: int | None = None
        repair_action = f"flow_edge_{int(edge_index)}"

        if enable_atlas and atlas is not None and signature is not None and hasattr(atlas, "lookup"):
            pattern = atlas.lookup(signature)
            if pattern is not None:
                repair_action = str(pattern.get("repair", repair_action))
                repaired, repaired_index = self._apply_repair_action(graph, repair_action)
                if repaired_index is not None:
                    atlas_hit = True
                    if "pattern_index" in pattern:
                        atlas_pattern_index = int(pattern["pattern_index"])
                    else:
                        atlas_pattern_index = int(repaired_index)
                    return repaired, {
                        "flow_edge_index": int(repaired_index),
                        "flow_strength": flow_strength,
                        "defect_signature": signature,
                        "atlas_hit": bool(atlas_hit),
                        "atlas_pattern_index": int(atlas_pattern_index),
                        "repair_action": repair_action,
                    }

        mutated = self._mutate_edge(graph, edge_index)
        return mutated, {
            "flow_edge_index": int(edge_index),
            "flow_strength": flow_strength,
            "defect_signature": signature,
            "atlas_hit": bool(atlas_hit),
            "atlas_pattern_index": atlas_pattern_index,
            "repair_action": repair_action,
        }

    def _apply_repair_action(self, graph: np.ndarray, repair_action: str) -> tuple[np.ndarray, int | None]:
        if not isinstance(repair_action, str):
            return np.asarray(graph, dtype=np.float64).copy(), None
        if not repair_action.startswith("flow_edge_"):
            return np.asarray(graph, dtype=np.float64).copy(), None
        try:
            edge_index = int(repair_action.split("flow_edge_", 1)[1])
        except ValueError:
            return np.asarray(graph, dtype=np.float64).copy(), None
        return self._mutate_edge(graph, edge_index), int(edge_index)

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
