"""
v11.5.0 — Residual Cluster Analyzer.

Identifies connected high-residual basins from the BP residual map.
Two variables belong to the same cluster if they share a check node
(i.e. are connected via a short two-hop Tanner path).

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from src.qec.analysis.bp_residuals import BPResidualAnalyzer


class ResidualClusterAnalyzer:
    """Identify connected high-residual basins from the BP residual map.

    Given a parity-check matrix and a residual map (or parameters to
    compute one), this analyzer identifies clusters of variable nodes
    that form connected high-residual regions in the Tanner graph.

    Two high-residual variables belong to the same cluster if they
    share at least one check node (one-hop Tanner connectivity).

    Fully deterministic: same inputs → identical output.
    Does not modify inputs.
    """

    def __init__(self, threshold_percentile: float = 75.0) -> None:
        """Initialize the analyzer.

        Parameters
        ----------
        threshold_percentile : float
            Percentile threshold for classifying a variable as
            "high-residual".  Default 75.0 (top quartile).
        """
        self.threshold_percentile = threshold_percentile
        self._bp_analyzer = BPResidualAnalyzer()

    def find_clusters(
        self,
        H: np.ndarray,
        residual_map: np.ndarray | None = None,
        *,
        seed: int = 0,
        bp_iterations: int = 10,
    ) -> dict:
        """Identify connected high-residual basins.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).
        residual_map : np.ndarray or None
            Pre-computed residual map of shape (n,).  If None, a BP
            residual map is computed using ``BPResidualAnalyzer``.
        seed : int
            Deterministic seed for BP residual computation (used only
            when ``residual_map`` is None).
        bp_iterations : int
            Number of BP iterations for residual computation (used
            only when ``residual_map`` is None).

        Returns
        -------
        dict
            ``clusters`` : list[dict] — each cluster entry contains
            ``variables``, ``size``, ``mean_residual``,
            ``max_residual``, ``boundary_score``, ``internal_density``.
            ``num_clusters`` : int — number of clusters found.
            ``largest_cluster_size`` : int — size of the largest cluster.
            ``max_cluster_residual`` : float — highest max_residual
            across all clusters.
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        empty_result = {
            "clusters": [],
            "num_clusters": 0,
            "largest_cluster_size": 0,
            "max_cluster_residual": 0.0,
        }

        if m == 0 or n == 0:
            return empty_result

        # Compute residual map if not provided
        if residual_map is None:
            bp_result = self._bp_analyzer.compute_residual_map(
                H_arr, iterations=bp_iterations, seed=seed,
            )
            residual_map = bp_result["residual_map"]

        residual_map = np.asarray(residual_map, dtype=np.float64)
        if len(residual_map) != n:
            return empty_result

        # Determine high-residual threshold
        threshold = float(np.percentile(residual_map, self.threshold_percentile))

        # Identify high-residual variable nodes
        high_vars = sorted(
            vi for vi in range(n) if residual_map[vi] >= threshold
        )

        if not high_vars:
            return empty_result

        high_var_set = set(high_vars)

        # Build variable-variable adjacency restricted to high-residual
        # nodes.  Two variables are adjacent if they share a check node.
        adj: dict[int, list[int]] = {vi: [] for vi in high_vars}

        # Build check-to-variable mapping
        check_to_var: list[list[int]] = [[] for _ in range(m)]
        rows, cols = np.nonzero(H_arr)
        for ci, vi in zip(rows, cols):
            check_to_var[int(ci)].append(int(vi))

        for ci in range(m):
            # Find high-residual variables in this check
            high_in_check = sorted(
                vi for vi in check_to_var[ci] if vi in high_var_set
            )
            # Connect all pairs
            for i in range(len(high_in_check)):
                for j in range(i + 1, len(high_in_check)):
                    vi, vj = high_in_check[i], high_in_check[j]
                    adj[vi].append(vj)
                    adj[vj].append(vi)

        # Deduplicate and sort adjacency lists
        for vi in adj:
            adj[vi] = sorted(set(adj[vi]))

        # Find connected components via BFS
        visited: set[int] = set()
        components: list[list[int]] = []

        for vi in high_vars:
            if vi in visited:
                continue
            component: list[int] = []
            queue = [vi]
            visited.add(vi)
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))

        # Score each cluster
        # Build variable-to-check mapping for boundary computation
        var_to_check: list[list[int]] = [[] for _ in range(n)]
        for ci, vi in zip(rows, cols):
            var_to_check[int(vi)].append(int(ci))

        clusters: list[dict] = []
        for comp in components:
            comp_set = set(comp)
            size = len(comp)

            # Residual statistics
            comp_residuals = np.array(
                [residual_map[vi] for vi in comp], dtype=np.float64,
            )
            mean_residual = float(np.mean(comp_residuals))
            max_residual = float(np.max(comp_residuals))

            # Internal density: fraction of possible internal edges
            # that actually exist (via shared checks).
            internal_edges = 0
            for vi in comp:
                for neighbor in adj.get(vi, []):
                    if neighbor in comp_set:
                        internal_edges += 1
            # Each edge counted twice
            internal_edges //= 2
            max_possible = size * (size - 1) // 2 if size > 1 else 1
            internal_density = float(internal_edges) / float(max_possible)

            # Boundary score: fraction of check connections that lead
            # outside the cluster.
            total_connections = 0
            boundary_connections = 0
            for vi in comp:
                for ci in var_to_check[vi]:
                    for vj in check_to_var[ci]:
                        if vj == vi:
                            continue
                        total_connections += 1
                        if vj not in comp_set:
                            boundary_connections += 1
            boundary_score = (
                float(boundary_connections) / float(total_connections)
                if total_connections > 0
                else 0.0
            )

            clusters.append({
                "variables": comp,
                "size": size,
                "mean_residual": mean_residual,
                "max_residual": max_residual,
                "boundary_score": boundary_score,
                "internal_density": internal_density,
            })

        # Sort clusters by risk: high residual + high density + low boundary
        clusters.sort(
            key=lambda c: (
                -round(c["max_residual"], 12),
                -round(c["internal_density"], 12),
                round(c["boundary_score"], 12),
                -c["size"],
            ),
        )

        largest = max((c["size"] for c in clusters), default=0)
        max_cluster_res = max(
            (c["max_residual"] for c in clusters), default=0.0,
        )

        return {
            "clusters": clusters,
            "num_clusters": len(clusters),
            "largest_cluster_size": largest,
            "max_cluster_residual": max_cluster_res,
        }
