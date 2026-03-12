"""
v12.3.0 — NB Eigenvector Guided Mutation Operator (Experimental).

Rewires edges with high non-backtracking eigenvector magnitude to
disrupt potential trapping-set substructures.  The hypothesis that
targeting high-eigenvector edges improves decoder performance is
plausible but unproven — this feature is for research experimentation.

Layer 5 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 6


class NBGuidedMutator:
    """Deterministic mutation operator guided by the NB eigenvector.

    Experimental — the hypothesis that rewiring high-eigenvector edges
    disrupts trapping sets is plausible but unproven.  Designed for
    research experimentation only.

    Parameters
    ----------
    k : int
        Maximum number of edge rewirings per mutation (default 5).
    precision : int
        Decimal places for score rounding before ranking (default 6).
    avoid_4cycles : bool
        If True, reject rewirings that create new 4-cycles (default True).
    """

    def __init__(
        self,
        *,
        k: int = 5,
        precision: int = _ROUND,
        avoid_4cycles: bool = True,
        enabled: bool = False,
    ) -> None:
        self.k = k
        self.precision = precision
        self.avoid_4cycles = avoid_4cycles
        self.enabled = enabled

    @classmethod
    def from_config(cls, config: dict) -> "NBGuidedMutator":
        """Create mutator from a configuration dictionary.

        Expected keys (all optional):
            enabled : bool (default False)
            k : int (default 5)
            precision : int (default 6)
            avoid_4cycles : bool (default True)
        """
        nb = config.get("nb_mutation", config)
        return cls(
            k=nb.get("k", 5),
            precision=nb.get("precision", _ROUND),
            avoid_4cycles=nb.get("avoid_4cycles", True),
            enabled=nb.get("enabled", False),
        )

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        k: int | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Apply NB eigenvector guided mutation.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix, shape (m, n).
        k : int or None
            Override for maximum number of mutations.  Uses constructor
            value when None.

        Returns
        -------
        H_mutated : np.ndarray
            Mutated parity-check matrix.
        mutation_log : list[dict]
            Log of applied mutations with keys ``removed_edge``,
            ``added_edge``, and ``score``.
        """
        if k is None:
            k = self.k

        if scipy.sparse.issparse(H):
            H_arr = np.asarray(H.todense(), dtype=np.float64)
        else:
            H_arr = np.asarray(H, dtype=np.float64).copy()

        if not self.enabled:
            return H_arr, []

        m, n = H_arr.shape
        if m == 0 or n == 0:
            return H_arr, []

        # Step 1: Compute NB eigenvector flow.
        analyzer = NonBacktrackingFlowAnalyzer()
        flow = analyzer.compute_flow(H_arr)
        directed_edge_flow = flow["directed_edge_flow"]

        if len(directed_edge_flow) == 0:
            return H_arr, []

        # Step 2: Collect undirected edges and compute scores.
        edges = self._collect_edges(H_arr)
        if not edges:
            return H_arr, []

        scores = self._compute_edge_scores(
            H_arr, edges, directed_edge_flow,
            flow["directed_edges"],
        )

        # Step 3: Rank edges deterministically.
        ranked = self._rank_edges(edges, scores)

        # Step 4: Greedy mutation selection with degree-preserving swaps.
        # Track used check and variable nodes separately since they occupy
        # different partitions of the bipartite graph.
        H_new = H_arr.copy()
        mutations: list[dict[str, Any]] = []
        used_checks: set[int] = set()
        used_vars: set[int] = set()

        # Precompute neighbor sets for efficient swap search.
        check_neighbors, var_neighbors = self._build_neighbor_sets(H_new)

        for ci, vi in ranked:
            if len(mutations) >= k:
                break

            # Skip if nodes already involved in a mutation.
            if ci in used_checks or vi in used_vars:
                continue

            # Must keep at least one edge per row and column.
            if len(check_neighbors[ci]) <= 1 or len(var_neighbors[vi]) <= 1:
                continue

            swap = self._find_valid_rewire(
                H_new, ci, vi, check_neighbors, var_neighbors,
            )
            if swap is None:
                continue

            cj, vj = swap
            if cj in used_checks or vj in used_vars:
                continue

            score = scores.get((ci, vi), 0.0)

            # Apply degree-preserving two-edge swap.
            H_new[ci, vi] = 0.0
            H_new[cj, vj] = 0.0
            H_new[ci, vj] = 1.0
            H_new[cj, vi] = 1.0

            # Update neighbor sets to reflect the swap.
            check_neighbors[ci].discard(vi)
            check_neighbors[ci].add(vj)
            check_neighbors[cj].discard(vj)
            check_neighbors[cj].add(vi)
            var_neighbors[vi].discard(ci)
            var_neighbors[vi].add(cj)
            var_neighbors[vj].discard(cj)
            var_neighbors[vj].add(ci)

            mutations.append({
                "removed_edge": (ci, vi),
                "added_edge": (ci, vj),
                "score": score,
            })
            used_checks.update({ci, cj})
            used_vars.update({vi, vj})

        return H_new, mutations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
        """Collect (ci, vi) edges sorted deterministically."""
        coords = np.argwhere(H != 0)
        # np.argwhere returns rows in lexicographic (ci, vi) order for
        # a dense array, matching the required deterministic sort.
        return [(int(r[0]), int(r[1])) for r in coords]

    def _compute_edge_scores(
        self,
        H: np.ndarray,
        edges: list[tuple[int, int]],
        directed_edge_flow: np.ndarray,
        directed_edges: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """Score each undirected edge by summed directed-edge magnitudes.

        ``directed_edges`` must be in the exact ordering used to produce
        ``directed_edge_flow`` (from ``NonBacktrackingFlowAnalyzer``),
        so we do not attempt to reconstruct that ordering here.
        """
        m, n = H.shape

        if len(directed_edge_flow) != len(directed_edges):
            raise ValueError(
                f"Mismatch between directed_edge_flow "
                f"(len={len(directed_edge_flow)}) and directed_edges "
                f"(len={len(directed_edges)})"
            )

        de_index: dict[tuple[int, int], int] = {
            e: i for i, e in enumerate(directed_edges)
        }

        scores: dict[tuple[int, int], float] = {}
        for ci, vi in edges:
            j1 = de_index.get((vi, n + ci))
            j2 = de_index.get((n + ci, vi))
            s = 0.0
            if j1 is not None:
                s += abs(float(directed_edge_flow[j1]))
            if j2 is not None:
                s += abs(float(directed_edge_flow[j2]))
            scores[(ci, vi)] = round(s, self.precision)

        return scores

    def _rank_edges(
        self,
        edges: list[tuple[int, int]],
        scores: dict[tuple[int, int], float],
    ) -> list[tuple[int, int]]:
        """Sort edges by (-score, (ci, vi)) for deterministic ordering."""
        return sorted(
            edges,
            key=lambda e: (-scores.get(e, 0.0), e[0], e[1]),
        )

    @staticmethod
    def _build_neighbor_sets(
        H: np.ndarray,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Build adjacency sets for checks and variables.

        Returns
        -------
        check_neighbors : dict mapping check index -> set of variable indices
        var_neighbors : dict mapping variable index -> set of check indices
        """
        m, n = H.shape
        check_neighbors: dict[int, set[int]] = {ci: set() for ci in range(m)}
        var_neighbors: dict[int, set[int]] = {vi: set() for vi in range(n)}
        coords = np.argwhere(H != 0)
        for r in coords:
            ci, vi = int(r[0]), int(r[1])
            check_neighbors[ci].add(vi)
            var_neighbors[vi].add(ci)
        return check_neighbors, var_neighbors

    def _find_valid_rewire(
        self,
        H: np.ndarray,
        ci: int,
        vi: int,
        check_neighbors: dict[int, set[int]],
        var_neighbors: dict[int, set[int]],
    ) -> tuple[int, int] | None:
        """Find a degree-preserving swap partner for edge (ci, vi).

        Performs a two-edge swap to preserve both row and column degrees:
            remove (ci, vi) and (cj, vj)
            add    (ci, vj) and (cj, vi)
        """
        # Iterate over existing edges (cj, vj) as swap candidates.
        # Use neighbor sets instead of scanning the full matrix.
        m = len(check_neighbors)
        for cj in range(m):
            if cj == ci:
                continue
            for vj in sorted(check_neighbors[cj]):
                if vj == vi:
                    continue
                # Check that new edges don't already exist.
                if vj in check_neighbors[ci] or vi in check_neighbors[cj]:
                    continue

                if self.avoid_4cycles and self._creates_4cycle(
                    ci, vj, cj, vi, var_neighbors,
                ):
                    continue

                return (cj, vj)

        return None

    @staticmethod
    def _creates_4cycle(
        ci1: int,
        vi1: int,
        ci2: int,
        vi2: int,
        var_neighbors: dict[int, set[int]],
    ) -> bool:
        """Check whether adding edges (ci1,vi1) and (ci2,vi2) creates a 4-cycle.

        A 4-cycle exists if there is any check c (other than ci1, ci2)
        adjacent to both vi1 and vi2.
        """
        # Intersect the check neighbors of vi1 and vi2, excluding ci1 and ci2.
        shared = var_neighbors.get(vi1, set()) & var_neighbors.get(vi2, set())
        shared.discard(ci1)
        shared.discard(ci2)
        return len(shared) > 0
