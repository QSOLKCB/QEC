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
        )

        # Step 3: Rank edges deterministically.
        ranked = self._rank_edges(edges, scores)

        # Step 4: Greedy mutation selection with degree-preserving swaps.
        H_new = H_arr.copy()
        mutations: list[dict[str, Any]] = []
        used_nodes: set[int] = set()

        for ci, vi in ranked:
            if len(mutations) >= k:
                break

            # Skip if nodes already involved in a mutation.
            if ci in used_nodes or vi in used_nodes:
                continue

            # Must keep at least one edge per row and column.
            if H_new[ci].sum() <= 1 or H_new[:, vi].sum() <= 1:
                continue

            swap = self._find_valid_rewire(H_new, ci, vi, m, n)
            if swap is None:
                continue

            cj, vj = swap
            if cj in used_nodes or vj in used_nodes:
                continue

            score = scores.get((ci, vi), 0.0)

            # Apply degree-preserving two-edge swap.
            H_new[ci, vi] = 0.0
            H_new[cj, vj] = 0.0
            H_new[ci, vj] = 1.0
            H_new[cj, vi] = 1.0

            mutations.append({
                "removed_edge": (ci, vi),
                "added_edge": (ci, vj),
                "score": score,
            })
            used_nodes.update({ci, vi, cj, vj})

        return H_new, mutations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
        """Collect (ci, vi) edges sorted deterministically."""
        m, n = H.shape
        edges: list[tuple[int, int]] = []
        for ci in range(m):
            for vi in range(n):
                if H[ci, vi] != 0:
                    edges.append((ci, vi))
        return edges

    def _compute_edge_scores(
        self,
        H: np.ndarray,
        edges: list[tuple[int, int]],
        directed_edge_flow: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        """Score each undirected edge by summed directed-edge magnitudes."""
        m, n = H.shape

        # Rebuild directed edge list matching NonBacktrackingFlowAnalyzer
        # convention: sorted list of (src, dst) with variable nodes 0..n-1
        # and check nodes n..n+m-1.
        directed_edges: list[tuple[int, int]] = []
        for ci, vi in edges:
            directed_edges.append((vi, n + ci))
            directed_edges.append((n + ci, vi))
        directed_edges.sort()

        de_index = {e: i for i, e in enumerate(directed_edges)}

        scores: dict[tuple[int, int], float] = {}
        for ci, vi in edges:
            j1 = de_index.get((vi, n + ci))
            j2 = de_index.get((n + ci, vi))
            s = 0.0
            if j1 is not None and j1 < len(directed_edge_flow):
                s += abs(float(directed_edge_flow[j1]))
            if j2 is not None and j2 < len(directed_edge_flow):
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

    def _find_valid_rewire(
        self,
        H: np.ndarray,
        ci: int,
        vi: int,
        m: int,
        n: int,
    ) -> tuple[int, int] | None:
        """Find a degree-preserving swap partner for edge (ci, vi).

        Performs a two-edge swap to preserve both row and column degrees:
            remove (ci, vi) and (cj, vj)
            add    (ci, vj) and (cj, vi)
        """
        for cj in range(m):
            if cj == ci:
                continue
            for vj in range(n):
                if vj == vi:
                    continue
                if H[cj, vj] == 0:
                    continue
                if H[ci, vj] != 0 or H[cj, vi] != 0:
                    continue

                if self.avoid_4cycles and self._creates_4cycle(
                    H, ci, vj, cj, vi,
                ):
                    continue

                return (cj, vj)

        return None

    @staticmethod
    def _creates_4cycle(
        H: np.ndarray,
        ci1: int,
        vi1: int,
        ci2: int,
        vi2: int,
    ) -> bool:
        """Check whether adding edges (ci1,vi1) and (ci2,vi2) creates a 4-cycle."""
        m, n = H.shape
        for c in range(m):
            if c == ci1 or c == ci2:
                continue
            if H[c, vi1] != 0 and H[c, vi2] != 0:
                return True
        return False
