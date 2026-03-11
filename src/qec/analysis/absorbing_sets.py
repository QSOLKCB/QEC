"""
v11.3.0 — Absorbing Set Predictor.

Estimates absorbing-set risk using short-cycle structure, external
connectivity, and non-backtracking eigenvector localization.

This is a predictor (heuristic), not an exhaustive search.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_MAX_CYCLE_LENGTH = 6
_MAX_CYCLE_ENUM = 10_000
_POWER_ITER = 50

# Risk score weights
_W_INTERNAL = 0.4
_W_LOCALIZATION = 0.4
_W_EXTERNAL = 0.2


class AbsorbingSetPredictor:
    """Estimate absorbing-set risk via cycle motifs and NB localization.

    Parameters
    ----------
    max_cycle_length : int
        Maximum cycle length to enumerate (default 6).
    max_cycles : int
        Maximum number of cycles to enumerate (default 10000).
    """

    def __init__(
        self,
        max_cycle_length: int = _MAX_CYCLE_LENGTH,
        max_cycles: int = _MAX_CYCLE_ENUM,
    ) -> None:
        self.max_cycle_length = max_cycle_length
        self.max_cycles = max_cycles

    def predict(self, H: np.ndarray) -> dict[str, Any]:
        """Predict absorbing-set risk for a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``absorbing_set_risk`` : float — overall risk score in [0, 1]
            - ``num_candidate_absorbing_sets`` : int
            - ``min_candidate_size`` : int
            - ``candidate_sets`` : list[dict]
            - ``localized_variables`` : list[int]
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "absorbing_set_risk": 0.0,
                "num_candidate_absorbing_sets": 0,
                "min_candidate_size": 0,
                "candidate_sets": [],
                "localized_variables": [],
            }

        # Build adjacency lists
        var_to_checks: list[list[int]] = [[] for _ in range(n)]
        check_to_vars: list[list[int]] = [[] for _ in range(m)]
        for ci in range(m):
            for vi in range(n):
                if H_arr[ci, vi] != 0:
                    var_to_checks[vi].append(ci)
                    check_to_vars[ci].append(vi)

        # Sort for determinism
        for lst in var_to_checks:
            lst.sort()
        for lst in check_to_vars:
            lst.sort()

        # Step 1: Enumerate short cycles via adjacency lists
        cycles = self._enumerate_short_cycles(
            n, m, var_to_checks, check_to_vars,
        )

        # Step 2: Build candidate sets from overlapping cycles
        candidates = self._build_candidates(
            cycles, n, m, H_arr, var_to_checks, check_to_vars,
        )

        # Step 3: Compute NB localization
        localization = self._compute_nb_localization(H_arr, n, m)
        localized_variables = self._get_localized_variables(localization, n)

        # Step 4: Score candidates
        scored = self._score_candidates(
            candidates, localization, n, m, H_arr, var_to_checks,
        )

        # Overall risk
        if scored:
            absorbing_set_risk = max(c["risk_score"] for c in scored)
        else:
            absorbing_set_risk = 0.0

        min_size = min((c["size"] for c in scored), default=0)

        return {
            "absorbing_set_risk": round(absorbing_set_risk, 12),
            "num_candidate_absorbing_sets": len(scored),
            "min_candidate_size": min_size,
            "candidate_sets": scored,
            "localized_variables": localized_variables,
        }

    def _enumerate_short_cycles(
        self,
        n: int,
        m: int,
        var_to_checks: list[list[int]],
        check_to_vars: list[list[int]],
    ) -> list[list[int]]:
        """Enumerate short cycles (length 4 and 6) in the Tanner graph.

        Returns cycles as lists of variable-node indices.
        """
        cycles: list[list[int]] = []
        count = 0

        # 4-cycles: two variable nodes sharing two check nodes
        for vi in range(n):
            if count >= self.max_cycles:
                break
            for ci1_idx in range(len(var_to_checks[vi])):
                if count >= self.max_cycles:
                    break
                ci1 = var_to_checks[vi][ci1_idx]
                for ci2_idx in range(ci1_idx + 1, len(var_to_checks[vi])):
                    if count >= self.max_cycles:
                        break
                    ci2 = var_to_checks[vi][ci2_idx]
                    # Find other variables in both checks
                    shared = []
                    for vj in check_to_vars[ci1]:
                        if vj > vi and vj in check_to_vars[ci2]:
                            shared.append(vj)
                    for vj in shared:
                        cycles.append(sorted([vi, vj]))
                        count += 1
                        if count >= self.max_cycles:
                            break

        if self.max_cycle_length >= 6 and count < self.max_cycles:
            # 6-cycles: three variable nodes forming a cycle through checks
            for vi in range(n):
                if count >= self.max_cycles:
                    break
                for ci1 in var_to_checks[vi]:
                    if count >= self.max_cycles:
                        break
                    for vj in check_to_vars[ci1]:
                        if vj <= vi or count >= self.max_cycles:
                            continue
                        for ci2 in var_to_checks[vj]:
                            if ci2 == ci1 or count >= self.max_cycles:
                                continue
                            for vk in check_to_vars[ci2]:
                                if vk <= vj or count >= self.max_cycles:
                                    continue
                                # Check if vk connects back to vi via a check
                                for ci3 in var_to_checks[vk]:
                                    if ci3 != ci1 and ci3 != ci2 and vi in check_to_vars[ci3]:
                                        cycles.append(sorted([vi, vj, vk]))
                                        count += 1
                                        if count >= self.max_cycles:
                                            break
                                    if count >= self.max_cycles:
                                        break
                                if count >= self.max_cycles:
                                    break
                            if count >= self.max_cycles:
                                break
                        if count >= self.max_cycles:
                            break
                    if count >= self.max_cycles:
                        break

        return cycles

    def _build_candidates(
        self,
        cycles: list[list[int]],
        n: int,
        m: int,
        H: np.ndarray,
        var_to_checks: list[list[int]],
        check_to_vars: list[list[int]],
    ) -> list[set[int]]:
        """Build candidate absorbing sets from overlapping cycles.

        Merges cycles that share variable nodes to form candidate sets.
        """
        if not cycles:
            return []

        # Deduplicate cycles
        unique_cycles: list[tuple[int, ...]] = sorted(
            set(tuple(c) for c in cycles)
        )

        # Build variable-to-cycle index
        var_cycles: dict[int, list[int]] = {}
        for idx, cyc in enumerate(unique_cycles):
            for vi in cyc:
                var_cycles.setdefault(vi, []).append(idx)

        # Merge overlapping cycles into candidate sets
        visited_cycles: set[int] = set()
        candidates: list[set[int]] = []

        for idx in range(len(unique_cycles)):
            if idx in visited_cycles:
                continue

            # BFS to find all cycles connected through shared variables
            cluster: set[int] = set()
            queue = [idx]
            visited_cycles.add(idx)

            while queue:
                cur = queue.pop(0)
                cluster.add(cur)
                for vi in unique_cycles[cur]:
                    for neighbor_idx in var_cycles.get(vi, []):
                        if neighbor_idx not in visited_cycles:
                            visited_cycles.add(neighbor_idx)
                            queue.append(neighbor_idx)

            # Collect all variables in this cluster
            var_set: set[int] = set()
            for cyc_idx in cluster:
                for vi in unique_cycles[cyc_idx]:
                    var_set.add(vi)

            candidates.append(var_set)

        return candidates

    def _compute_nb_localization(
        self,
        H: np.ndarray,
        n: int,
        m: int,
    ) -> np.ndarray:
        """Compute NB eigenvector localization via power iteration.

        Returns per-variable-node localization scores.
        Does not construct the full Hashimoto matrix.
        """
        # Build directed-edge representation
        directed_edges: list[tuple[int, int]] = []
        adj: dict[int, list[int]] = {}

        for ci in range(m):
            for vi in range(n):
                if H[ci, vi] != 0:
                    # variable vi -> check (n + ci) and back
                    directed_edges.append((vi, n + ci))
                    directed_edges.append((n + ci, vi))
                    adj.setdefault(vi, []).append(n + ci)
                    adj.setdefault(n + ci, []).append(vi)

        for node in adj:
            adj[node] = sorted(adj[node])
        directed_edges.sort()

        num_de = len(directed_edges)
        if num_de == 0:
            return np.zeros(n, dtype=np.float64)

        edge_index = {e: i for i, e in enumerate(directed_edges)}

        # Power iteration on NB operator
        x = np.ones(num_de, dtype=np.float64)
        x /= np.linalg.norm(x)

        for _ in range(_POWER_ITER):
            y = np.zeros(num_de, dtype=np.float64)
            for i, (u, v) in enumerate(directed_edges):
                for w in adj.get(v, []):
                    if w == u:
                        continue
                    j = edge_index.get((v, w))
                    if j is not None:
                        y[j] += x[i]
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-15:
                return np.zeros(n, dtype=np.float64)
            x = y / norm_y

        # Aggregate edge weights to variable nodes
        var_mass = np.zeros(n, dtype=np.float64)
        for i, (u, v) in enumerate(directed_edges):
            if u < n:
                var_mass[u] += abs(x[i])
            if v < n:
                var_mass[v] += abs(x[i])

        # Normalize to [0, 1]
        max_mass = var_mass.max()
        if max_mass > 0:
            var_mass /= max_mass

        return var_mass

    def _get_localized_variables(
        self,
        localization: np.ndarray,
        n: int,
        threshold: float = 0.5,
    ) -> list[int]:
        """Return variable nodes with localization above threshold."""
        result = sorted(
            [int(vi) for vi in range(n) if localization[vi] > threshold],
            key=lambda vi: (-round(localization[vi], 12), vi),
        )
        return result

    def _score_candidates(
        self,
        candidates: list[set[int]],
        localization: np.ndarray,
        n: int,
        m: int,
        H: np.ndarray,
        var_to_checks: list[list[int]],
    ) -> list[dict[str, Any]]:
        """Score each candidate set for absorbing-set risk."""
        scored: list[dict[str, Any]] = []

        for cand in candidates:
            variables = sorted(cand)
            size = len(variables)
            if size == 0:
                continue

            # Internal score: average internal connectivity
            # (fraction of edges staying within candidate)
            total_edges = 0
            internal_edges = 0
            for vi in variables:
                for ci in var_to_checks[vi]:
                    total_edges += 1
                    # Count how many other variables in this check are in the set
                    for vj_idx in range(n):
                        if H[ci, vj_idx] != 0 and vj_idx != vi and vj_idx in cand:
                            internal_edges += 1
                            break  # only count once per check

            internal_score = internal_edges / max(total_edges, 1)

            # External score: fraction of checks with connections outside set
            check_set: set[int] = set()
            for vi in variables:
                for ci in var_to_checks[vi]:
                    check_set.add(ci)

            external_checks = 0
            for ci in check_set:
                has_external = False
                for vj in range(n):
                    if H[ci, vj] != 0 and vj not in cand:
                        has_external = True
                        break
                if has_external:
                    external_checks += 1

            external_score = external_checks / max(len(check_set), 1)

            # Localization score: mean localization of variables in set
            loc_vals = [localization[vi] for vi in variables]
            localization_score = float(np.mean(loc_vals))

            # Risk score
            risk_score = (
                _W_INTERNAL * internal_score
                + _W_LOCALIZATION * localization_score
                - _W_EXTERNAL * external_score
            )
            risk_score = max(0.0, min(1.0, risk_score))

            scored.append({
                "variables": variables,
                "size": size,
                "internal_score": round(internal_score, 12),
                "external_score": round(external_score, 12),
                "localization_score": round(localization_score, 12),
                "risk_score": round(risk_score, 12),
            })

        # Sort by risk descending, then by size ascending for determinism
        scored.sort(key=lambda c: (-c["risk_score"], c["size"], c["variables"]))

        return scored
