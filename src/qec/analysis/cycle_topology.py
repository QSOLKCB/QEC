"""
v11.3.0 — Cycle Topology Analyzer.

Computes deterministic cycle irregularity metrics for LDPC/QLDPC
Tanner graphs.  Measures degree-twist (variance of node degrees
along cycles) to identify irregular cycle structures.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_MAX_CYCLE_LENGTH = 6
_MAX_CYCLE_ENUM = 10_000


class CycleTopologyAnalyzer:
    """Compute cycle irregularity metrics for a Tanner graph.

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

    def analyze(self, H: np.ndarray) -> dict[str, Any]:
        """Compute cycle topology metrics for a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``num_short_cycles`` : int
            - ``cycle_parity_histogram`` : dict — {cycle_length: count}
            - ``mean_degree_twist`` : float
            - ``max_degree_twist`` : float
            - ``twisted_cycle_fraction`` : float
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "num_short_cycles": 0,
                "cycle_parity_histogram": {},
                "mean_degree_twist": 0.0,
                "max_degree_twist": 0.0,
                "twisted_cycle_fraction": 0.0,
            }

        # Build adjacency lists
        var_to_checks: list[list[int]] = [[] for _ in range(n)]
        check_to_vars: list[list[int]] = [[] for _ in range(m)]
        for ci in range(m):
            for vi in range(n):
                if H_arr[ci, vi] != 0:
                    var_to_checks[vi].append(ci)
                    check_to_vars[ci].append(vi)

        for lst in var_to_checks:
            lst.sort()
        for lst in check_to_vars:
            lst.sort()

        # Compute variable node degrees
        var_degrees = np.array(
            [len(var_to_checks[vi]) for vi in range(n)],
            dtype=np.float64,
        )

        # Enumerate short cycles and collect variable nodes per cycle
        cycles = self._enumerate_cycles_with_length(
            n, m, var_to_checks, check_to_vars,
        )

        num_short_cycles = len(cycles)

        # Cycle parity histogram (by number of variable nodes in cycle)
        parity_hist: dict[int, int] = {}
        for length, _ in cycles:
            parity_hist[length] = parity_hist.get(length, 0) + 1

        # Degree-twist metrics
        twists: list[float] = []
        twist_threshold = 0.0  # any variance > 0 counts as twisted

        for _, var_nodes in cycles:
            if len(var_nodes) < 2:
                twists.append(0.0)
                continue

            degrees = var_degrees[var_nodes]
            mean_deg = float(np.mean(degrees))
            if mean_deg > 0:
                # Normalized variance: variance / mean^2
                twist = float(np.var(degrees)) / (mean_deg * mean_deg)
            else:
                twist = 0.0
            twists.append(twist)

        if twists:
            mean_twist = float(np.mean(twists))
            max_twist = float(np.max(twists))
            twisted_count = sum(1 for t in twists if t > twist_threshold)
            twisted_fraction = twisted_count / len(twists)
        else:
            mean_twist = 0.0
            max_twist = 0.0
            twisted_fraction = 0.0

        return {
            "num_short_cycles": num_short_cycles,
            "cycle_parity_histogram": dict(sorted(parity_hist.items())),
            "mean_degree_twist": round(mean_twist, 12),
            "max_degree_twist": round(max_twist, 12),
            "twisted_cycle_fraction": round(twisted_fraction, 12),
        }

    def _enumerate_cycles_with_length(
        self,
        n: int,
        m: int,
        var_to_checks: list[list[int]],
        check_to_vars: list[list[int]],
    ) -> list[tuple[int, list[int]]]:
        """Enumerate short cycles, returning (cycle_length, variable_nodes).

        cycle_length is the number of edges in the Tanner graph cycle
        (always even for bipartite graphs).
        """
        cycles: list[tuple[int, list[int]]] = []
        count = 0

        # 4-cycles (length 4 in Tanner graph = 2 variable nodes)
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
                    for vj in check_to_vars[ci1]:
                        if count >= self.max_cycles:
                            break
                        if vj > vi and vj in check_to_vars[ci2]:
                            cycles.append((4, sorted([vi, vj])))
                            count += 1

        if self.max_cycle_length >= 6 and count < self.max_cycles:
            # 6-cycles (length 6 in Tanner graph = 3 variable nodes)
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
                                for ci3 in var_to_checks[vk]:
                                    if ci3 != ci1 and ci3 != ci2 and vi in check_to_vars[ci3]:
                                        cycles.append((6, sorted([vi, vj, vk])))
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
