"""
v12.0.0 — Constraint Tension Analyzer.

Computes a constraint tension functional kappa, representing the
structural shear induced by instability signals on Tanner graphs.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse


_ROUND = 12

# Default tension weights.
_WR = 1.0
_WF = 1.0
_WC = 0.5

# Percentile thresholds for triadic state labels.
_TOP_PERCENTILE = 80
_BOTTOM_PERCENTILE = 20


class ConstraintTensionAnalyzer:
    """Compute constraint tension and triadic state labels.

    Parameters
    ----------
    wr : float
        Weight for residual component (default 1.0).
    wf : float
        Weight for flow component (default 1.0).
    wc : float
        Weight for cluster component (default 0.5).
    """

    def __init__(
        self,
        wr: float = _WR,
        wf: float = _WF,
        wc: float = _WC,
    ) -> None:
        self.wr = wr
        self.wf = wf
        self.wc = wc

    def compute_tension(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        *,
        residual_map: np.ndarray | None = None,
        flow: dict[str, Any] | None = None,
        clusters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute constraint tension functional kappa.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix, shape (m, n).
        residual_map : np.ndarray or None
            Per-variable residual map, shape (n,).
        flow : dict or None
            Output from NonBacktrackingFlowAnalyzer.compute_flow().
        clusters : dict or None
            Output from ResidualClusterAnalyzer.find_clusters().

        Returns
        -------
        dict
            tension : float
            residual_component : float
            flow_component : float
            cluster_component : float
            trapping_component : float
            state_labels : np.ndarray, shape (n,), values in {0, 1, 2}
        """
        if scipy.sparse.issparse(H):
            m, n = H.shape
        else:
            H_arr = np.asarray(H, dtype=np.float64)
            m, n = H_arr.shape

        if n == 0:
            return {
                "tension": 0.0,
                "residual_component": 0.0,
                "flow_component": 0.0,
                "cluster_component": 0.0,
                "trapping_component": 0.0,
                "state_labels": np.zeros(0, dtype=np.int64),
            }

        # --- Residual energy ---
        if residual_map is not None:
            rmap = np.asarray(residual_map, dtype=np.float64)
            r_max = rmap.max()
            if r_max > 1e-15:
                rmap_norm = rmap / r_max
            else:
                rmap_norm = np.zeros(n, dtype=np.float64)
            residual_energy = float(np.mean(rmap_norm ** 2))
        else:
            rmap_norm = np.zeros(n, dtype=np.float64)
            residual_energy = 0.0

        # --- Flow localization ---
        if flow is not None:
            flow_localization = float(flow.get("flow_localization", 0.0))
            variable_flow = np.asarray(
                flow.get("variable_flow", np.zeros(n, dtype=np.float64)),
                dtype=np.float64,
            )
        else:
            flow_localization = 0.0
            variable_flow = np.zeros(n, dtype=np.float64)

        # --- Cluster risk ---
        if clusters is not None:
            cluster_list = clusters.get("clusters", [])
            if cluster_list and n > 0:
                total_cluster_vars = sum(
                    len(c.get("variables", [])) for c in cluster_list
                )
                cluster_risk = min(total_cluster_vars / n, 1.0)
            else:
                cluster_risk = 0.0
        else:
            cluster_risk = 0.0

        # --- Tension functional ---
        residual_component = float(round(residual_energy ** 2, _ROUND))
        flow_component = float(round(flow_localization ** 2, _ROUND))
        cluster_component = float(round(cluster_risk ** 2, _ROUND))
        trapping_component = 0.0  # Reserved for future trapping-set integration.

        tension = float(round(
            self.wr * residual_component
            + self.wf * flow_component
            + self.wc * cluster_component,
            _ROUND,
        ))

        # --- Triadic state labels ---
        # Combine flow and residual for classification.
        combined = variable_flow + rmap_norm

        if combined.max() > 1e-15:
            top_threshold = float(np.percentile(combined, _TOP_PERCENTILE))
            bottom_threshold = float(np.percentile(combined, _BOTTOM_PERCENTILE))
        else:
            top_threshold = 0.0
            bottom_threshold = 0.0

        state_labels = np.zeros(n, dtype=np.int64)
        for vi in range(n):
            if combined[vi] >= top_threshold and top_threshold > 1e-15:
                state_labels[vi] = 1  # perturbed
            elif combined[vi] <= bottom_threshold:
                state_labels[vi] = 2  # corrective
            else:
                state_labels[vi] = 0  # stable

        return {
            "tension": tension,
            "residual_component": residual_component,
            "flow_component": flow_component,
            "cluster_component": cluster_component,
            "trapping_component": trapping_component,
            "state_labels": state_labels,
        }
