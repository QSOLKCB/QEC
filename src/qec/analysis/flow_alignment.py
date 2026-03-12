"""
v12.1.0 — Flow Alignment Analyzer.

Measures alignment between BP residual gradient and non-backtracking
flow eigenvector direction.  High alignment indicates true structural
instability rather than transient decoder noise.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12

_STRONG_THRESHOLD = 0.6
_MODERATE_THRESHOLD = 0.3


class FlowAlignmentAnalyzer:
    """Compute alignment between NB flow vector and BP residual vector.

    The cosine similarity between these vectors indicates whether
    non-backtracking flow direction predicts decoder instability regions.
    """

    def compute_alignment(
        self,
        residual_map: np.ndarray,
        variable_flow: np.ndarray,
    ) -> dict[str, Any]:
        """Compute alignment between BP residual and NB flow vectors.

        Parameters
        ----------
        residual_map : np.ndarray
            BP residual magnitude per variable node, shape (n,).
        variable_flow : np.ndarray
            NB flow per variable node, shape (n,).

        Returns
        -------
        dict
            alignment_score : float — cosine similarity in [-1, 1].
            residual_norm : float — L2 norm of residual_map.
            flow_norm : float — L2 norm of variable_flow.
            alignment_type : str — "strong", "moderate", or "weak".
        """
        residual = np.asarray(residual_map, dtype=np.float64)
        flow = np.asarray(variable_flow, dtype=np.float64)

        residual_norm = float(np.linalg.norm(residual))
        flow_norm = float(np.linalg.norm(flow))

        if residual_norm < 1e-15 or flow_norm < 1e-15:
            return {
                "alignment_score": 0.0,
                "residual_norm": round(residual_norm, _ROUND),
                "flow_norm": round(flow_norm, _ROUND),
                "alignment_type": "weak",
            }

        dot = float(np.dot(flow, residual))
        score = dot / (flow_norm * residual_norm)

        # Clamp to [-1, 1] for numerical safety.
        score = max(-1.0, min(1.0, score))
        score = round(score, _ROUND)

        if score >= _STRONG_THRESHOLD:
            alignment_type = "strong"
        elif score >= _MODERATE_THRESHOLD:
            alignment_type = "moderate"
        else:
            alignment_type = "weak"

        return {
            "alignment_score": score,
            "residual_norm": round(residual_norm, _ROUND),
            "flow_norm": round(flow_norm, _ROUND),
            "alignment_type": alignment_type,
        }
