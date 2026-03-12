"""
v12.2.0 — Eigenvector Localization Analyzer (IPR).

Detects when the dominant NB eigenvector concentrates on a small
subset of variable nodes, a common signature of trapping sets.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


# IPR classification thresholds.
IPR_DISTRIBUTED_UPPER = 0.05
IPR_MILD_UPPER = 0.15

_ROUND = 12


class EigenvectorLocalizationAnalyzer:
    """Analyse eigenvector localization via Inverse Participation Ratio."""

    @staticmethod
    def compute_ipr(eigenvector: np.ndarray) -> dict:
        """Compute IPR and classify localization.

        Parameters
        ----------
        eigenvector : np.ndarray
            Real-valued eigenvector.

        Returns
        -------
        dict
            ipr : float in [0, 1], rounded to 12 decimals.
            localization_type : str
            vector_norm : float
        """
        v = np.asarray(eigenvector, dtype=np.float64).ravel()
        vector_norm = float(np.linalg.norm(v))

        sum_v2 = float(np.sum(v ** 2))
        if sum_v2 < 1e-30:
            ipr = 0.0
        else:
            sum_v4 = float(np.sum(v ** 4))
            ipr = sum_v4 / (sum_v2 ** 2)

        # Clamp and round for determinism.
        ipr = max(0.0, min(1.0, ipr))
        ipr = round(ipr, _ROUND)

        if ipr < IPR_DISTRIBUTED_UPPER:
            localization_type = "distributed"
        elif ipr < IPR_MILD_UPPER:
            localization_type = "mild_localization"
        else:
            localization_type = "strong_localization"

        return {
            "ipr": ipr,
            "localization_type": localization_type,
            "vector_norm": round(vector_norm, _ROUND),
        }
