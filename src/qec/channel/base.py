"""
Abstract base class for channel models.
"""

from __future__ import annotations

import numpy as np


class ChannelModel:
    """Base interface for channel LLR computation.

    Subclasses must implement :meth:`compute_llr` which returns a
    per-variable log-likelihood ratio vector of shape ``(n,)``.
    """

    _EPSILON = 1e-30

    def _validate_probability(self, p: float) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")

    def compute_llr(
        self,
        p: float,
        n: int,
        error_vector: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute LLR vector for a given channel error probability.

        Parameters
        ----------
        p : float
            Physical error probability in (0, 1).
        n : int
            Block length (number of variable nodes).
        error_vector : ndarray or None
            True error vector.  Required by oracle channels; ignored by
            syndrome-only channels.
        **kwargs
            Additional keyword arguments for extended channel models
            (e.g. ``H``, ``syndrome_vec`` for structured channels).

        Returns
        -------
        ndarray of shape (n,), dtype float64.
        """
        raise NotImplementedError
