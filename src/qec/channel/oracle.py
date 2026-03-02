"""
Oracle channel model — backward-compatible with ``channel_llr()``.

Produces per-variable LLR whose sign encodes the true error position.
This gives the decoder perfect side-information and yields near-zero FER
at moderate noise levels.
"""

from __future__ import annotations

import numpy as np

from .base import ChannelModel


class OracleChannel(ChannelModel):
    """Oracle channel: LLR sign derived from the true error vector.

    Output is numerically identical to :func:`src.qec_qldpc_codes.channel_llr`
    called without bias, preserving byte-identical benchmark artifacts.
    """

    def compute_llr(
        self,
        p: float,
        n: int,
        error_vector: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        if error_vector is None:
            raise ValueError("Oracle channel requires error_vector.")
        self._validate_probability(p)

        eps = self._EPSILON
        base_llr = np.log((1.0 - p + eps) / (p + eps))
        sign = 1 - 2 * np.asarray(error_vector, dtype=np.float64)
        return base_llr * sign
