"""
BSC syndrome-structured channel model (v3.8.0 probe).

Extends the uniform BSC syndrome LLR with a deterministic, syndrome-
derived per-variable bias.  The bias breaks the symmetry of the uniform
LLR by incorporating local syndrome information from the Tanner graph.

Mathematical definition
-----------------------
Let ``llr0 = log((1-p) / p)`` be the uniform base LLR (same as
:class:`BSCSyndromeChannel`).

For each variable node *i*, compute a syndrome-incidence score:

    g_i = sum( (2*s[c] - 1) for c in N(i) )

where ``N(i)`` is the set of check indices connected to variable *i*.

Optionally, if ``structured_norm="deg_norm"``, normalise by variable
degree:

    g_i = g_i / max(1, |N(i)|)

The structured LLR is:

    llr[i] = llr0 + kappa * g_i

When ``kappa=0.0`` (default), the output is bit-identical to
:class:`BSCSyndromeChannel`.

No randomness.  No adaptive logic.  Purely deterministic function
of ``(H, s, p, kappa, structured_norm)``.
"""

from __future__ import annotations

import numpy as np

from .base import ChannelModel


class BSCSyndromeStructuredChannel(ChannelModel):
    """BSC syndrome-structured channel: syndrome-biased LLR.

    Parameters
    ----------
    structured_kappa : float
        Scaling factor for the syndrome-incidence bias.
        Default 0.0 → bit-identical to BSCSyndromeChannel.
    structured_norm : str
        ``"none"`` (default) or ``"deg_norm"`` (divide g_i by variable
        degree).
    """

    def __init__(
        self,
        structured_kappa: float = 0.0,
        structured_norm: str = "none",
    ) -> None:
        if structured_norm not in ("none", "deg_norm"):
            raise ValueError(
                f"structured_norm must be 'none' or 'deg_norm', "
                f"got {structured_norm!r}"
            )
        self._kappa = float(structured_kappa)
        self._norm = structured_norm

    def compute_llr(
        self,
        p: float,
        n: int,
        error_vector: np.ndarray | None = None,
        *,
        H: np.ndarray | None = None,
        syndrome_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute structured LLR vector.

        Parameters
        ----------
        p : float
            Physical error probability in (0, 1).
        n : int
            Block length (number of variable nodes).
        error_vector : ndarray or None
            Ignored (syndrome-only channel).
        H : ndarray, optional
            Parity-check matrix, shape (m, n).  Required when
            ``kappa != 0``.
        syndrome_vec : ndarray, optional
            Syndrome vector, shape (m,).  Required when
            ``kappa != 0``.

        Returns
        -------
        ndarray of shape (n,), dtype float64.
        """
        self._validate_probability(p)

        eps = self._EPSILON
        base_llr = np.log((1.0 - p + eps) / (p + eps))
        llr = np.full(n, base_llr, dtype=np.float64)

        # Fast path: kappa=0 → bit-identical to BSCSyndromeChannel.
        if self._kappa == 0.0:
            return llr

        if H is None or syndrome_vec is None:
            raise ValueError(
                "bsc_syndrome_structured with kappa != 0 requires "
                "H and syndrome_vec."
            )

        H_arr = np.asarray(H, dtype=np.uint8)
        s = np.asarray(syndrome_vec, dtype=np.uint8)
        m, n_h = H_arr.shape

        if n_h != n:
            raise ValueError(
                f"H has {n_h} columns but n={n}."
            )

        # Compute syndrome-incidence score g_i for each variable.
        # g_i = sum( (2*s[c] - 1) for c in N(i) )
        # This is equivalent to: g = H^T @ (2*s - 1)
        s_signed = (2.0 * s.astype(np.float64)) - 1.0  # shape (m,)
        g = H_arr.astype(np.float64).T @ s_signed       # shape (n,)

        # Optional degree normalisation.
        if self._norm == "deg_norm":
            # Variable degree = number of checks connected to each variable.
            deg = np.sum(H_arr, axis=0).astype(np.float64)  # shape (n,)
            deg = np.maximum(deg, 1.0)  # avoid division by zero
            g = g / deg

        llr += self._kappa * g

        return llr
