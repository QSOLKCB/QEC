"""
ququart_lattice_prior.py

High-density lattice prior layer for the 3-ququart [[3,1]]_4 code.

Idea:
    - Work in the 4D logical amplitude space of the code:
        alpha_j = <j_L | psi>, j in {0,1,2,3}
    - Extract a 4D real vector from the real parts Re(alpha_j)
    - Project this vector onto a high-density lattice in R^4
      (square baseline vs D4 / E8-inspired).
    - Rebuild a denoised logical state from the projected amplitudes
      and re-embed it into the full 3-ququart Hilbert space.

This acts as a geometric "pre-decoder" before syndrome decoding:
    geometry snaps noisy logical amplitudes back onto a structured
    high-distance lattice of allowed configurations.

It's an E8-inspired surrogate at d=4, not a full E8 implementation.
"""

import numpy as np
from typing import Literal

from qec_ququart import QuquartRepetitionCode3


class QuquartLatticePrior:
    """
    High-density lattice prior in logical amplitude space.

    mode:
        - 'square': Z^4 baseline
        - 'd4':     D4 lattice (even-sum subset of Z^4), denser packing

    beta:
        - sharpening factor; interpolates between "just renormalize"
          and "project hard" in the logical basis.
    """

    def __init__(
        self,
        code: QuquartRepetitionCode3,
        mode: Literal["square", "d4"] = "d4",
        beta: float = 1.0,
    ):
        self.code = code
        self.mode = mode
        self.beta = beta

        # Precompute logical basis states |j_L>
        self.logical_basis = [self.code.encode_logical(j) for j in range(4)]

    # -----------------------------
    #  LATTICE PROJECTIONS IN R^4
    # -----------------------------

    def _project_square(self, x: np.ndarray) -> np.ndarray:
        """Projection onto Z^4 (elementwise rounding)."""
        return np.round(x)

    def _nearest_d4(self, x: np.ndarray) -> np.ndarray:
        """
        Project onto the D4 lattice:
            D4 = { v ∈ Z^4 : sum(v_i) is even }

        Simple nearest-neighbor projection:
            1. Round to nearest integer vector y
            2. If sum(y) is odd, flip the smallest-magnitude coordinate
        """
        y = np.round(x).astype(int)
        if np.sum(y) % 2 == 0:
            return y.astype(float)

        # Sum is odd: adjust the smallest |component|
        idx = int(np.argmin(np.abs(y)))
        # Flip by ±1 to change parity
        if y[idx] >= 0:
            y[idx] -= 1
        else:
            y[idx] += 1

        # Now sum(y) is even
        return y.astype(float)

    def _project_lattice(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "square":
            return self._project_square(x)
        elif self.mode == "d4":
            return self._nearest_d4(x)
        else:
            raise ValueError(f"Unknown lattice mode: {self.mode}")

    # -----------------------------
    #  LOGICAL AMPLITUDE PROCESSING
    # -----------------------------

    def _extract_logical_amplitudes(self, state: np.ndarray) -> np.ndarray:
        """
        Compute alpha_j = <j_L | state> for j in {0,1,2,3}.

        Returns:
            alpha: shape (4,) complex array
        """
        alpha = []
        for j in range(4):
            alpha.append(np.vdot(self.logical_basis[j], state))
        return np.array(alpha, dtype=complex)

    def _rebuild_logical_state(self, alpha: np.ndarray) -> np.ndarray:
        """
        Given logical amplitudes alpha_j, rebuild the 3-ququart
        state in the codespace:
            |psi> = sum_j alpha_j |j_L>.
        """
        state = np.zeros_like(self.logical_basis[0])
        for j in range(4):
            state += alpha[j] * self.logical_basis[j]
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        return state

    # -----------------------------
    #  PUBLIC API
    # -----------------------------

    def apply_prior(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the lattice prior to a 3-ququart state.

        Steps:
            1. Compute logical amplitudes alpha_j
            2. Extract 4D real vector x = Re(alpha_j)
            3. Optionally sharpen amplitudes via softmax-like beta
            4. Project x to chosen lattice
            5. Combine projected Re(alpha_j) with original Im(alpha_j)
            6. Rebuild logical codespace state and renormalize

        This is a geometric prior in the space of logical
        amplitudes, not a literal physical lattice in real space,
        but it captures the intended "snap-to-structured-grid"
        behavior.
        """
        alpha = self._extract_logical_amplitudes(state)

        # Soft sharpening: emphasize larger |alpha_j|
        if self.beta != 0.0:
            weights = np.exp(self.beta * (np.abs(alpha) ** 2))
            weights /= np.sum(weights)
            alpha = weights * np.exp(1j * np.angle(alpha))

        # Project real parts onto lattice
        x = np.real(alpha)
        x_proj = self._project_lattice(x)

        # Rebuild alpha' with new real parts, old phases
        phases = np.exp(1j * np.angle(alpha + 1e-16))
        alpha_proj = x_proj * phases

        # Rebuild full state in codespace
        return self._rebuild_logical_state(alpha_proj)
