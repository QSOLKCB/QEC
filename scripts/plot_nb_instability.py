#!/usr/bin/env python
"""
Optional visualization of NB eigenvector edge energy on a Tanner graph.

Produces a bar chart of directed-edge energy from the dominant
non-backtracking eigenvector, highlighting trapping-set structure.

Requires matplotlib (not a runtime dependency of the main package).

Usage::

    python scripts/plot_nb_instability.py

This script is for research debugging only.  It does not modify any
decoder or diagnostic state.
"""

from __future__ import annotations

import sys
import os

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.spectral_nb import compute_nb_spectrum


def plot_nb_instability(H: np.ndarray) -> None:
    """Plot NB eigenvector edge energy for a parity-check matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    """
    import matplotlib.pyplot as plt

    spec = compute_nb_spectrum(H)
    edge_energy = spec["edge_energy"]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(edge_energy)), edge_energy)
    plt.xlabel("Directed Edge Index")
    plt.ylabel("NB Eigenvector Energy")
    plt.title("Tanner Graph Instability Map")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo with a small parity-check matrix
    H = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)
    plot_nb_instability(H)
