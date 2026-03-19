"""
v7.9.0 — Plot Incremental Spectral Convergence.

Plots iteration vs eigenvalue estimate for the warm-started
power iteration solver.  Demonstrates convergence acceleration
from warm-starting with previous eigenvector.

Matplotlib is optional — the script exits gracefully if not installed.
"""

from __future__ import annotations

import sys
import os

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum
from qec.diagnostics._spectral_utils import build_nb_operator
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
)


def _collect_convergence_trace(
    H: np.ndarray,
    previous_eigenvector: np.ndarray,
    max_iter: int = 30,
) -> list[dict]:
    """Run warm-start power iteration and record per-iteration estimates."""
    graph = _TannerGraph(H)
    op, directed_edges = build_nb_operator(graph)
    n_edges = len(directed_edges)

    v = np.array(previous_eigenvector, dtype=np.float64)
    if len(v) != n_edges:
        v = np.ones(n_edges, dtype=np.float64)

    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm

    trace = []

    for it in range(max_iter):
        w = op.matvec(v)
        lam = np.dot(v, w)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            break
        v_new = w / norm_w
        diff = np.linalg.norm(v_new - v)

        trace.append({
            "iteration": it + 1,
            "eigenvalue_estimate": abs(lam),
            "residual": diff,
        })

        v = v_new
        if diff < 1e-10:
            break

    return trace


def main():
    """Generate convergence plot for a small example."""
    # Small parity-check matrix
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)

    # Compute original spectrum
    orig = compute_nb_spectrum(H)
    prev_eigvec = orig["eigenvector"]

    # Generate a repair candidate and apply it
    candidates = propose_repair_candidates(
        H, top_k_edges=5, max_candidates=10,
    )

    if not candidates:
        print("No repair candidates available.")
        return

    H_repaired = apply_repair_candidate(H, candidates[0])

    # Collect convergence trace
    trace = _collect_convergence_trace(H_repaired, prev_eigvec, max_iter=30)

    if not trace:
        print("Power iteration produced no iterations.")
        return

    # Print text summary
    print("Iteration | Eigenvalue Estimate | Residual")
    print("-" * 50)
    for entry in trace:
        print(
            f"  {entry['iteration']:>5d}   | "
            f"  {entry['eigenvalue_estimate']:>18.12f} | "
            f"  {entry['residual']:.2e}"
        )

    # Attempt matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not installed — skipping plot.")
        return

    iterations = [e["iteration"] for e in trace]
    eigenvalues = [e["eigenvalue_estimate"] for e in trace]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(iterations, eigenvalues, "o-", color="steelblue", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Eigenvalue Estimate")
    ax.set_title("Warm-Start Power Iteration Convergence (v7.9.0)")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(_repo_root, "incremental_convergence.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
