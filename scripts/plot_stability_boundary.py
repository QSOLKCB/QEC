"""
v8.2.0 — Stability Boundary Visualization Script.

Loads the stability dataset artifact and plots spectral_radius vs SIS,
overlaying the learned stability boundary, empirical critical radius,
and spectral predicted critical radius.

Uses matplotlib if available, otherwise prints an ASCII summary.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure repo root is on path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.critical_radius import estimate_critical_spectral_radius
from qec.diagnostics.spectral_critical_line import (
    predict_spectral_critical_radius,
)
from qec.diagnostics.stability_boundary import estimate_stability_boundary


def _load_dataset(path: str) -> list[dict]:
    """Load stability dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def _print_ascii_summary(
    dataset: list[dict],
    boundary: dict,
    critical: dict,
) -> None:
    """Print an ASCII summary of the stability boundary."""
    print("=" * 60)
    print("Stability Boundary Summary")
    print("=" * 60)
    print(f"Observations:       {len(dataset)}")
    print(f"Converged:          {sum(1 for d in dataset if d['bp_converged'])}")
    print(f"Failed:             {sum(1 for d in dataset if not d['bp_converged'])}")
    print(f"Boundary accuracy:  {boundary['accuracy']:.4f}")
    print(f"Critical radius:    {critical['critical_radius']:.6f}")
    print(f"Transition width:   {critical['transition_width']:.6f}")
    print(f"Weights:            {boundary['weights']}")
    print(f"Bias:               {boundary['bias']:.6f}")
    print()

    # ASCII scatter: spectral_radius vs SIS
    print("spectral_radius vs SIS (+ = converged, - = failed)")
    print("-" * 40)
    for obs in sorted(dataset, key=lambda d: d["spectral_radius"]):
        sr = obs["spectral_radius"]
        sis = obs["sis"]
        marker = "+" if obs["bp_converged"] else "-"
        bar = "#" * max(1, int(sis * 100))
        print(f"  SR={sr:6.3f} SIS={sis:8.5f} {marker} {bar}")
    print()


def _plot_matplotlib(
    dataset: list[dict],
    boundary: dict,
    critical: dict,
    spectral_critical: float | None,
) -> None:
    """Plot stability boundary with matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    sr_conv = [d["spectral_radius"] for d in dataset if d["bp_converged"]]
    sis_conv = [d["sis"] for d in dataset if d["bp_converged"]]
    sr_fail = [d["spectral_radius"] for d in dataset if not d["bp_converged"]]
    sis_fail = [d["sis"] for d in dataset if not d["bp_converged"]]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter
    ax.scatter(sr_conv, sis_conv, c="green", marker="o", label="BP converged", alpha=0.7)
    ax.scatter(sr_fail, sis_fail, c="red", marker="x", label="BP failed", alpha=0.7)

    # Critical radius
    cr = critical["critical_radius"]
    ax.axvline(x=cr, color="blue", linestyle="--", label=f"Empirical critical radius ({cr:.3f})")

    # Spectral prediction
    if spectral_critical is not None:
        ax.axvline(
            x=spectral_critical, color="purple", linestyle=":",
            label=f"Spectral predicted ({spectral_critical:.3f})",
        )

    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("SIS")
    ax.set_title("BP Stability Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(_repo_root, "artifacts", "stability_boundary_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main() -> None:
    """Main entry point."""
    dataset_path = os.path.join(_repo_root, "artifacts", "stability_dataset.json")

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Run the stability dataset builder first.")
        return

    dataset = _load_dataset(dataset_path)
    boundary = estimate_stability_boundary(dataset)
    critical = estimate_critical_spectral_radius(dataset)

    _print_ascii_summary(dataset, boundary, critical)

    try:
        import matplotlib  # noqa: F401
        _plot_matplotlib(dataset, boundary, critical, None)
    except ImportError:
        print("matplotlib not available — ASCII summary only.")


if __name__ == "__main__":
    main()
