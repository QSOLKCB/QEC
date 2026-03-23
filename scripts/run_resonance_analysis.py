#!/usr/bin/env python
"""v87.3.0 — Resonance analysis + phase diagram demo.

Builds a small deterministic trajectory, runs resonance analysis,
and renders a resonance phase diagram PNG.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def _sample_series():
    """Return a minimal deterministic ternary trajectory."""
    return [
        (1, 0, -1, 1),
        (1, 0, -1, 1),
        (0, 1, 0, -1),
        (0, 1, 0, -1),
        (0, 1, 0, -1),
        (1, -1, 1, 0),
        (1, 0, -1, 1),
        (1, 0, -1, 1),
    ]


def _sample_drift():
    """Return deterministic drift values (len = series - 1)."""
    return [0.0, 0.8, 0.0, 0.0, 0.5, 0.3, 0.0]


def main() -> None:
    from qec.experiments.phase_resonance_analysis import run_resonance_analysis
    from qec.experiments.phase_motif_graph import run_motif_graph_analysis
    from qec.visualization.resonance_phase_plot import (
        plot_resonance_phase_diagram,
    )

    series = _sample_series()
    drift = _sample_drift()

    # Build state graph from motif graph analysis.
    motif_result = run_motif_graph_analysis(series)
    state_graph = motif_result["state_graph"]

    # Run resonance analysis.
    result = run_resonance_analysis(series, drift, state_graph)

    print("Resonance Analysis")
    print(f"  n_locks:            {result['locks']['n_locks']}")
    print(f"  mean_lock_length:   {result['locks']['mean_lock_length']:.2f}")
    print(f"  lock_strength:      {result['lock_strength']:.4f}")
    print(f"  n_attractors:       {result['attractor_field']['n_attractors']}")
    print(f"  field_type:         {result['field_classification']['field_type']}")
    print(f"  confidence:         {result['field_classification']['confidence']:.4f}")

    # Plot.
    tmp = Path(tempfile.gettempdir())
    plot_result = plot_resonance_phase_diagram(
        series,
        drift,
        result["attractor_field"],
        result["locks"],
        output_path=tmp / "resonance_phase_diagram.png",
        mode="debug",
        field_type=result["field_classification"]["field_type"],
    )

    print(f"  plot:               {plot_result['output_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
