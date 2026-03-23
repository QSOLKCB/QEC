#!/usr/bin/env python
"""v85.0.0 — Phase Map Plot demo script.

Builds a small deterministic phase map and renders it as a PNG figure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def main() -> None:
    from qec.visualization.phase_map_plot import plot_phase_map

    phase_map = {
        "nodes": [
            {"id": 0, "range": [0, 9], "dominant_class": "stable",
             "dominant_phase": "aligned", "mean_score": 0.95},
            {"id": 1, "range": [10, 19], "dominant_class": "fragile",
             "dominant_phase": "misaligned", "mean_score": 0.50},
            {"id": 2, "range": [20, 29], "dominant_class": "chaotic",
             "dominant_phase": "unknown", "mean_score": 0.10},
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "phase_boundary",
             "weight": 0.45},
            {"source": 1, "target": 2, "type": "strong_boundary",
             "weight": 0.80},
        ],
    }

    out_path = Path(tempfile.gettempdir()) / "phase_map_plot.png"
    result = plot_phase_map(phase_map, output_path=out_path)

    print("Phase Map Plot")
    print(f"  n_nodes:     {result['n_nodes']}")
    print(f"  n_edges:     {result['n_edges']}")
    print(f"  output_path: {result['output_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
