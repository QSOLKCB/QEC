#!/usr/bin/env python
"""v85.3.0 — Phase Map Plot demo script.

Builds a small deterministic phase map and renders it as a PNG figure
with annotated overlays.  Produces both "debug" and "paper" exports.
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

    interface_ranking = {
        "ranked_interfaces": [
            {"from_index": 1, "to_index": 2, "strength": 0.80},
            {"from_index": 0, "to_index": 1, "strength": 0.45},
        ],
        "strongest_interface": {
            "from_index": 1, "to_index": 2, "strength": 0.80,
        },
    }

    transition_summary = {
        "n_transitions": 2,
        "max_delta_score": 0.45,
        "class_change_count": 2,
        "phase_change_count": 2,
    }

    tmp = Path(tempfile.gettempdir())

    # debug export (full detail)
    debug_path = tmp / "phase_map_debug.png"
    debug_result = plot_phase_map(
        phase_map,
        output_path=debug_path,
        interface_ranking=interface_ranking,
        transition_summary=transition_summary,
        mode="debug",
    )

    # paper export (clean, publication-ready)
    paper_path = tmp / "phase_map_paper.png"
    paper_result = plot_phase_map(
        phase_map,
        output_path=paper_path,
        interface_ranking=interface_ranking,
        transition_summary=transition_summary,
        mode="paper",
    )

    print("Phase Map Plot — debug")
    print(f"  n_nodes:     {debug_result['n_nodes']}")
    print(f"  n_edges:     {debug_result['n_edges']}")
    print(f"  output_path: {debug_result['output_path']}")
    print("Phase Map Plot — paper")
    print(f"  n_nodes:     {paper_result['n_nodes']}")
    print(f"  n_edges:     {paper_result['n_edges']}")
    print(f"  output_path: {paper_result['output_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
