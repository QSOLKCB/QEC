#!/usr/bin/env python
"""v84.5.0 — Phase Map JSON Export demo script.

Builds a tiny deterministic target sweep, extracts the phase map,
and saves it to a local JSON file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def main() -> None:
    from qec.experiments.hybrid_target_sweep import save_phase_map

    phase_map = {
        "nodes": [
            {"id": 0, "label": "stable", "score": 0.95},
            {"id": 1, "label": "boundary", "score": 0.50},
            {"id": 2, "label": "chaotic", "score": 0.10},
        ],
        "edges": [
            {"source": 0, "target": 1, "weight": 0.45},
            {"source": 1, "target": 2, "weight": 0.40},
        ],
    }

    out_path = Path(tempfile.gettempdir()) / "phase_map_export.json"
    meta = save_phase_map(phase_map, out_path)

    print("Phase Map Export")
    print(f"  output_path: {meta['output_path']}")
    print(f"  n_nodes:     {meta['n_nodes']}")
    print(f"  n_edges:     {meta['n_edges']}")
    print("Done.")


if __name__ == "__main__":
    main()
