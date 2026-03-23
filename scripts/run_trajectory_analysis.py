#!/usr/bin/env python
"""v86.2.0 — Spectral trajectory visualization + sonification demo.

Builds a small deterministic trajectory, then renders both a 3-panel
PNG plot and a WAV audio file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def _sample_trajectory():
    """Return a minimal deterministic trajectory dict."""
    return {
        "n_steps": 5,
        "spectral_trajectory": [],
        "drift": [0.8, 0.5, 0.3, 0.1],
        "lambda_max": [1.0, 1.2, 1.1, 0.9, 0.85],
        "rank_evolution": [4, 4, 3, 3, 3],
        "degeneracy_evolution": [0, 1, 1, 2, 2],
        "temporal_transitions": [
            {"time_index": 0, "drift": 0.8},
            {"time_index": 1, "drift": 0.5},
        ],
        "trajectory_type": "convergent",
    }


def main() -> None:
    from qec.visualization.trajectory_plot import plot_spectral_trajectory
    from qec.visualization.trajectory_audio import sonify_spectral_trajectory

    traj = _sample_trajectory()
    tmp = Path(tempfile.gettempdir())

    plot_result = plot_spectral_trajectory(
        traj,
        output_path=tmp / "spectral_trajectory.png",
        mode="debug",
    )

    audio_result = sonify_spectral_trajectory(
        traj,
        output_path=tmp / "spectral_trajectory.wav",
    )

    print("Spectral Trajectory Plot")
    print(f"  n_steps:     {plot_result['n_steps']}")
    print(f"  output_path: {plot_result['output_path']}")
    print("Spectral Trajectory Audio")
    print(f"  duration:    {audio_result['duration']:.3f}s")
    print(f"  output_path: {audio_result['output_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
