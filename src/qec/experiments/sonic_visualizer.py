"""
v74.3.0 — Dual-Lattice Invariant Visualizer (Sierpinski + Rubik).

Deterministic visualization layer that maps:
- Rubik lattice (8x8x8)  → measurement / signal field
- Sierpinski lattice (3x3x3) → interpretation / classification field

Reads pre-computed outputs from v74.0 (per-file analysis.json) and
v74.2 (sequence_analysis.json).  Does NOT recompute analysis.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUBIK_SIZE = 8
SIERPINSKI_SIZE = 3

# Deterministic spectral band colors (RGB 0-1) for Rubik voxels.
# 8 bands from low-frequency to high-frequency.
SPECTRAL_BAND_COLORS: List[Tuple[float, float, float]] = [
    (0.1, 0.1, 0.6),   # deep blue   — lowest band
    (0.1, 0.4, 0.8),   # blue
    (0.1, 0.7, 0.7),   # cyan
    (0.1, 0.8, 0.3),   # green
    (0.6, 0.8, 0.1),   # yellow-green
    (0.9, 0.7, 0.1),   # orange
    (0.9, 0.3, 0.1),   # red-orange
    (0.8, 0.1, 0.1),   # red         — highest band
]

# Classification colors for Sierpinski nodes (RGB 0-1).
CLASSIFICATION_COLORS: Dict[str, Tuple[float, float, float]] = {
    "stable":     (0.0, 0.0, 1.0),   # blue
    "divergent":  (1.0, 1.0, 0.0),   # yellow
    "transition": (1.0, 1.0, 1.0),   # white
    "collapse":   (1.0, 0.0, 0.0),   # red
    "recovery":   (0.0, 1.0, 0.0),   # green
}

# Default color for unknown classifications.
_DEFAULT_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Activation threshold for Rubik voxels (normalised energy).
ACTIVATION_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data Loading (read-only, no recomputation)
# ---------------------------------------------------------------------------

def load_analysis(path: str) -> Dict[str, Any]:
    """Load a per-file analysis.json produced by v74.0.

    Returns a *deep copy* so callers cannot mutate the source data.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return copy.deepcopy(data)


def load_sequence_analysis(path: str) -> Dict[str, Any]:
    """Load sequence_analysis.json produced by v74.2.

    Returns a *deep copy* so callers cannot mutate the source data.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return copy.deepcopy(data)


# ---------------------------------------------------------------------------
# Rubik Lattice (8x8x8) — Measurement Field
# ---------------------------------------------------------------------------

def build_rubik_lattice(
    analysis: Dict[str, Any],
    *,
    n_bins: int = RUBIK_SIZE,
) -> np.ndarray:
    """Build an 8x8x8 voxel grid from a per-file analysis.

    Axes:
        axis-0 (x): frequency bin  — derived from fft_top_peaks
        axis-1 (y): energy level   — quantised normalised magnitude
        axis-2 (z): time/channel   — distributed uniformly for single-frame

    Returns
    -------
    np.ndarray
        Shape ``(8, 8, 8)`` with values in ``[0.0, 1.0]`` representing
        normalised voxel intensity.
    """
    grid = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)

    peaks = analysis.get("fft_top_peaks", [])
    if not peaks:
        return grid

    # Collect frequencies and magnitudes.
    freqs = np.array([p["frequency_hz"] for p in peaks], dtype=np.float64)
    mags = np.array([p["magnitude"] for p in peaks], dtype=np.float64)

    # Normalise magnitudes to [0, 1].
    mag_max = np.max(mags) if len(mags) > 0 else 1.0
    if mag_max == 0.0:
        mag_max = 1.0
    norm_mags = mags / mag_max

    # Normalise frequencies to [0, 1] for binning.
    freq_min = np.min(freqs) if len(freqs) > 1 else 0.0
    freq_max = np.max(freqs) if len(freqs) > 1 else (freqs[0] + 1.0)
    freq_range = freq_max - freq_min
    if freq_range == 0.0:
        freq_range = 1.0
    norm_freqs = (freqs - freq_min) / freq_range

    # Global features for z-axis modulation.
    rms = float(analysis.get("rms_energy", 0.0))
    centroid = float(analysis.get("spectral_centroid_hz", 0.0))
    spread = float(analysis.get("spectral_spread_hz", 0.0))

    # Quantise into bins deterministically.
    for i, (nf, nm) in enumerate(zip(norm_freqs, norm_mags)):
        x = int(np.clip(nf * (n_bins - 1), 0, n_bins - 1))
        y = int(np.clip(nm * (n_bins - 1), 0, n_bins - 1))
        # z-axis: distribute across slices using peak index modulo.
        z = i % n_bins
        # Intensity is the normalised magnitude.
        grid[x, y, z] = max(grid[x, y, z], nm)

    # Overlay global energy as a base layer on z=0.
    if rms > 0:
        peak_amp = float(analysis.get("peak_amplitude", 1.0))
        if peak_amp == 0.0:
            peak_amp = 1.0
        energy_level = min(rms / peak_amp, 1.0)
        for x in range(n_bins):
            for y_idx in range(max(1, int(energy_level * n_bins))):
                grid[x, y_idx, 0] = max(grid[x, y_idx, 0], energy_level * 0.3)

    return grid


def rubik_colors(grid: np.ndarray) -> np.ndarray:
    """Assign RGBA colors to each voxel based on spectral band.

    Parameters
    ----------
    grid : np.ndarray
        Shape ``(8, 8, 8)`` intensity grid.

    Returns
    -------
    np.ndarray
        Shape ``(8, 8, 8, 4)`` RGBA array.
    """
    n = grid.shape[0]
    colors = np.zeros((*grid.shape, 4), dtype=np.float64)
    for x in range(n):
        r, g, b = SPECTRAL_BAND_COLORS[x % len(SPECTRAL_BAND_COLORS)]
        for y in range(n):
            for z in range(n):
                intensity = grid[x, y, z]
                if intensity >= ACTIVATION_THRESHOLD:
                    colors[x, y, z] = (r, g, b, intensity)
    return colors


# ---------------------------------------------------------------------------
# Sierpinski Lattice (3x3x3) — Interpretation Field
# ---------------------------------------------------------------------------

def _sierpinski_node_positions() -> List[Tuple[int, int, int]]:
    """Return the 20 active node positions in a 3x3x3 Sierpinski lattice.

    A Sierpinski tetrahedron at recursion-1 in a 3x3x3 grid keeps only
    specific nodes (not a full cube).  We use a fixed deterministic set.
    """
    # Corners of the tetrahedron in a 3x3x3 grid.
    corners = [(0, 0, 0), (2, 0, 0), (1, 2, 0), (1, 1, 2)]
    # Edge midpoints.
    midpoints = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 1, 0), (2, 0, 1), (1, 1, 0),
        (1, 0, 1), (0, 1, 1), (1, 2, 1),
        (1, 1, 1),
    ]
    # Centre node.
    nodes = sorted(set(corners + midpoints))
    return nodes


SIERPINSKI_NODES = _sierpinski_node_positions()


def build_sierpinski_lattice(
    sequence: Dict[str, Any],
) -> Dict[str, Any]:
    """Build Sierpinski lattice state from sequence_analysis.json.

    Returns a dict with:
        - ``nodes``: list of (x, y, z) positions
        - ``activations``: list of activation values [0.0, 1.0]
        - ``classifications``: list of classification labels per node
        - ``colors``: list of (r, g, b, a) per node
    """
    transitions = sequence.get("transitions", [])
    nodes = list(SIERPINSKI_NODES)
    n_nodes = len(nodes)

    # Count classifications across all transitions.
    class_counts: Dict[str, int] = {
        "stable": 0,
        "divergent": 0,
        "transition": 0,
        "collapse": 0,
        "recovery": 0,
    }
    for t in transitions:
        label = t.get("classification", "stable")
        if label in class_counts:
            class_counts[label] += 1

    # Total transitions for normalisation.
    total = max(sum(class_counts.values()), 1)

    # Assign classifications to nodes deterministically.
    # Distribute nodes proportionally across classification types.
    classifications: List[str] = []
    activations: List[float] = []
    colors_list: List[Tuple[float, float, float, float]] = []

    # Build ordered assignment list.
    ordered_labels: List[str] = []
    for label in ["stable", "divergent", "transition", "collapse", "recovery"]:
        count = class_counts[label]
        n_assigned = max(1, round(count / total * n_nodes)) if count > 0 else 0
        ordered_labels.extend([label] * n_assigned)

    # Pad or trim to n_nodes.
    while len(ordered_labels) < n_nodes:
        ordered_labels.append("stable")
    ordered_labels = ordered_labels[:n_nodes]

    for label in ordered_labels:
        classifications.append(label)
        activation = class_counts.get(label, 0) / total
        activations.append(activation)
        r, g, b = CLASSIFICATION_COLORS.get(label, _DEFAULT_COLOR)
        colors_list.append((r, g, b, max(0.2, activation)))

    return {
        "nodes": nodes,
        "activations": activations,
        "classifications": classifications,
        "colors": colors_list,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    """Import matplotlib with Agg backend. Returns (matplotlib, plt, Axes3D)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    return matplotlib, plt


def render_rubik(
    grid: np.ndarray,
    output_path: str,
    *,
    dpi: int = 150,
    title: str = "Rubik Lattice (8x8x8) — Signal Field",
) -> str:
    """Render the Rubik lattice as a 3D voxel plot and save to PNG.

    Returns the output path.
    """
    _, plt = _ensure_matplotlib()

    activated = grid >= ACTIVATION_THRESHOLD
    colors = rubik_colors(grid)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(activated, facecolors=colors, edgecolor=(0.2, 0.2, 0.2, 0.1))
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Energy Level")
    ax.set_zlabel("Channel / Slice")
    ax.set_title(title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_sierpinski(
    lattice: Dict[str, Any],
    output_path: str,
    *,
    dpi: int = 150,
    title: str = "Sierpinski Lattice (3x3x3) — Interpretation Field",
) -> str:
    """Render the Sierpinski lattice as a 3D scatter plot and save to PNG.

    Returns the output path.
    """
    _, plt = _ensure_matplotlib()

    nodes = lattice["nodes"]
    colors = lattice["colors"]
    activations = lattice["activations"]

    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    zs = [n[2] for n in nodes]
    # Scale marker size by activation.
    sizes = [max(40, a * 300) for a in activations]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolors="black", linewidth=0.5)

    # Add classification labels.
    for i, (x, y, z) in enumerate(nodes):
        label = lattice["classifications"][i]
        if activations[i] > 0.1:
            ax.text(x, y, z + 0.15, label, fontsize=6, ha="center")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_combined(
    grid: np.ndarray,
    lattice: Dict[str, Any],
    output_path: str,
    *,
    dpi: int = 150,
) -> str:
    """Render both lattices side-by-side and save to PNG.

    Returns the output path.
    """
    _, plt = _ensure_matplotlib()

    fig = plt.figure(figsize=(16, 8))

    # Left: Rubik
    ax1 = fig.add_subplot(121, projection="3d")
    activated = grid >= ACTIVATION_THRESHOLD
    colors = rubik_colors(grid)
    ax1.voxels(activated, facecolors=colors, edgecolor=(0.2, 0.2, 0.2, 0.1))
    ax1.set_xlabel("Frequency Bin")
    ax1.set_ylabel("Energy Level")
    ax1.set_zlabel("Channel / Slice")
    ax1.set_title("Rubik (8x8x8) — Signal")

    # Right: Sierpinski
    ax2 = fig.add_subplot(122, projection="3d")
    nodes = lattice["nodes"]
    scolors = lattice["colors"]
    activations = lattice["activations"]
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    zs = [n[2] for n in nodes]
    sizes = [max(40, a * 300) for a in activations]
    ax2.scatter(xs, ys, zs, c=scolors, s=sizes, edgecolors="black", linewidth=0.5)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Sierpinski (3x3x3) — Interpretation")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------

def visualize(
    analysis_paths: List[str],
    sequence_path: str,
    output_dir: str = "artifacts/sonic_visualization",
) -> Dict[str, str]:
    """Run the full dual-lattice visualisation pipeline.

    Parameters
    ----------
    analysis_paths : list[str]
        Paths to per-file analysis.json files (v74.0 outputs).
    sequence_path : str
        Path to sequence_analysis.json (v74.2 output).
    output_dir : str
        Directory for output PNGs.

    Returns
    -------
    dict
        Mapping of artifact name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}

    # Load sequence analysis for Sierpinski.
    sequence = load_sequence_analysis(sequence_path)
    sierpinski = build_sierpinski_lattice(sequence)

    sierpinski_path = os.path.join(output_dir, "sierpinski_state.png")
    render_sierpinski(sierpinski, sierpinski_path)
    outputs["sierpinski_state"] = sierpinski_path

    # Build Rubik for each analysis file; use first for combined view.
    first_grid: Optional[np.ndarray] = None
    for i, apath in enumerate(analysis_paths):
        analysis = load_analysis(apath)
        grid = build_rubik_lattice(analysis)
        fname = Path(apath).parent.name or f"state_{i}"
        rubik_path = os.path.join(output_dir, f"rubik_frame_{fname}.png")
        render_rubik(grid, rubik_path, title=f"Rubik (8x8x8) — {fname}")
        outputs[f"rubik_frame_{fname}"] = rubik_path
        if first_grid is None:
            first_grid = grid

    # Combined view using first analysis.
    if first_grid is not None:
        combined_path = os.path.join(output_dir, "combined_view.png")
        render_combined(first_grid, sierpinski, combined_path)
        outputs["combined_view"] = combined_path

    return outputs
