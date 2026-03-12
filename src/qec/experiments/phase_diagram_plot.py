"""
v12.5.0 — Phase Diagram Visualization.

Produces ASCII heatmaps and optional matplotlib plots for spectral
instability phase diagrams.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
Matplotlib is optional — ASCII output is always available.
"""

from __future__ import annotations

from typing import Any


_ROUND = 12

# ASCII shading characters ordered by intensity (low → high FER).
_SHADES = " ░▒▓█"


# ── ASCII heatmap ─────────────────────────────────────────────────


def render_ascii_heatmap(
    phase_data: dict[str, Any],
    *,
    sr_bins: int = 10,
    er_bins: int = 10,
) -> str:
    """Render an ASCII heatmap of the spectral instability phase diagram.

    Parameters
    ----------
    phase_data : dict
        Output from SpectralPhaseDiagramGenerator.generate_phase_diagram.
    sr_bins : int
        Number of bins along the spectral radius axis.
    er_bins : int
        Number of bins along the error rate axis.

    Returns
    -------
    str
        Multi-line ASCII heatmap string.
    """
    points = phase_data.get("points", [])
    if not points:
        return "Spectral Instability Phase Diagram\n(no data)\n"

    # Determine axis ranges.
    sr_values = [p["spectral_radius"] for p in points]
    er_values = [p["error_rate"] for p in points]

    sr_min = min(sr_values)
    sr_max = max(sr_values)
    er_min = min(er_values)
    er_max = max(er_values)

    # Avoid degenerate ranges.
    if sr_max <= sr_min:
        sr_max = sr_min + 1.0
    if er_max <= er_min:
        er_max = er_min + 1.0

    sr_step = (sr_max - sr_min) / sr_bins
    er_step = (er_max - er_min) / er_bins

    # Accumulate FER into bins.
    grid: dict[tuple[int, int], list[float]] = {}
    for p in points:
        si = min(int((p["spectral_radius"] - sr_min) / sr_step), sr_bins - 1)
        ei = min(int((p["error_rate"] - er_min) / er_step), er_bins - 1)
        si = max(0, si)
        ei = max(0, ei)
        grid.setdefault((si, ei), []).append(p["FER"])

    # Compute mean FER per cell.
    mean_grid: dict[tuple[int, int], float] = {}
    for key, fers in grid.items():
        mean_grid[key] = sum(fers) / len(fers)

    # Render.
    lines: list[str] = []
    lines.append("Spectral Instability Phase Diagram")
    lines.append("")
    lines.append("error_rate \u2192")
    lines.append("spectral_radius \u2193")
    lines.append("")

    # Column headers (error rate bins).
    header = "        "
    for ei in range(er_bins):
        er_center = er_min + (ei + 0.5) * er_step
        header += f"{er_center:5.3f} "
    lines.append(header)

    for si in range(sr_bins):
        sr_center = sr_min + (si + 0.5) * sr_step
        row = f"{sr_center:6.3f} |"
        for ei in range(er_bins):
            fer = mean_grid.get((si, ei))
            if fer is None:
                row += "  .   "
            else:
                shade_idx = min(int(fer * len(_SHADES)), len(_SHADES) - 1)
                shade_idx = max(0, shade_idx)
                row += f" {_SHADES[shade_idx]}{fer:4.2f} "
        lines.append(row)

    lines.append("")
    lines.append(f"Legend: {' '.join(_SHADES)} (low FER \u2192 high FER)")

    return "\n".join(lines)


# ── Matplotlib plot ───────────────────────────────────────────────


def plot_phase_diagram(
    phase_data: dict[str, Any],
    output_path: str | None = None,
) -> Any | None:
    """Plot a spectral instability phase diagram with matplotlib.

    Parameters
    ----------
    phase_data : dict
        Output from SpectralPhaseDiagramGenerator.generate_phase_diagram.
    output_path : str or None
        If provided, save figure to this path instead of showing.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if matplotlib is available, None otherwise.
    """
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    points = phase_data.get("points", [])
    if not points:
        return None

    sr = [p["spectral_radius"] for p in points]
    er = [p["error_rate"] for p in points]
    fer = [p["FER"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(sr, er, c=fer, cmap="hot", vmin=0.0, vmax=1.0, s=40)
    fig.colorbar(scatter, ax=ax, label="FER")
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Error Rate")
    ax.set_title("Spectral Instability Phase Diagram")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig
