"""
v12.5.0 — FER vs Spectral Radius Plot.

Produces ASCII and optional matplotlib plots of FER vs spectral radius
with overlaid mutation strategies.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
Matplotlib is optional.
"""

from __future__ import annotations

from typing import Any


_ROUND = 12

# Strategy display markers for ASCII plots.
_STRATEGY_MARKERS = {
    "baseline": "o",
    "random_swap": "x",
    "nb_swap": "+",
    "nb_ipr_swap": "*",
}


# ── ASCII scatter plot ────────────────────────────────────────────


def render_ascii_fer_vs_spectral_radius(
    ablation_data: dict[str, Any],
    *,
    width: int = 60,
    height: int = 20,
) -> str:
    """Render an ASCII scatter plot of FER vs spectral radius.

    Overlays all four mutation strategies.

    Parameters
    ----------
    ablation_data : dict
        Output from run_ablation containing trials.
    width : int
        Plot width in characters.
    height : int
        Plot height in characters.

    Returns
    -------
    str
        Multi-line ASCII plot string.
    """
    trials = ablation_data.get("trials", [])
    strategies = ["baseline", "random_swap", "nb_swap", "nb_ipr_swap"]

    if not trials:
        return "FER vs Spectral Radius\n(no data)\n"

    # Collect all (sr, fer) pairs per strategy.
    all_sr: list[float] = []
    all_fer: list[float] = []
    for trial in trials:
        for strat in strategies:
            if strat in trial:
                all_sr.append(trial[strat]["spectral_radius"])
                all_fer.append(trial[strat]["FER"])

    if not all_sr:
        return "FER vs Spectral Radius\n(no data)\n"

    sr_min = min(all_sr)
    sr_max = max(all_sr)
    fer_min = min(all_fer)
    fer_max = max(all_fer)

    if sr_max <= sr_min:
        sr_max = sr_min + 1.0
    if fer_max <= fer_min:
        fer_max = fer_min + 0.1

    # Build character grid.
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for trial in trials:
        for strat in strategies:
            if strat not in trial:
                continue
            sr = trial[strat]["spectral_radius"]
            fer = trial[strat]["FER"]

            col = int((sr - sr_min) / (sr_max - sr_min) * (width - 1))
            row = int((1.0 - (fer - fer_min) / (fer_max - fer_min)) * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))

            marker = _STRATEGY_MARKERS.get(strat, ".")
            grid[row][col] = marker

    lines: list[str] = []
    lines.append("FER vs Spectral Radius")
    lines.append("")

    # Y-axis labels.
    for row_idx in range(height):
        fer_val = fer_max - (fer_max - fer_min) * row_idx / max(height - 1, 1)
        label = f"{fer_val:5.3f} |"
        lines.append(label + "".join(grid[row_idx]))

    # X-axis.
    lines.append("       " + "-" * width)
    x_label = f"       {sr_min:<{width // 2}.3f}{sr_max:>{width - width // 2}.3f}"
    lines.append(x_label)
    lines.append("       spectral_radius -->")
    lines.append("")

    # Legend.
    for strat, marker in _STRATEGY_MARKERS.items():
        lines.append(f"  {marker} = {strat}")

    return "\n".join(lines)


# ── Matplotlib plot ───────────────────────────────────────────────


def plot_fer_vs_spectral_radius(
    ablation_data: dict[str, Any],
    output_path: str | None = None,
) -> Any | None:
    """Plot FER vs spectral radius with overlaid mutation strategies.

    Parameters
    ----------
    ablation_data : dict
        Output from run_ablation containing trials.
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

    trials = ablation_data.get("trials", [])
    strategies = ["baseline", "random_swap", "nb_swap", "nb_ipr_swap"]
    colors = {"baseline": "gray", "random_swap": "blue", "nb_swap": "green", "nb_ipr_swap": "red"}
    markers = {"baseline": "o", "random_swap": "x", "nb_swap": "+", "nb_ipr_swap": "*"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for strat in strategies:
        sr_vals = [t[strat]["spectral_radius"] for t in trials if strat in t]
        fer_vals = [t[strat]["FER"] for t in trials if strat in t]
        ax.scatter(
            sr_vals, fer_vals,
            c=colors.get(strat, "black"),
            marker=markers.get(strat, "."),
            label=strat,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("FER")
    ax.set_title("FER vs Spectral Radius by Mutation Strategy")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig
