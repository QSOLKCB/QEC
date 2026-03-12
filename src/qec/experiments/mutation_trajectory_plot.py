"""Render mutation trajectories in spectral basin coordinates."""

from __future__ import annotations

from typing import Any


def render_trajectory_ascii(trajectory: list[dict[str, Any]]) -> str:
    """Render trajectory as deterministic plain-text table."""
    if not trajectory:
        return "Spectral Basin Trajectory\n\n(no data)"

    lines = [
        "Spectral Basin Trajectory",
        "",
        "iter   radius   IPR",
    ]
    for point in trajectory:
        lines.append(
            f"{int(point.get('iteration', 0)):<6} "
            f"{float(point.get('spectral_radius', 0.0)):.2f}     "
            f"{float(point.get('ipr', 0.0)):.3f}",
        )
    return "\n".join(lines)


def plot_trajectory_matplotlib(trajectory: list[dict[str, Any]]) -> Any:
    """Plot trajectory when matplotlib is available; else return ASCII."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return render_trajectory_ascii(trajectory)

    if not trajectory:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Spectral Basin Trajectory")
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
        ax.set_axis_off()
        return fig

    xs = [int(p.get("iteration", 0)) for p in trajectory]
    ys_radius = [float(p.get("spectral_radius", 0.0)) for p in trajectory]
    ys_ipr = [float(p.get("ipr", 0.0)) for p in trajectory]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(xs, ys_radius, marker="o", label="Spectral radius", color="tab:blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Spectral radius", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(xs, ys_ipr, marker="s", label="IPR", color="tab:orange")
    ax2.set_ylabel("IPR", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Spectral Basin Trajectory")
    fig.tight_layout()
    return fig
