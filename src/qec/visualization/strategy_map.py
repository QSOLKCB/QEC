"""v102.0.0 — ASCII strategy map visualization.

Renders embedded 2D strategy points on a fixed-size ASCII grid.
No external dependencies (no plotting libraries).

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure text generation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Grid dimensions (fixed for deterministic output)
GRID_WIDTH = 40
GRID_HEIGHT = 20


def render_strategy_map(
    embedded: List[Dict[str, Any]],
    *,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
) -> str:
    """Render embedded strategies as an ASCII scatter plot.

    Maps (x, y) coordinates to a fixed-size character grid.
    The best strategy (highest x + y) is marked with ``*``.
    Other strategies are marked with ``o``.

    Parameters
    ----------
    embedded : list of dict
        Each entry has ``"name"``, ``"x"``, ``"y"`` keys.
    width : int
        Grid width in characters (default 40).
    height : int
        Grid height in characters (default 20).

    Returns
    -------
    str
        Multi-line ASCII map string.
    """
    if not embedded:
        return "(no strategies to display)"

    # Determine bounds
    xs = [e["x"] for e in embedded]
    ys = [e["y"] for e in embedded]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # Avoid division by zero for single-point or identical coordinates
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    # Find best strategy (highest x + y, tie-break by name)
    sorted_by_quality = sorted(
        embedded,
        key=lambda e: (-(e["x"] + e["y"]), e["name"]),
    )
    best_name = sorted_by_quality[0]["name"]

    # Initialize grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place points (process in sorted name order for deterministic overlap)
    sorted_embedded = sorted(embedded, key=lambda e: e["name"])
    for e in sorted_embedded:
        col = int((e["x"] - x_min) / x_range * (width - 1))
        row = int((1.0 - (e["y"] - y_min) / y_range) * (height - 1))

        # Clamp to grid bounds
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))

        marker = "*" if e["name"] == best_name else "o"
        grid[row][col] = marker

    # Build output
    lines = []
    lines.append("=== Strategy Map ===")
    lines.append(f"  Strategies: {len(embedded)}  |  * = best  o = other")
    lines.append("")

    # Y-axis label
    lines.append(f"  y={y_max:.2f} |{''.join(grid[0])}|")
    for row_idx in range(1, height - 1):
        lines.append(f"           |{''.join(grid[row_idx])}|")
    lines.append(f"  y={y_min:.2f} |{''.join(grid[height - 1])}|")

    # X-axis
    lines.append(f"            {'-' * width}")
    x_label = f"x={x_min:.2f}"
    x_label_r = f"x={x_max:.2f}"
    pad = width - len(x_label) - len(x_label_r)
    if pad < 1:
        pad = 1
    lines.append(f"           {x_label}{' ' * pad}{x_label_r}")

    # Legend: best strategy name
    lines.append("")
    lines.append(f"  Best: {best_name}")

    return "\n".join(lines)


__all__ = [
    "render_strategy_map",
]
