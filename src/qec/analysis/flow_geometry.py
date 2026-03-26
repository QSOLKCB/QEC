"""v102.7.0 — Flow geometry and deterministic embedding.

Provides a deterministic geometric embedding of behavioral types using
the transition graph.  Nodes (taxonomy types) are mapped to 2D coordinates
via power iteration on the adjacency/transition matrix, producing a spatial
representation where strongly connected nodes cluster together and transient
states spread outward.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs (coordinates normalized to [-1, 1])

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12
POWER_ITERATIONS = 20
ASCII_GRID_SIZE = 21


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _vec_norm(v: List[float]) -> float:
    """Euclidean norm of a vector."""
    s = 0.0
    for x in v:
        s += x * x
    return s ** 0.5


def _normalize_vec(v: List[float]) -> List[float]:
    """Normalize a vector to unit length.  Returns zero vector if norm is 0."""
    n = _vec_norm(v)
    if n < 1e-15:
        return [0.0] * len(v)
    return [x / n for x in v]


def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix A by vector v."""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result[i] = s
    return result


def _transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose a square matrix."""
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_adjacency_matrix(
    graph: Dict[Tuple[str, str], int],
) -> Tuple[List[str], List[List[float]]]:
    """Build a deterministic adjacency matrix from the transition graph.

    Parameters
    ----------
    graph : dict
        Maps ``(source_type, target_type)`` tuples to transition counts.
        Output of ``build_transition_graph``.

    Returns
    -------
    (nodes, matrix)
        ``nodes`` is a sorted list of all node type names.
        ``matrix`` is an NxN list-of-lists where ``matrix[i][j]`` is the
        transition count from ``nodes[i]`` to ``nodes[j]``, as a float.
    """
    node_set: set = set()
    for src, tgt in graph.keys():
        node_set.add(src)
        node_set.add(tgt)

    nodes = sorted(node_set)
    n = len(nodes)
    index = {name: i for i, name in enumerate(nodes)}

    matrix = [[0.0] * n for _ in range(n)]
    for (src, tgt), count in graph.items():
        i = index[src]
        j = index[tgt]
        matrix[i][j] = float(count)

    return nodes, matrix


def normalize_matrix(A: List[List[float]]) -> List[List[float]]:
    """Normalize rows of a matrix to transition probabilities.

    Each row is divided by its sum (plus a small epsilon to avoid
    division by zero).

    Parameters
    ----------
    A : list of list of float
        Adjacency matrix (NxN).

    Returns
    -------
    list of list of float
        Row-normalized matrix where each row sums to approximately 1
        (or 0 for zero rows).
    """
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_sum = sum(A[i]) + 1e-12
        for j in range(n):
            result[i][j] = _round(A[i][j] / row_sum)
    return result


def embed_types(
    A: List[List[float]],
) -> List[Tuple[float, float]]:
    """Compute a deterministic 2D embedding via power iteration.

    Method:
    1. Initialize vector v = [1, 1, ...] (deterministic)
    2. Iterate: v = A @ v, then normalize(v), for POWER_ITERATIONS steps
    3. Compute second vector: w = A^T @ v, then normalize(w)
    4. Coordinates: (v[i], w[i]) for each node

    Parameters
    ----------
    A : list of list of float
        Normalized transition matrix (NxN).

    Returns
    -------
    list of (float, float)
        Coordinate pairs for each node (ordered by matrix index).
        Returns empty list for empty matrix.
    """
    n = len(A)
    if n == 0:
        return []

    # Step 1: initialize
    v = [1.0] * n

    # Step 2: power iteration on A
    for _ in range(POWER_ITERATIONS):
        v = _mat_vec(A, v)
        v = _normalize_vec(v)

    # Step 3: second vector from A^T
    At = _transpose(A)
    w = _mat_vec(At, v)
    w = _normalize_vec(w)

    # Step 4: coordinates
    return [(_round(v[i]), _round(w[i])) for i in range(n)]


def normalize_coordinates(
    coords: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Center and scale coordinates to the [-1, 1] range.

    Parameters
    ----------
    coords : dict
        Maps type names to (x, y) coordinate pairs.

    Returns
    -------
    dict
        Centered and scaled coordinates.  If all coordinates are
        identical, returns all zeros.
    """
    if not coords:
        return {}

    names = sorted(coords.keys())
    xs = [coords[n][0] for n in names]
    ys = [coords[n][1] for n in names]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]

    max_abs = 0.0
    for x in centered_x:
        if abs(x) > max_abs:
            max_abs = abs(x)
    for y in centered_y:
        if abs(y) > max_abs:
            max_abs = abs(y)

    if max_abs < 1e-15:
        return {name: (0.0, 0.0) for name in names}

    result: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(names):
        result[name] = (
            _round(centered_x[idx] / max_abs),
            _round(centered_y[idx] / max_abs),
        )

    return result


def compute_geometric_metrics(
    coords: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, Any]]:
    """Compute geometric metrics for each embedded type.

    For each type, computes:
    - ``distance_from_center``: Euclidean distance from origin (0, 0)
    - ``density``: average distance to all other types
    - ``cluster_score``: inverse of density (1 / (1 + density))

    Parameters
    ----------
    coords : dict
        Maps type names to (x, y) coordinate pairs (normalized).

    Returns
    -------
    dict
        Keyed by type name.  Each value contains:

        - ``distance_from_center`` : float
        - ``density`` : float
        - ``cluster_score`` : float in [0, 1]
    """
    if not coords:
        return {}

    names = sorted(coords.keys())
    result: Dict[str, Dict[str, Any]] = {}

    for name in names:
        x, y = coords[name]
        dist_center = _round((x * x + y * y) ** 0.5)

        # Average distance to all other types.
        total_dist = 0.0
        count = 0
        for other in names:
            if other == name:
                continue
            ox, oy = coords[other]
            dx = x - ox
            dy = y - oy
            total_dist += (dx * dx + dy * dy) ** 0.5
            count += 1

        density = _round(total_dist / count) if count > 0 else 0.0
        cluster_score = _round(1.0 / (1.0 + density))

        result[name] = {
            "distance_from_center": dist_center,
            "density": density,
            "cluster_score": cluster_score,
        }

    return result


def compute_flow_geometry(
    graph: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """Run the full flow geometry pipeline.

    Pipeline:
    1. Build adjacency matrix from transition graph
    2. Normalize to transition probabilities
    3. Embed via power iteration
    4. Normalize coordinates to [-1, 1]
    5. Compute geometric metrics

    Parameters
    ----------
    graph : dict
        Maps ``(source_type, target_type)`` tuples to transition counts.

    Returns
    -------
    dict
        Contains:

        - ``coordinates`` : dict mapping type names to (x, y) tuples
        - ``metrics`` : dict mapping type names to geometric metric dicts
        - ``nodes`` : sorted list of type names
    """
    if not graph:
        return {
            "coordinates": {},
            "metrics": {},
            "nodes": [],
        }

    nodes, adjacency = build_adjacency_matrix(graph)
    normalized = normalize_matrix(adjacency)
    raw_coords = embed_types(normalized)

    coords_dict = {nodes[i]: raw_coords[i] for i in range(len(nodes))}
    coords_dict = normalize_coordinates(coords_dict)

    metrics = compute_geometric_metrics(coords_dict)

    return {
        "coordinates": coords_dict,
        "metrics": metrics,
        "nodes": nodes,
    }


def render_ascii_map(
    coords: Dict[str, Tuple[float, float]],
    grid_size: int = ASCII_GRID_SIZE,
) -> str:
    """Render an ASCII map of embedded type coordinates.

    Places the first character of each type name on a grid.
    Deterministic placement based on coordinate values.

    Parameters
    ----------
    coords : dict
        Maps type names to (x, y) coordinate pairs (normalized to [-1, 1]).
    grid_size : int
        Grid dimensions (default 21x21).

    Returns
    -------
    str
        Multi-line ASCII art string representing the embedding.
    """
    if not coords:
        return "=== Flow Geometry Map ===\n(empty)"

    # Initialize grid with dots.
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    half = (grid_size - 1) / 2.0

    # Place types on grid (sorted for determinism).
    placed: Dict[str, Tuple[int, int]] = {}
    for name in sorted(coords.keys()):
        x, y = coords[name]
        col = int(round(x * half + half))
        row = int(round(-y * half + half))  # flip y for display

        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))

        label = name[0].upper()
        grid[row][col] = label
        placed[name] = (row, col)

    lines = ["=== Flow Geometry Map ==="]
    for row in grid:
        lines.append(" ".join(row))

    # Legend.
    lines.append("")
    lines.append("Legend:")
    for name in sorted(coords.keys()):
        x, y = coords[name]
        lines.append(f"  {name[0].upper()} = {name} ({x:.2f}, {y:.2f})")

    return "\n".join(lines)


__all__ = [
    "ASCII_GRID_SIZE",
    "POWER_ITERATIONS",
    "ROUND_PRECISION",
    "build_adjacency_matrix",
    "compute_flow_geometry",
    "compute_geometric_metrics",
    "embed_types",
    "normalize_coordinates",
    "normalize_matrix",
    "render_ascii_map",
]
