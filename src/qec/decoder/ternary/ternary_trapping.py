"""
Ternary trapping diagnostics — detect persistent undecided states.

Identifies connected regions of ternary messages with value 0,
computes frustration metrics, detects persistent zero states across
iterations, and estimates trapping indicators from graph structure.

All outputs are deterministic.  No randomness.  numpy.float64 numerics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_zero_regions(messages: np.ndarray) -> dict[str, Any]:
    """Identify connected regions of ternary messages with value 0.

    A connected region is a maximal run of consecutive zero-valued
    entries in the flattened message vector.

    Parameters
    ----------
    messages : np.ndarray
        Ternary message vector with values in {-1, 0, +1}.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - region_ids: list[int], sorted region identifiers (0-indexed)
        - region_sizes: list[int], size of each region (same order)
        - node_indices: list[list[int]], sorted node indices per region
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()

    region_ids: list[int] = []
    region_sizes: list[int] = []
    node_indices: list[list[int]] = []

    current_region: list[int] = []
    for i in range(arr.size):
        if arr[i] == 0:
            current_region.append(int(i))
        else:
            if current_region:
                rid = len(region_ids)
                region_ids.append(rid)
                region_sizes.append(len(current_region))
                node_indices.append(list(current_region))
                current_region = []
    # Flush last region
    if current_region:
        rid = len(region_ids)
        region_ids.append(rid)
        region_sizes.append(len(current_region))
        node_indices.append(list(current_region))

    return {
        "region_ids": region_ids,
        "region_sizes": region_sizes,
        "node_indices": node_indices,
    }


def compute_frustration_index(messages: np.ndarray) -> np.float64:
    """Compute the frustration index of ternary messages.

    The frustration index is the fraction of messages that remain
    in the undecided state (value 0).

    Parameters
    ----------
    messages : np.ndarray
        Ternary message vector with values in {-1, 0, +1}.

    Returns
    -------
    np.float64
        Frustration index in [0.0, 1.0].
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    if arr.size == 0:
        return np.float64(0.0)
    zero_count = int(np.sum(arr == 0))
    return np.float64(zero_count / arr.size)


def detect_persistent_zero_states(history: list[np.ndarray]) -> list[int]:
    """Detect nodes that remain undecided across multiple iterations.

    A node is persistent-zero if it has value 0 in every iteration
    of the provided history.

    Parameters
    ----------
    history : list[np.ndarray]
        List of ternary message vectors, one per iteration.
        Each array has values in {-1, 0, +1}.

    Returns
    -------
    list[int]
        Sorted list of node indices that are zero in all iterations.
    """
    if not history:
        return []

    first = np.asarray(history[0], dtype=np.int8).ravel()
    persistent = (first == 0)

    for snapshot in history[1:]:
        arr = np.asarray(snapshot, dtype=np.int8).ravel()
        if arr.size != first.size:
            raise ValueError(
                "History snapshots must all have identical message lengths"
            )
        persistent = persistent & (arr == 0)

    indices = sorted(int(i) for i in np.where(persistent)[0])
    return indices


def estimate_trapping_indicator(
    messages: np.ndarray,
    parity_matrix: np.ndarray,
) -> np.float64:
    """Estimate likelihood of trapping behavior.

    Combines three signals:
    1. Zero region density: fraction of nodes in zero regions
    2. Conflict density: fraction of edges with sign disagreement
    3. Unsatisfied check fraction: fraction of checks with nonzero syndrome

    The indicator is the arithmetic mean of these three signals,
    returned as a deterministic float64.

    Parameters
    ----------
    messages : np.ndarray
        Ternary message vector with values in {-1, 0, +1}.
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).

    Returns
    -------
    np.float64
        Trapping indicator score in [0.0, 1.0].
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    H = np.asarray(parity_matrix, dtype=np.float64)
    m, n = H.shape

    if arr.size == 0 or n != arr.size:
        raise ValueError(
            "messages length must match parity_matrix column count"
        )

    # Signal 1: zero region density
    zero_density = np.float64(int(np.sum(arr == 0)) / arr.size)

    # Signal 2: conflict density (edges with sign disagreement)
    # For each check, count variable pairs where both are nonzero and disagree.
    nonzero_mask = (arr != 0)
    arr_f = arr.astype(np.float64)
    # Per-check: number of nonzero variables connected
    H_bin = (H != 0).astype(np.float64)
    nz_per_check = H_bin @ nonzero_mask.astype(np.float64)  # (m,)
    # Per-check: total variable pairs = deg*(deg-1)/2
    deg_per_check = np.sum(H_bin, axis=1)  # (m,)
    total_pairs = (deg_per_check * (deg_per_check - 1.0)) / 2.0
    edge_count = int(np.sum(total_pairs))

    if edge_count == 0:
        conflict_density = np.float64(0.0)
    else:
        # Sum of message values over nonzero-connected variables per check
        nz_vals = arr_f * nonzero_mask.astype(np.float64)
        s = H_bin @ nz_vals  # (m,) sum of nonzero messages per check
        # Count of +1 and -1 per check
        pos_count = H_bin @ (arr == 1).astype(np.float64)   # (m,)
        neg_count = H_bin @ (arr == -1).astype(np.float64)  # (m,)
        # Conflicting pairs per check = pos_count * neg_count
        conflict_pairs = pos_count * neg_count
        conflict_count = int(np.sum(conflict_pairs))
        conflict_density = np.float64(conflict_count / edge_count)

    # Signal 3: unsatisfied check fraction via matrix-vector product
    check_sums = H_bin @ arr_f  # (m,)
    unsatisfied = int(np.sum(check_sums != 0.0))
    if m == 0:
        check_fraction = np.float64(0.0)
    else:
        check_fraction = np.float64(unsatisfied / m)

    indicator = np.float64((zero_density + conflict_density + check_fraction) / 3.0)
    return indicator
