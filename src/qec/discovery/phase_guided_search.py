"""Deterministic phase-guided spectral search helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_phase_target(
    phase_map: dict[str, Any],
    phase_visit_counts: dict[int, int],
) -> dict[str, int]:
    """Select the highest-priority under-explored phase deterministically."""
    regions = list(phase_map.get("phase_regions", []))
    if not regions:
        return {"target_phase_id": -1}

    phase_ids: list[int] = []
    priorities: list[float] = []
    source_indices: list[int] = []
    for idx, rec in enumerate(regions):
        phase_id = int(rec.get("phase_id", idx))
        visit_count = int(phase_visit_counts.get(phase_id, 0))
        priority = float(np.float64(1.0 / (1.0 + float(np.float64(visit_count)))))
        phase_ids.append(phase_id)
        priorities.append(priority)
        source_indices.append(int(idx))

    order = np.lexsort(
        (
            np.asarray(source_indices, dtype=np.int64),
            np.asarray(phase_ids, dtype=np.int64),
            -np.asarray(priorities, dtype=np.float64),
        )
    )
    return {"target_phase_id": int(phase_ids[int(order[0])])}


def propose_phase_guided_step(
    current_vector: np.ndarray,
    phase_map: dict[str, Any],
    target_phase: dict[str, Any] | int,
) -> np.ndarray:
    """Take a deterministic scaled step from current vector toward a phase centroid."""
    current = np.asarray(current_vector, dtype=np.float64).reshape(-1)
    target_phase_id = (
        int(target_phase.get("target_phase_id", -1))
        if isinstance(target_phase, dict)
        else int(target_phase)
    )

    target_centroid = None
    for rec in phase_map.get("phase_regions", []):
        if int(rec.get("phase_id", -1)) == target_phase_id:
            target_centroid = np.asarray(rec.get("centroid", []), dtype=np.float64).reshape(-1)
            break
    if target_centroid is None or target_centroid.size == 0:
        return current.copy()

    dims = int(min(current.size, target_centroid.size))
    if dims <= 0:
        return current.copy()

    out = current.copy()
    direction = target_centroid[:dims] - current[:dims]
    out[:dims] = current[:dims] + np.float64(0.25) * direction
    return out.astype(np.float64, copy=False)


def update_phase_visit_counts(
    phase_id: int,
    visit_counts: dict[int, int],
) -> dict[int, int]:
    """Deterministically increment and return stably ordered phase visit counts."""
    updated = {int(k): int(v) for k, v in visit_counts.items()}
    pid = int(phase_id)
    updated[pid] = int(updated.get(pid, 0) + 1)

    keys = np.asarray(list(updated.keys()), dtype=np.int64)
    order = np.lexsort((keys,))
    return {int(keys[idx]): int(updated[int(keys[idx])]) for idx in order.tolist()}
