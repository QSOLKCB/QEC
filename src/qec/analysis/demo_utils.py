"""Utilities shared by deterministic analysis demo scripts."""

from __future__ import annotations

from qec.analysis.iterative_system_abstraction_layer import IterativeStateSnapshot


def build_snapshots(
    state_ids: list[str],
    metrics: list[float],
) -> tuple[IterativeStateSnapshot, ...]:
    """Construct deterministic iterative snapshots from aligned ids and metrics."""
    if len(state_ids) != len(metrics):
        raise ValueError("state_ids and metrics must have the same length")

    snapshots: list[IterativeStateSnapshot] = []
    for index, (state_id, metric) in enumerate(zip(state_ids, metrics)):
        snapshots.append(
            IterativeStateSnapshot(
                step_index=index,
                state_id=state_id,
                state_payload={"state": state_id},
                convergence_metric=metric,
                active=True,
            )
        )
    return tuple(snapshots)
