from __future__ import annotations

import numpy as np

from src.qec.analysis.trap_memory import TrapSubspaceMemory, subspace_similarity


def test_trap_subspace_memory_rotation() -> None:
    memory = TrapSubspaceMemory(max_entries=4, max_subspace_dim=3)

    v = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    v_rot = np.asarray([0.999999, 0.001, 0.0], dtype=np.float64)

    memory.add((v,))
    sim = memory.compute_similarity((v_rot,))
    assert sim > 0.99


def test_trap_subspace_fifo_eviction_unchanged() -> None:
    memory = TrapSubspaceMemory(max_entries=2, max_subspace_dim=3)

    v1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    v2 = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    v3 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    s1 = memory.add((v1,))
    s2 = memory.add((v2,))
    s3 = memory.add((v3,))

    assert len(memory) == 2
    assert subspace_similarity(memory.entries[0], s2) == 1.0
    assert subspace_similarity(memory.entries[1], s3) == 1.0
    assert subspace_similarity(memory.entries[0], s1) == 0.0


def test_trap_subspace_similarity_deterministic() -> None:
    memory = TrapSubspaceMemory(max_entries=4, max_subspace_dim=3)
    basis = (
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )
    probe = (
        np.asarray([0.7071067811865475, 0.7071067811865475, 0.0], dtype=np.float64),
    )

    memory.add(basis)
    s1 = memory.compute_similarity(probe)
    s2 = memory.compute_similarity(probe)
    assert s1 == s2
