from __future__ import annotations

import numpy as np

from src.qec.analysis.trap_memory import TrapMemory, TrapMemoryConfig
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def test_trap_registration_canonical_sign_and_norm() -> None:
    mem = TrapMemory(max_traps=4)
    v = np.array([-3.0, 0.0, 4.0], dtype=np.float64)
    mem.add(v)

    assert len(mem.trap_vectors) == 1
    stored = mem.trap_vectors[0]
    assert stored.dtype == np.float64
    assert np.isclose(np.linalg.norm(stored), 1.0)
    idx = int(np.argmax(np.abs(stored)))
    assert stored[idx] > 0.0


def test_similarity_identical_vectors_is_one() -> None:
    mem = TrapMemory(max_traps=4)
    v = np.array([0.0, -2.0, 0.0, 0.0], dtype=np.float64)
    mem.add(v)

    sim_same = mem.similarity(np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float64))
    sim_orth = mem.similarity(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

    assert np.isclose(sim_same, 1.0)
    assert np.isclose(sim_orth, 0.0)


def test_memory_fifo_eviction() -> None:
    mem = TrapMemory(max_traps=2)
    mem.add(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    mem.add(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    mem.add(np.array([0.0, 0.0, 1.0], dtype=np.float64))

    assert len(mem.trap_vectors) == 2
    assert np.isclose(mem.similarity(np.array([1.0, 0.0, 0.0], dtype=np.float64)), 0.0)
    assert np.isclose(mem.similarity(np.array([0.0, 1.0, 0.0], dtype=np.float64)), 1.0)
    assert np.isclose(mem.similarity(np.array([0.0, 0.0, 1.0], dtype=np.float64)), 1.0)


def test_penalty_application_in_swap_scoring() -> None:
    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        frustration_guided=True,
        eta_frustration=0.0,
        frustration_eval_limit=1,
        trap_memory_config=TrapMemoryConfig(enabled=True, max_traps=8, eta_trap=0.5),
    )
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    vec = np.array([1.0, 0.0], dtype=np.float64)

    class _R:
        def __init__(self, score: float, modes: tuple[np.ndarray, ...]):
            self.frustration_score = score
            self.negative_modes = 1
            self.max_ipr = 0.3
            self.trap_modes = modes

    calls = {"i": 0}

    def _compute(_H: np.ndarray):
        calls["i"] += 1
        if calls["i"] == 1:
            return _R(1.0, (vec,))
        return _R(1.0, (vec,))

    mut._frustration.compute_frustration = _compute  # type: ignore[assignment]
    candidates = [{"score": -1.0, "swap_index": 0, "remove": ((0, 0),), "add": ((0, 1),)}]
    mut._apply_frustration_guidance(H, candidates)

    assert np.isclose(candidates[0]["trap_similarity"], 1.0)
    assert np.isclose(candidates[0]["trap_penalty"], 0.5)
    assert np.isclose(candidates[0]["score"], -1.5)


def test_trap_memory_similarity_determinism() -> None:
    mem = TrapMemory(max_traps=4)
    base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mem.add(base)
    q = np.array([3.0, 2.0, 1.0], dtype=np.float64)

    s1 = mem.similarity(q)
    s2 = mem.similarity(q)

    assert s1 == s2


def test_trap_subspace_similarity() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.array([0.999999999999, 0.0, 0.0], dtype=np.float64)

    mem = TrapMemory(max_traps=4)
    mem.add(v)
    sim = mem.similarity(u)

    assert sim > 0.999999999998
