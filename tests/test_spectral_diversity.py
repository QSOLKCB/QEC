from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_diversity_memory import (
    SpectralDiversityConfig,
    SpectralDiversityMemory,
    spectral_distance,
)
from src.qec.analysis.spectral_signature import SpectralSignature, compute_signature
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def _h_base() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=np.float64)


def _h_similar() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
    ], dtype=np.float64)


def _h_different() -> np.ndarray:
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.float64)


def test_signature_consistency() -> None:
    H = _h_base()
    s1 = compute_signature(H)
    s2 = compute_signature(H)

    np.testing.assert_array_equal(s1.nb_spectrum, s2.nb_spectrum)
    assert s1.bh_negative_modes == s2.bh_negative_modes
    assert s1.bh_energy == s2.bh_energy
    assert s1.max_ipr == s2.max_ipr


def test_distance_metric_identical_signatures_zero() -> None:
    sig = compute_signature(_h_base())
    d = spectral_distance(sig, sig)
    assert d == 0.0


def test_memory_fifo_eviction() -> None:
    mem = SpectralDiversityMemory(max_entries=2)
    s1 = compute_signature(_h_base())
    s2 = compute_signature(_h_similar())
    s3 = compute_signature(_h_different())

    mem.add(s1)
    mem.add(s2)
    mem.add(s3)

    assert len(mem) == 2
    assert mem.entries[0] == s2
    assert mem.entries[1] == s3


def test_novelty_reward_larger_for_different_graph() -> None:
    base = compute_signature(_h_base())
    similar = compute_signature(_h_similar())
    different = compute_signature(_h_different())

    mem = SpectralDiversityMemory(max_entries=8)
    mem.add(base)

    novelty_similar = mem.novelty_score(similar)
    novelty_different = mem.novelty_score(different)
    assert novelty_different > novelty_similar


def test_novelty_determinism_repeated_runs() -> None:
    mem = SpectralDiversityMemory(max_entries=8)
    mem.add(compute_signature(_h_base()))
    target = compute_signature(_h_different())

    n1 = mem.novelty_score(target)
    n2 = mem.novelty_score(target)
    assert n1 == n2


def test_mutator_spectral_diversity_is_opt_in_and_deterministic() -> None:
    H = _h_base()
    mut_default = NBGradientMutator(enabled=True, avoid_4cycles=False)
    mut_optin_a = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        spectral_diversity=SpectralDiversityConfig(enabled=True, max_memory=16, eta_novelty=0.15),
    )
    mut_optin_b = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        spectral_diversity=SpectralDiversityConfig(enabled=True, max_memory=16, eta_novelty=0.15),
    )

    _, log_default = mut_default.mutate(H, steps=2)
    _, log_a = mut_optin_a.mutate(H, steps=2)
    _, log_b = mut_optin_b.mutate(H, steps=2)

    assert all(step.get("novelty_score", 0.0) == 0.0 for step in log_default)
    assert log_a == log_b
    assert all(isinstance(step.get("spectral_signature"), SpectralSignature) for step in log_a)
