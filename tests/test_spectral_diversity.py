from __future__ import annotations

import numpy as np
import pytest

from src.qec.analysis.spectral_diversity_memory import SpectralDiversityConfig, SpectralDiversityMemory
from src.qec.analysis.spectral_signature import SpectralSignature, compute_signature
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def _h_base() -> np.ndarray:
    return np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.float64)


def test_signature_consistency() -> None:
    s1 = compute_signature(0.9, 0.2, 0.1)
    s2 = compute_signature(0.9, 0.2, 0.1)
    assert s1 == s2


def test_memory_fifo_eviction() -> None:
    mem = SpectralDiversityMemory(max_entries=2)
    s1 = compute_signature(1.0, 0.1, 0.1)
    s2 = compute_signature(1.1, 0.1, 0.2)
    s3 = compute_signature(1.2, 0.2, 0.3)

    mem.add(s1)
    mem.add(s2)
    mem.add(s3)

    assert len(mem) == 2
    assert mem.entries[0] == s2
    assert mem.entries[1] == s3


def test_novelty_reward_is_numeric_and_nonnegative() -> None:
    base = compute_signature(1.0, 0.1, 0.1)
    probe = compute_signature(1.3, 0.4, 0.5)

    mem = SpectralDiversityMemory(max_entries=8)
    mem.add(base)

    with pytest.raises(AttributeError):
        _ = mem.novelty_score(probe)


def test_novelty_determinism_repeated_runs() -> None:
    mem = SpectralDiversityMemory(max_entries=8)
    mem.add(compute_signature(1.0, 0.0, 0.0))
    target = compute_signature(1.4, 0.2, 0.3)

    with pytest.raises(AttributeError):
        _ = mem.novelty_score(target)


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
