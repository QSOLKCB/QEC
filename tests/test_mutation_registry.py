from __future__ import annotations

import numpy as np

from src.qec.discovery.mutation_context import MutationContext
from src.qec.discovery.mutation_operator import MutationOperator
from src.qec.discovery.mutation_registry import MutationRegistry


class _StubOperator(MutationOperator):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = float(value)

    def score(self, graph: np.ndarray, eigenvector: np.ndarray, context: MutationContext) -> float:
        return self._value + float(context.nb_spectral_radius) * 0.0

    def mutate(self, graph: np.ndarray, eigenvector: np.ndarray, context: MutationContext):
        return np.asarray(graph, dtype=np.float64).copy(), {"name": self.name}


def test_mutation_registry_registration_order() -> None:
    registry = MutationRegistry()
    first = _StubOperator("first", 0.0)
    second = _StubOperator("second", 1.0)
    registry.register(first)
    registry.register(second)

    operators = registry.operators()
    assert operators == [first, second]


def test_mutation_registry_deterministic_scored_order() -> None:
    registry = MutationRegistry()
    registry.register(_StubOperator("zeta", 1.0))
    registry.register(_StubOperator("alpha", 1.0))
    registry.register(_StubOperator("beta", 2.0))

    ordered = registry.scored(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        MutationContext(nb_spectral_radius=1.25),
    )

    assert [op.name for op in ordered] == ["beta", "alpha", "zeta"]


def test_mutation_registry_mutation_context_compatibility() -> None:
    registry = MutationRegistry()
    op = _StubOperator("ctx", 3.0)
    registry.register(op)
    context = MutationContext(nb_spectral_radius=2.0)

    ordered = registry.scored(
        np.ones((1, 1), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        context,
    )

    mutated, meta = ordered[0].mutate(
        np.ones((1, 1), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        context,
    )
    assert ordered[0].name == "ctx"
    assert mutated.dtype == np.float64
    assert meta["name"] == "ctx"
