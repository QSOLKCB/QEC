from typing import Any, Dict, Tuple

import numpy as np

from .mutation_context import MutationContext


class MutationOperator:

    name = "base_mutation"

    def score(
        self,
        graph: np.ndarray,
        eigenvector: np.ndarray,
        context: MutationContext,
    ) -> float:
        """
        Deterministic mutation priority score.
        Higher score = higher priority.
        """
        return 0.0

    def mutate(
        self,
        graph: np.ndarray,
        eigenvector: np.ndarray,
        context: MutationContext,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError
