"""
Minimal deterministic pytest configuration.

This file intentionally avoids complex hooks that previously caused
collection instability.
"""

import random
import os
import numpy as np
import pytest

from src.qec.testing.experiment_harness import DeterministicExperimentHarness
from src.qec.dev.test_selection import select_tests_for_changed_files


def pytest_configure(config):
    """
    Ensure deterministic test execution.

    This seeds Python's random module and NumPy so tests relying on
    stochastic components behave reproducibly.
    """
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QEC_FAST_TESTS") != "1":
        return

    selected = select_tests_for_changed_files()

    if not selected:
        return

    selected = {os.path.basename(path) for path in selected}

    items[:] = [item for item in items if item.fspath.basename in selected]


@pytest.fixture
def deterministic_harness():
    return DeterministicExperimentHarness(seed=0)
