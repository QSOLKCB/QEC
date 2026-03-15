"""
Minimal deterministic pytest configuration.

This file intentionally avoids complex hooks that previously caused
collection instability.
"""

import random
import numpy as np


def pytest_configure(config):
    """
    Ensure deterministic test execution.

    This seeds Python's random module and NumPy so tests relying on
    stochastic components behave reproducibly.
    """
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
