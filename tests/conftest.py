"""
Pytest configuration for QEC test suite.

This file intentionally stays minimal to avoid introducing
non-deterministic fixtures.
"""

import random

import numpy as np


def pytest_configure(config):
    """Ensure deterministic tests."""
    del config
    random.seed(0)
    np.random.seed(0)
