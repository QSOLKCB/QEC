"""
Minimal deterministic pytest configuration.

This file intentionally avoids complex hooks that previously caused
collection instability.
"""

from __future__ import annotations

import builtins
import importlib
import os
import random
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest

from qec.dev.test_selection import select_tests_for_changed_files
from qec.testing.experiment_harness import DeterministicExperimentHarness

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="PIL",
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="cffi",
)


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


# ---------------------------------------------------------------------------
# Import hygiene helper for cffi detection
# ---------------------------------------------------------------------------

_CFFI_NAMES = frozenset({"cffi", "_cffi_backend"})


@contextmanager
def track_cffi_imports() -> Iterator[list[tuple[str, str]]]:
    """Context manager that records direct cffi imports from QEC modules.

    Instruments ``builtins.__import__`` to capture when a module starting with
    ``cffi`` or ``_cffi_backend`` is imported directly by a ``qec.*`` module.
    Yields a list of ``(importer, imported)`` tuples that were detected.

    Usage::

        with track_cffi_imports() as violations:
            importlib.reload(some_qec_module)
        assert violations == [], f"QEC imported cffi directly: {violations}"
    """
    original_import = builtins.__import__
    violations: list[tuple[str, str]] = []

    def _spy_import(
        name: str,
        globals_: dict | None = None,
        locals_: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        # Determine the importer module name from globals
        importer = ""
        if globals_ is not None:
            importer = globals_.get("__name__", "")

        # Check if a qec.* module is directly importing cffi
        if importer.startswith("qec"):
            top_level = name.split(".")[0]
            if top_level in _CFFI_NAMES:
                violations.append((importer, name))

        return original_import(name, globals_, locals_, fromlist, level)

    builtins.__import__ = _spy_import
    try:
        yield violations
    finally:
        builtins.__import__ = original_import


def assert_no_cffi_imports(module_name: str) -> None:
    """Helper to assert that reloading a module does not directly import cffi.

    Removes the module (and submodules) from sys.modules, then reimports while
    tracking cffi imports. Raises AssertionError if any QEC module directly
    imports cffi.

    Parameters
    ----------
    module_name : str
        Fully qualified module name (e.g., ``"qec.analysis.attractor_analysis"``).
    """
    # Remove the module and its submodules to ensure fresh import
    to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    for k in to_remove:
        del sys.modules[k]

    with track_cffi_imports() as violations:
        importlib.import_module(module_name)

    assert violations == [], (
        f"QEC code directly imported cffi:\n"
        + "\n".join(f"  {imp} -> {name}" for imp, name in violations)
    )


@pytest.fixture
def cffi_import_tracker():
    """Fixture providing the cffi import tracking context manager."""
    return track_cffi_imports
