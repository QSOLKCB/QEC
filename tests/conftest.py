"""Pytest plugin for opt-in coverage-aware test selection."""

from __future__ import annotations

from pathlib import Path

from src.qec.dev.coverage_data import CoverageAwareSelector
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qec.dev.test_selection import SpectralTestSelector


def pytest_addoption(parser):
    parser.addoption(
        "--coverage-select",
        action="store_true",
        default=False,
        help="Select tests that executed changed lines using .coverage context data.",
        "--spectral-select",
        action="store_true",
        default=False,
        help="Select only tests affected by modified source modules.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--coverage-select"):
        return

    selector = CoverageAwareSelector(repo_root=Path(__file__).resolve().parents[1])
    selected_tests = selector.select_tests()

    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if selected_tests is None:
        if terminal:
            terminal.write_line("coverage-select: no coverage data, using default collection")
        return

    if not selected_tests:
        if terminal:
            terminal.write_line("coverage-select: no changed lines detected, using default collection")
        return

    selected_set = set(selected_tests)
    kept = [item for item in items if item.nodeid in selected_set]
    if not kept:
        if terminal:
            terminal.write_line("coverage-select: no tests matched changed lines, using default collection")
        return

    deselected = [item for item in items if item.nodeid not in selected_set]
    items[:] = kept
    config.hook.pytest_deselected(items=deselected)
    if terminal:
        terminal.write_line(f"coverage-select: running {len(kept)} selected tests")
    if not config.getoption("--spectral-select"):
        return

    selector = SpectralTestSelector(repo_root=REPO_ROOT)
    selected_tests = selector.select_tests()
    if not selected_tests:
        return

    selected_set = set(selected_tests)
    filtered_items = [
        item
        for item in items
        if Path(item.nodeid.split("::", maxsplit=1)[0]).as_posix() in selected_set
    ]

    if filtered_items:
        items[:] = filtered_items
