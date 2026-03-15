"""Pytest plugin for opt-in coverage-aware test selection."""

from __future__ import annotations

from pathlib import Path

from src.qec.dev.coverage_data import CoverageAwareSelector


def pytest_addoption(parser):
    parser.addoption(
        "--coverage-select",
        action="store_true",
        default=False,
        help="Select tests that executed changed lines using .coverage context data.",
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
