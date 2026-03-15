from __future__ import annotations

from pathlib import Path
import sys

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--spectral-select",
        action="store_true",
        default=False,
        help="Select only tests affected by changed qec modules and their dependents.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--spectral-select"):
        return

    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        from src.qec.dev.test_selection import detect_changed_files, select_tests_for_changed_files

        changed_files = detect_changed_files(repo_root)
        result = select_tests_for_changed_files(changed_files, repo_root)
    except Exception:
        # Fallback to full test suite for any selector/tooling failure.
        return

    if not result.selected_tests:
        # Fallback to full suite if selection resolves to empty.
        return

    selected = {str((repo_root / rel).resolve()) for rel in result.selected_tests}
    kept = [item for item in items if str(Path(str(item.fspath)).resolve()) in selected]

    if not kept:
        # Fallback to full suite if no collected item matches selected paths.
        return

    kept.sort(key=lambda item: str(item.fspath))
    items[:] = kept
