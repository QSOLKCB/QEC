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
        "--spectral-select",
        action="store_true",
        default=False,
        help="Select only tests affected by modified source modules.",
    )


def pytest_collection_modifyitems(config, items):
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
