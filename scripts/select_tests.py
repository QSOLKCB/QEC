#!/usr/bin/env python3
"""Debug helper for coverage-aware changed-line test selection."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec.dev.coverage_data import CoverageAwareSelector, changed_lines


def main() -> int:
    lines = sorted(changed_lines(repo_root=REPO_ROOT))

    print("Changed lines:")
    if not lines:
        print("  (none)")
    else:
        for path, line in lines:
            print(f"  {path}:{line}")

    selector = CoverageAwareSelector(repo_root=REPO_ROOT)
    tests = selector.select_tests()

    if tests is None:
        print("\nCoverage data unavailable; fallback selection should be used.")
        return 0

    print("\nTests covering changed lines:")
    if not tests:
        print("  (none)")
    else:
        for test in tests:
            print(f"  {test}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
