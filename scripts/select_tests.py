#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec.dev.test_selection import detect_changed_files, select_tests_for_changed_files


def main() -> int:
    repo_root = REPO_ROOT
    changed_files = detect_changed_files(repo_root)
    result = select_tests_for_changed_files(changed_files, repo_root)

    print("Changed modules:")
    if result.changed_modules:
        for module in result.changed_modules:
            print(module)
    else:
        print("(none)")

    print("\nDependent modules:")
    if result.affected_modules:
        for module in result.affected_modules:
            print(module)
    else:
        print("(none)")

    print("\nSelected tests:")
    if result.selected_tests:
        for test_path in result.selected_tests:
            print(test_path)
    else:
        print("(none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
