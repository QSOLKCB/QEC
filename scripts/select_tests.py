from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qec.dev.test_selection import SpectralTestSelector


def main() -> int:
    selector = SpectralTestSelector(repo_root=REPO_ROOT)
    changed_files = selector.detect_changed_files()
    changed_modules = selector.changed_modules(changed_files)
    selected_tests = selector.select_tests(changed_files)

    print("Changed modules:")
    if changed_modules:
        for module in changed_modules:
            print(f"  {module}")
    else:
        print("(none)")

    print("\nSelected tests:")
    if selected_tests:
        for test_path in selected_tests:
            print(f"  {test_path}")
    else:
        print("(none; run full suite)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
