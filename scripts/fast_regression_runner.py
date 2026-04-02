#!/usr/bin/env python3
"""Fast regression runner — tiered test harness for QEC.

Reads changed files from git, infers impacted test modules,
runs smoke + impacted tests only, and compares pass counts
against the frozen baseline registry.

Usage:
    python scripts/fast_regression_runner.py          # fast mode (smoke + impacted)
    python scripts/fast_regression_runner.py --full    # full suite (release mode)
    python scripts/fast_regression_runner.py --smoke   # smoke tests only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Return the resolved repository root directory."""
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
BASELINE_PATH = REPO_ROOT / "tests" / "test_baseline_registry.json"
SMOKE_TEST = "tests/test_fast_smoke.py"

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Strict regex for pytest summary line, e.g.:
#   "9870 passed, 11 skipped in 79.83s"
#   "3 failed, 120 passed, 2 skipped, 1 error in 5.00s"
_SUMMARY_RE = re.compile(
    r"(?:(\d+)\s+passed)?"
    r"(?:,?\s*(\d+)\s+failed)?"
    r"(?:,?\s*(\d+)\s+skipped)?"
    r"(?:,?\s*(\d+)\s+errors?)?"
)

# Individual count patterns for robust extraction
_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_SKIPPED_RE = re.compile(r"(\d+)\s+skipped")
_ERRORS_RE = re.compile(r"(\d+)\s+errors?")


def load_baseline() -> dict:
    """Load the frozen baseline registry."""
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _validate_test_path(path_str: str) -> str:
    """Validate and normalize a test path to ensure it is inside the repo.

    Raises ValueError if the resolved path escapes the repository root.
    """
    resolved = (REPO_ROOT / path_str).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        raise ValueError(
            f"Test path escapes repository root: {path_str!r}"
        )
    return str(Path(path_str))


def _parse_pytest_summary(stdout: str) -> dict:
    """Parse pytest counts from stdout using strict regex.

    Searches for individual count patterns (e.g. '123 passed') in
    the output. Returns zero for any count not found.  Never uses
    implicit variable state.
    """
    passed = 0
    failed = 0
    skipped = 0
    errors = 0

    m = _PASSED_RE.search(stdout)
    if m:
        passed = int(m.group(1))

    m = _FAILED_RE.search(stdout)
    if m:
        failed = int(m.group(1))

    m = _SKIPPED_RE.search(stdout)
    if m:
        skipped = int(m.group(1))

    m = _ERRORS_RE.search(stdout)
    if m:
        errors = int(m.group(1))

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }


def run_pytest(targets: list[str], *, collect_only: bool = False) -> dict:
    """Run pytest on the given targets and return result counts.

    Returns dict with keys: passed, failed, skipped, errors, returncode.
    """
    # Validate all dynamic test paths before building command
    safe_targets = [_validate_test_path(t) for t in targets]

    # Safe: command is static argv list, shell disabled
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header"]
    if collect_only:
        cmd.append("--co")
    cmd.extend(safe_targets)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        shell=False,  # Explicit: never use shell
    )

    counts = _parse_pytest_summary(result.stdout)
    counts["returncode"] = result.returncode
    counts["stdout"] = result.stdout
    counts["stderr"] = result.stderr
    return counts


def detect_impacted_tests() -> list[str]:
    """Use SpectralTestSelector to find tests impacted by current changes."""
    from qec.dev.test_selection import SpectralTestSelector

    selector = SpectralTestSelector(repo_root=REPO_ROOT)
    changed = selector.detect_changed_files()
    if not changed:
        return []
    return selector.select_tests(changed)


def run_smoke() -> dict:
    """Run the smoke test suite."""
    smoke_path = REPO_ROOT / SMOKE_TEST
    if not smoke_path.exists():
        print(f"ERROR: Smoke test not found: {SMOKE_TEST}")
        return {"passed": 0, "failed": 1, "returncode": 1}
    return run_pytest([SMOKE_TEST])


def run_fast(*, include_smoke: bool = True) -> dict:
    """Run smoke + impacted tests. Return combined summary."""
    results = {}

    # Phase 1: smoke
    if include_smoke:
        print("=" * 60)
        print("PHASE 1: Smoke tests")
        print("=" * 60)
        smoke = run_smoke()
        results["smoke"] = smoke
        print(f"  passed={smoke['passed']}  failed={smoke['failed']}  "
              f"skipped={smoke.get('skipped', 0)}")
        if smoke.get("failed", 0) > 0 or smoke.get("returncode", 1) != 0:
            print("SMOKE FAILED — aborting fast regression run.")
            results["overall"] = "FAIL"
            return results

    # Phase 2: impacted tests
    print()
    print("=" * 60)
    print("PHASE 2: Impacted tests")
    print("=" * 60)
    impacted = detect_impacted_tests()
    if impacted:
        # Remove smoke test if already run
        impacted = [t for t in impacted if t != SMOKE_TEST]
        print(f"  {len(impacted)} impacted test file(s) detected:")
        for t in impacted:
            print(f"    {t}")
        if impacted:
            imp_result = run_pytest(impacted)
            results["impacted"] = imp_result
            print(f"  passed={imp_result['passed']}  failed={imp_result['failed']}  "
                  f"skipped={imp_result.get('skipped', 0)}")
        else:
            results["impacted"] = {"passed": 0, "failed": 0, "skipped": 0}
    else:
        print("  No impacted tests detected — only smoke ran.")
        results["impacted"] = {"passed": 0, "failed": 0, "skipped": 0}

    # Check for failures
    total_failed = sum(
        r.get("failed", 0) for r in results.values() if isinstance(r, dict)
    )
    results["overall"] = "FAIL" if total_failed > 0 else "PASS"
    return results


def run_full() -> dict:
    """Run the complete pytest suite and check against baseline."""
    print("=" * 60)
    print("FULL SUITE — release mode")
    print("=" * 60)
    result = run_pytest(["tests/"])
    baseline = load_baseline()

    print()
    print("-" * 60)
    print("BASELINE DELTA CHECK")
    print("-" * 60)
    print(f"  Baseline passed : {baseline['baseline_total_passed']}")
    print(f"  Current passed  : {result['passed']}")
    print(f"  Baseline skipped: {baseline['baseline_skipped']}")
    print(f"  Current skipped : {result.get('skipped', 0)}")
    print(f"  Baseline failed : {baseline['baseline_failed']}")
    print(f"  Current failed  : {result['failed']}")

    delta = result["passed"] - baseline["baseline_total_passed"]
    if delta < 0:
        print(f"\n  REGRESSION: {abs(delta)} fewer tests passing than baseline!")
        result["baseline_delta"] = delta
        result["overall"] = "REGRESSION"
    elif result["failed"] > baseline["baseline_failed"]:
        print(f"\n  REGRESSION: more failures than baseline!")
        result["overall"] = "REGRESSION"
    else:
        if delta > 0:
            print(f"\n  +{delta} new passing tests above baseline.")
        else:
            print(f"\n  Baseline matched exactly.")
        result["overall"] = "PASS"
        result["baseline_delta"] = delta

    return result


def print_summary(results: dict) -> None:
    """Print final summary."""
    print()
    print("=" * 60)
    overall = results.get("overall", "UNKNOWN")
    print(f"RESULT: {overall}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QEC fast regression runner"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the complete test suite (release mode)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run smoke tests only",
    )
    args = parser.parse_args()

    if args.full:
        results = run_full()
    elif args.smoke:
        results = {"smoke": run_smoke()}
        smoke = results["smoke"]
        results["overall"] = (
            "FAIL" if smoke.get("failed", 0) > 0 or smoke.get("returncode", 1) != 0
            else "PASS"
        )
    else:
        results = run_fast()

    print_summary(results)

    if results.get("overall") in ("FAIL", "REGRESSION"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
