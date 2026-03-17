"""Deterministic multi-run benchmark harness for the QEC test suite.

Runs pytest N times (first run is warm-up), saves each run as
structured JSON, and writes a summary file.  All measurements use
``time.perf_counter`` for monotonic high-resolution timing.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path


def run_pytest():
    """Run pytest once and return timing + result metadata."""
    start = time.perf_counter()
    result = subprocess.run(
        ["pytest", "-q"],
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()
    return {
        "returncode": result.returncode,
        "runtime_seconds": round(end - start, 4),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-10:]),
    }


def main():
    runs = 3
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"benchmark_runs_{timestamp}")
    out_dir.mkdir(exist_ok=True)

    results = []
    for i in range(runs):
        print(f"Running benchmark {i + 1}/{runs}...")
        res = run_pytest()
        res["run_index"] = i
        res["is_warmup"] = i == 0
        path = out_dir / f"run_{i}.json"
        path.write_text(json.dumps(res, indent=2))
        results.append(res)

        if res["returncode"] != 0:
            print("Pytest failed -- aborting benchmark")
            sys.exit(res["returncode"])

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nBenchmark runs saved to: {out_dir}")


if __name__ == "__main__":
    main()
