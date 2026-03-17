"""Deterministic benchmark harness for the QEC test suite.

Runs pytest twice (warm + measured) and writes timing results to a
JSON artifact.  All measurements use ``time.perf_counter`` for
monotonic high-resolution timing.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def run_pytest():
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
    # warm run
    run_pytest()
    # measured run
    result = run_pytest()
    out_path = Path("benchmark_v68_2_2.json")
    out_path.write_text(json.dumps(result, indent=2))
    print("Benchmark complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
