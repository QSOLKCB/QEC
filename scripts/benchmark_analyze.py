"""Aggregate and analyze benchmark runs from a results directory.

Loads all run JSON files, excludes warm-up runs, and computes
summary statistics (mean, min, max, standard deviation).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main(path_str: str) -> None:
    path = Path(path_str)
    runs: list[float] = []
    for f in sorted(path.glob("run_*.json")):
        data = json.loads(f.read_text())
        if not data.get("is_warmup", False):
            runs.append(data["runtime_seconds"])

    if not runs:
        print("No valid runs found")
        return

    print("\nBenchmark Analysis")
    print(f"Runs: {len(runs)}")
    print(f"Mean: {statistics.mean(runs):.4f}s")
    print(f"Min:  {min(runs):.4f}s")
    print(f"Max:  {max(runs):.4f}s")
    if len(runs) > 1:
        print(f"Std:  {statistics.stdev(runs):.4f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/benchmark_analyze.py <benchmark_runs_dir>")
        sys.exit(1)
    main(sys.argv[1])
