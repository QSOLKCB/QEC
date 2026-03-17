"""Analyze benchmark JSON artifacts from benchmark_runner.py (v68.3).

Reads one or more benchmark result files and prints a summary including:
- runtime statistics
- termination signal totals
- efficiency metric averages

Deterministic, no external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python benchmark_analyze.py <result1.json> [result2.json ...]")
        sys.exit(1)

    runtimes: list[float] = []
    termination_totals: dict[str, int] = {
        "convergence": 0,
        "markov": 0,
        "curvature": 0,
    }
    efficiency_accumulator: dict[str, list[float]] = {
        "early_exit_ratio": [],
        "markov_ratio": [],
        "curvature_ratio": [],
    }
    slow_test_totals: dict[str, list[float]] = {}

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        data = json.loads(path.read_text())

        runtimes.append(data.get("runtime_seconds", 0.0))

        stats = data.get("termination_stats", {})
        for k in termination_totals:
            termination_totals[k] += stats.get(k, 0)

        eff = data.get("efficiency_metrics", {})
        for k in efficiency_accumulator:
            if k in eff:
                efficiency_accumulator[k].append(eff[k])

        for entry in data.get("slow_tests", []):
            name = entry.get("test", "unknown")
            dur = entry.get("duration", 0.0)
            slow_test_totals.setdefault(name, []).append(dur)

    if not runtimes:
        print("No valid benchmark files found.")
        sys.exit(1)

    print(f"Analyzed {len(runtimes)} benchmark run(s)\n")

    print("Runtime:")
    print(f"  mean={statistics.mean(runtimes):.4f}s")
    if len(runtimes) > 1:
        print(f"  stdev={statistics.stdev(runtimes):.4f}s")
    print(f"  min={min(runtimes):.4f}s  max={max(runtimes):.4f}s")

    print("\nTermination Signals:")
    for k, v in termination_totals.items():
        print(f"  {k}: {v}")

    print("\nEfficiency Metrics:")
    for k, values in efficiency_accumulator.items():
        if values:
            print(f"  {k}: mean={statistics.mean(values):.4f}")
        else:
            print(f"  {k}: no data")

    if slow_test_totals:
        print("\nSlowest Tests (across runs):")
        sorted_tests = sorted(
            slow_test_totals.items(),
            key=lambda x: statistics.mean(x[1]),
            reverse=True,
        )
        for name, durations in sorted_tests[:10]:
            print(f"  {name}: mean={statistics.mean(durations):.4f}s")


if __name__ == "__main__":
    main()
