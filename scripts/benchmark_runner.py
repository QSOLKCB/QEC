"""Diagnostic + efficiency-aware benchmark runner (v68.3).

Extends the benchmark harness with:
- slow test detection (pytest --durations)
- termination signal counters
- efficiency metrics (early-exit / markov / curvature ratios)

All measurements use ``time.perf_counter`` for monotonic timing.
Deterministic, float64-safe, no external dependencies.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from src.qec.decoder.ternary.ternary_coevolution import (
    get_termination_stats,
    reset_termination_stats,
)


def extract_slowest_tests(output: str) -> list[dict]:
    """Parse pytest --durations output for slow test entries."""
    lines = output.splitlines()
    slow_section = False
    results: list[dict] = []
    for line in lines:
        if "slowest durations" in line.lower():
            slow_section = True
            continue
        if slow_section:
            if not line.strip():
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    duration = float(parts[0].rstrip("s"))
                    test_name = parts[-1]
                    results.append({
                        "test": test_name,
                        "duration": duration,
                    })
                except ValueError:
                    continue
    return results


def compute_efficiency_metrics(stats: dict) -> dict:
    """Compute termination efficiency ratios from signal counters."""
    total = (
        stats.get("convergence", 0)
        + stats.get("markov", 0)
        + stats.get("curvature", 0)
    )
    if total == 0:
        return {
            "early_exit_ratio": 0.0,
            "markov_ratio": 0.0,
            "curvature_ratio": 0.0,
        }
    return {
        "early_exit_ratio": stats.get("convergence", 0) / total,
        "markov_ratio": stats.get("markov", 0) / total,
        "curvature_ratio": stats.get("curvature", 0) / total,
    }


def run_pytest() -> dict:
    """Run pytest with --durations=10 and capture results."""
    reset_termination_stats()
    start = time.perf_counter()
    result = subprocess.run(
        ["pytest", "-q", "--durations=10"],
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()
    stats = get_termination_stats()
    return {
        "returncode": result.returncode,
        "runtime_seconds": round(end - start, 4),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-10:]),
        "slow_tests": extract_slowest_tests(result.stdout),
        "termination_stats": stats,
        "efficiency_metrics": compute_efficiency_metrics(stats),
    }


def main() -> None:
    # warm run
    run_pytest()
    # measured run
    result = run_pytest()
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"benchmark_v68_3_{timestamp}.json")
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    out_path.write_text(json.dumps(result, indent=2))
    print("Benchmark complete:")
    print(json.dumps(result, indent=2))
    # propagate pytest failure to caller/CI
    if result["returncode"] != 0:
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
