#!/usr/bin/env python3
"""CLI entry point for the BP dynamics benchmark + stress suite (v68.7).

Usage:
    python scripts/run_benchmark_stress.py
    python scripts/run_benchmark_stress.py --seed 123
    python scripts/run_benchmark_stress.py --output benchmark_results.json
    python scripts/run_benchmark_stress.py --summary

Produces deterministic JSON output.  Repeated runs with the same seed
produce byte-identical artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.qec.experiments.benchmark_stress import (
    run_benchmark_suite,
    serialize_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BP dynamics benchmark + stress baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for deterministic runs (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON path (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also write benchmark_summary.md",
    )
    args = parser.parse_args()

    artifact = run_benchmark_suite(master_seed=args.seed)
    json_str = serialize_artifact(artifact)

    out_path = Path(args.output)
    out_path.write_text(json_str, encoding="utf-8")
    print(f"Wrote {out_path} ({len(artifact['runs'])} scenarios)")

    # Print outcome summary
    summary = artifact["outcome_summary"]
    print(f"Outcomes: {summary}")

    if args.summary:
        md = _generate_summary_md(artifact)
        md_path = out_path.with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")
        print(f"Wrote {md_path}")


def _generate_summary_md(artifact: dict) -> str:
    """Generate a short markdown summary of benchmark results."""
    lines = [
        f"# Benchmark Stress Baseline v{artifact['benchmark_version']}",
        "",
        f"- **Master seed**: {artifact['master_seed']}",
        f"- **Git hash**: {artifact.get('git_hash', 'N/A')}",
        f"- **Scenarios**: {artifact['n_scenarios']}",
        "",
        "## Outcome Summary",
        "",
    ]
    for outcome, count in sorted(artifact["outcome_summary"].items()):
        lines.append(f"- {outcome}: {count}")
    lines.append("")
    lines.append("## Per-Scenario Results")
    lines.append("")
    lines.append("| Scenario | Regime | Outcome | Wall Time (s) |")
    lines.append("|----------|--------|---------|---------------|")
    for run in artifact["runs"]:
        lines.append(
            f"| {run['scenario']} | {run['regime']} "
            f"| {run['outcome']} | {run['wall_time_seconds']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
