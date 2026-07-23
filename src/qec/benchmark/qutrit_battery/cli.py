"""Command-line entry point for the qutrit decoder benchmark battery."""

from __future__ import annotations

import argparse
from pathlib import Path

from qec.sonify.canonical import canonical_json

from .report import build_report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build deterministic qutrit QEC benchmark artifacts.",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/qutrit_decoder_v1"),
    )
    result.add_argument(
        "--v3-baseline",
        type=Path,
        default=Path("qec_data_prepared.csv"),
    )
    result.add_argument(
        "--stress-limit",
        type=int,
        default=2048,
        help="Maximum deterministic corpus patterns per code and weight.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = build_report(
        args.output,
        v3_baseline_path=args.v3_baseline,
        stress_limit_per_weight=args.stress_limit,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
