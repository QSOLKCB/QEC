"""CLI for the v170.1.0 exact ququart FER and harmonic battery."""

from __future__ import annotations

import argparse
from pathlib import Path

from qec.sonify.canonical import canonical_json

from .oracle import DEFAULT_ERROR_RATES
from .report import build_report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build exact ququart FER, corrected Monte Carlo, and harmonic "
            "fault-battery artifacts."
        )
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ququart_fer_v170_1_0"),
    )
    result.add_argument(
        "--trials",
        type=int,
        default=5000,
        help="Monte Carlo trials per physical-channel/error-rate cell.",
    )
    result.add_argument(
        "--harmonic-trials",
        type=int,
        default=2000,
        help="Trials per harmonic physical-rate/noise-sigma cell.",
    )
    result.add_argument("--seed", type=int, default=1701001)
    result.add_argument(
        "--error-rates",
        default=",".join(DEFAULT_ERROR_RATES),
        help="Comma-separated independent per-site physical error rates.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    rates = tuple(
        item.strip()
        for item in args.error_rates.split(",")
        if item.strip()
    )
    manifest = build_report(
        args.output,
        error_rates=rates,
        monte_carlo_trials=args.trials,
        harmonic_trials=args.harmonic_trials,
        seed=args.seed,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
