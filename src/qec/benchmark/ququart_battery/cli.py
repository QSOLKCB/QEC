"""CLI for the v170.1.1 ququart evidence and claim-validation battery."""

from __future__ import annotations

import argparse

from qec.sonify.canonical import canonical_json
from qec.command_specs import QUQUART_BATTERY

from .report import build_report


def parser() -> argparse.ArgumentParser:
    return QUQUART_BATTERY.parser()


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
