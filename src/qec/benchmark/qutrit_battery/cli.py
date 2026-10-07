"""Command-line entry point for the qutrit decoder benchmark battery."""

from __future__ import annotations

import argparse

from qec.sonify.canonical import canonical_json
from qec.command_specs import QUTRIT_BATTERY

from .report import build_report


def parser() -> argparse.ArgumentParser:
    return QUTRIT_BATTERY.parser()


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
