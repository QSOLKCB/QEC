# SPDX-License-Identifier: MPL-2.0
"""CLI for the deterministic Strowger syndrome exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json

from .exchange import StrowgerExchange
from .model import ExchangeConfig, ExchangeMode, FaultPlan, RouteRequest, StageConfig
from .receipts import validate_receipt


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _mode(value: str) -> ExchangeMode:
    try:
        return ExchangeMode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode must be automatic, supervised, or manual") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qec-strowger",
        description="Deterministic mixed-radix syndrome routing exchange",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="route one correction request")
    route.add_argument("--digits", required=True, type=_csv_ints)
    route.add_argument("--radices", required=True, type=_csv_ints)
    route.add_argument("--trunks", type=int, default=10)
    route.add_argument("--linefinders", type=int, default=4)
    route.add_argument("--request-id", default="demo-call")
    route.add_argument("--destination", default="correction/demo")
    route.add_argument("--epoch", type=int, default=0)
    route.add_argument("--mode", type=_mode, default=ExchangeMode.AUTOMATIC)
    route.add_argument("--tone-offsets-hz", type=_csv_ints, default=(0, 0, 0))
    route.add_argument("--output", type=Path)

    check = sub.add_parser("validate", help="validate a route receipt")
    check.add_argument("--receipt", required=True, type=Path)
    return parser


def _config(radices: tuple[int, ...], trunks: int, linefinders: int) -> ExchangeConfig:
    if len(radices) < 3:
        raise ValueError("radices must contain at least one selector and two connector axes")
    selectors = tuple(
        StageConfig(name=f"selector-{index}", radix=radix, trunks=trunks)
        for index, radix in enumerate(radices[:-2])
    )
    return ExchangeConfig(
        linefinders=linefinders,
        selectors=selectors,
        connector_vertical_radix=radices[-2],
        connector_rotary_radix=radices[-1],
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            payload = json.loads(args.receipt.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("receipt must be a JSON object")
            print(canonical_json(validate_receipt(payload)))
            return 0

        if len(args.tone_offsets_hz) != 3:
            raise ValueError("tone offsets must contain exactly three integers")
        config = _config(args.radices, args.trunks, args.linefinders)
        request = RouteRequest(
            request_id=args.request_id,
            digits=args.digits,
            epoch=args.epoch,
            destination=args.destination,
        )
        result = StrowgerExchange(config, mode=args.mode).route(
            request,
            faults=FaultPlan(tone_offsets_hz=args.tone_offsets_hz),
        )
        text = canonical_json(result.receipt) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0 if result.outcome.value == "committed" else 2
    except (OSError, ValueError, TypeError, PermissionError, json.JSONDecodeError) as exc:
        print(f"qec-strowger: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
