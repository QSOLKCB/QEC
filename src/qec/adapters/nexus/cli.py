"""Command-line entry point for the QEC v170.2.0 NEXUS bridge."""
from __future__ import annotations

import argparse
from pathlib import Path

from qec.sonify.canonical import canonical_json
from .attestation import build_attestation
from .contract import NexusConfig, NexusInvocation
from .runner import run_nexus, validate_receipt_file
from .source import PROFILES


def _config(args: argparse.Namespace) -> NexusConfig:
    return NexusConfig(
        logical=args.logical,
        rendered=args.rendered,
        particles=args.particles,
        radius=args.radius,
        phase=args.phase,
        turns=args.turns,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="v4.0.0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical", type=int, default=16_777_216)
    parser.add_argument("--rendered", type=int, default=1_024)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--radius", default="0.56")
    parser.add_argument("--phase", default="0")
    parser.add_argument("--turns", default="1.5")
    parser.add_argument("--timeout", type=int, default=120)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Run pinned NEXUS geometry through QEC evidence contracts."
        )
    )
    sub = root.add_subparsers(dest="command", required=True)

    attest = sub.add_parser("attest-build")
    attest.add_argument("--binary", type=Path, required=True)
    attest.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="v4.0.0",
    )
    attest.add_argument("--toolchain", required=True)
    attest.add_argument("--output", type=Path, required=True)

    for command in (
        "verify",
        "verify-parallel",
        "trace",
        "fibonacci",
        "ternary",
        "receipt",
    ):
        item = sub.add_parser(command)
        _common(item)
        if command == "verify-parallel":
            item.add_argument("--workers", type=int, required=True)
        if command in {"trace", "fibonacci", "ternary"}:
            item.add_argument("--channel", type=int, required=True)
            item.add_argument("--steps", type=int, required=True)
        if command == "ternary":
            item.add_argument("--base-frequency-hz", default="432")
        if command == "receipt":
            item.add_argument("--samples", type=int, required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        result = validate_receipt_file(args.receipt)
    elif args.command == "attest-build":
        result = build_attestation(
            profile=args.profile,
            binary=args.binary,
            toolchain=args.toolchain,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            canonical_json(result) + "\n",
            encoding="utf-8",
        )
    else:
        invocation = NexusInvocation(
            operation=args.command,
            profile=args.profile,
            config=_config(args),
            channel=getattr(args, "channel", None),
            steps=getattr(args, "steps", None),
            samples=getattr(args, "samples", None),
            workers=getattr(args, "workers", None),
            base_frequency_hz=getattr(
                args,
                "base_frequency_hz",
                None,
            ),
        )
        result = run_nexus(
            invocation,
            binary=args.binary,
            attestation_path=args.attestation,
            output_dir=args.output,
            timeout_seconds=args.timeout,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
