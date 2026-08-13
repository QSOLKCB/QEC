# SPDX-License-Identifier: MPL-2.0
"""CLI for the deterministic v172.0 Crossbar matrix core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qec.sonify.canonical import canonical_json

from .core import LINK_STATES, demo_matrix, validate_matrix_manifest


def _link_state(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("link state must use LINK_ID=STATE")
    link_id, state = (part.strip() for part in text.split("=", 1))
    if not link_id:
        raise argparse.ArgumentTypeError("link id must be non-empty")
    if state not in LINK_STATES:
        raise argparse.ArgumentTypeError(f"state must be one of {LINK_STATES}")
    return link_id, state


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="qec-crossbar")
    sub = command.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="emit a canonical v172.0 matrix manifest")
    matrix.add_argument("--matrix-id", default="crossbar-demo")
    matrix.add_argument("--horizontal-count", type=int, default=4)
    matrix.add_argument("--vertical-count", type=int, default=4)
    matrix.add_argument(
        "--link-state",
        action="append",
        type=_link_state,
        default=[],
        metavar="LINK_ID=STATE",
        help="override a deterministic demo link state; may be repeated",
    )
    matrix.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar"))

    validate = sub.add_parser("validate", help="replay-validate a matrix manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            value = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("manifest must be a JSON object")
            print(canonical_json(validate_matrix_manifest(value)))
            return 0

        overrides: dict[str, str] = {}
        for link_id, state in args.link_state:
            if link_id in overrides:
                raise ValueError(f"duplicate link-state override for {link_id}")
            overrides[link_id] = state

        matrix = demo_matrix(
            args.matrix_id,
            horizontal_count=args.horizontal_count,
            vertical_count=args.vertical_count,
            state_overrides=overrides,
        )
        manifest = matrix.as_dict()
        validation = validate_matrix_manifest(manifest)
        _write(args.output_dir / "crossbar_matrix_manifest.json", manifest)
        _write(args.output_dir / "crossbar_matrix_validation.json", validation)
        print(
            canonical_json(
                {
                    "matrix_id": matrix.matrix_id,
                    "crossbar_matrix_receipt_hash": manifest["sha256"],
                    "intersection_count": len(matrix.intersections),
                    "validation_passed": validation["all_passed"],
                }
            )
        )
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"qec-crossbar: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
