# SPDX-License-Identifier: MPL-2.0
"""CLI for the v172.0 matrix and v172.1 marker contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qec.sonify.canonical import canonical_json

from .core import LINK_STATES, demo_matrix, validate_matrix_manifest
from .marker import (
    MAX_PAYLOAD_BYTES,
    CrossbarRequest,
    compile_marker_program,
    execute_marker_program,
    validate_common_control_receipt,
)


def _read_json(path: Path) -> object:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)



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
    marker = sub.add_parser("marker", help="compute a sealed exact-coordinate plan")
    marker.add_argument("--manifest", required=True, type=Path)
    marker.add_argument("--request-id", required=True)
    marker.add_argument("--horizontal-link-id", required=True)
    marker.add_argument("--vertical-link-id", required=True)
    marker.add_argument("--payload-file", required=True, type=Path)
    marker.add_argument("--decoder-output-sha256", required=True)
    marker.add_argument("--marker-id", default="marker-0")
    marker.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-marker"))

    marker_validate = sub.add_parser("marker-validate", help="replay a common-control receipt")
    marker_validate.add_argument("--receipt", required=True, type=Path)
    marker_validate.add_argument("--expected-matrix-sha256")
    marker_validate.add_argument("--expected-request-sha256")
    marker_validate.add_argument("--expected-program-sha256")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "marker-validate":
            validation = validate_common_control_receipt(
                _read_json(args.receipt),
                expected_matrix_sha256=args.expected_matrix_sha256,
                expected_request_sha256=args.expected_request_sha256,
                expected_program_sha256=args.expected_program_sha256,
            )
            print(canonical_json(validation))
            return 0
        if args.command == "marker":
            manifest = _read_json(args.manifest)
            with args.payload_file.open("rb") as stream:
                payload = stream.read(MAX_PAYLOAD_BYTES + 1)
            request = CrossbarRequest(
                args.request_id, args.horizontal_link_id, args.vertical_link_id,
                payload, args.decoder_output_sha256,
            )
            program = compile_marker_program(manifest, request, marker_id=args.marker_id)
            receipt = execute_marker_program(manifest, request, program)
            validation = validate_common_control_receipt(
                receipt,
                expected_matrix_sha256=manifest["sha256"],
                expected_request_sha256=request.as_dict()["sha256"],
                expected_program_sha256=program["sha256"],
            )
            for name, value in (
                ("crossbar_marker_input_register.json", request.as_dict()),
                ("crossbar_marker_program.json", program),
                ("crossbar_common_control_receipt.json", receipt),
                ("crossbar_common_control_validation.json", validation),
            ):
                _write(args.output_dir / name, value)
            print(canonical_json(validation))
            # A valid rejection receipt is evidence, not successful selection.
            return 0 if receipt["outcome"] == "plan_selected" else 2
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
