# SPDX-License-Identifier: MPL-2.0
"""Offline JSON entry points for the stored-program switching skeleton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json
from .core import SwitchInput, demo_switch_input, execute_switch, validate_switch_receipt


def _read(path: Path) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("INVALID_INPUT")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qec-ess")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="write a canonical two-call skeleton input")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/ess"))
    run = sub.add_parser("run", help="dispatch the closed queue through the fabric boundary")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/ess"))
    validate = sub.add_parser("validate", help="replay every command and native result")
    validate.add_argument("--receipt", required=True, type=Path)
    validate.add_argument("--expected-input-sha256")
    validate.add_argument("--expected-program-store-sha256")
    validate.add_argument("--expected-adapter-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            value = demo_switch_input().as_dict()
            _write(args.output_dir / "ess_switch_input.json", value)
        elif args.command == "run":
            source = SwitchInput.from_dict(_read(args.input))
            receipt = execute_switch(source)
            value = validate_switch_receipt(receipt)
            for name, artifact in (("ess_switch_input", source.as_dict()),
                                   ("ess_switch_skeleton_receipt", receipt),
                                   ("ess_command_stream", receipt["command_stream"]),
                                   ("ess_switch_skeleton_validation", value)):
                _write(args.output_dir / (name + ".json"), artifact)
        else:
            value = validate_switch_receipt(
                _read(args.receipt), expected_input_sha256=args.expected_input_sha256,
                expected_program_store_sha256=args.expected_program_store_sha256,
                expected_adapter_sha256=args.expected_adapter_sha256,
            )
        print(canonical_json(value))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
