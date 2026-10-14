# SPDX-License-Identifier: MPL-2.0
"""Explicit v173.5 offline entry point; the released qec-ess CLI is unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json
from . import programme
from .lifecycle import (
    LifecycleInput, create_fabric_equivalence_receipt, demo_lifecycle_input,
    execute_call_processing, validate_call_processing_receipt, validate_fabric_equivalence_receipt,
)


def _read(path: Path) -> object:
    # Bound the bytes actually read; a pre-read stat cannot close a growth race.
    with path.open("rb") as stream:
        raw = stream.read(programme.MAX_JSON_BYTES + 1)
    if len(raw) > programme.MAX_JSON_BYTES:
        raise ValueError("INVALID_INPUT")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("INVALID_INPUT")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    programme.bounded_json(value)
    return value


def _write(path: Path, value: dict) -> None:
    programme.bounded_json(value)
    # Do not append an unbudgeted newline: actual bytes match the preflight count.
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > programme.MAX_JSON_BYTES:
        raise ValueError("INVALID_INPUT")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qec.routing.ess.lifecycle_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "equivalence-demo"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=Path("artifacts/ess-lifecycle"))
        if name == "run":
            command.add_argument("--input", required=True, type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", required=True, type=Path)
    for field in ("input", "programme", "adapter", "call-store"):
        validate.add_argument("--expected-" + field + "-sha256")
    validate_eq = sub.add_parser("validate-equivalence")
    validate_eq.add_argument("--receipt", required=True, type=Path)
    for field in ("left-input", "right-input", "programme"):
        validate_eq.add_argument("--expected-" + field + "-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            value = demo_lifecycle_input().as_dict()
            _write(args.output_dir / "ess_lifecycle_input.json", value)
        elif args.command == "run":
            source = LifecycleInput.from_dict(_read(args.input))
            receipt = execute_call_processing(source)
            value = validate_call_processing_receipt(receipt)
            for name, artifact in (
                ("ess_lifecycle_input", source.as_dict()),
                ("ess_program_store_manifest", source.programme.as_dict()),
                ("ess_translation_table_manifest", source.programme.translations.as_dict()),
                ("ess_feature_module_manifest", source.programme.features.as_dict()),
                ("ess_event_schedule_receipt", receipt["schedule"]),
                ("ess_call_processing_receipt", receipt),
                ("ess_call_processing_validation", value),
            ):
                _write(args.output_dir / (name + ".json"), artifact)
        elif args.command == "equivalence-demo":
            receipt = create_fabric_equivalence_receipt(demo_lifecycle_input(),
                                                        demo_lifecycle_input(alternate_fabric=True))
            value = validate_fabric_equivalence_receipt(receipt)
            _write(args.output_dir / "ess_fabric_equivalence_receipt.json", receipt)
            _write(args.output_dir / "ess_fabric_equivalence_validation.json", value)
        else:
            pins = {key: value for key, value in vars(args).items() if key.startswith("expected_")}
            validator = validate_call_processing_receipt if args.command == "validate" else validate_fabric_equivalence_receipt
            value = validator(_read(args.receipt), **pins)
        print(canonical_json(value))
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
