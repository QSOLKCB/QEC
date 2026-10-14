# SPDX-License-Identifier: MPL-2.0
"""Bounded offline JSON CLI: python -m qec.routing.digital."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json
from .core import DigitalSwitchInput, demo_digital_input, execute_digital_switch, validate_digital_receipt
from .evidence import build_digital_report, run_digital_fault_battery, validate_fault_battery
from .lab import demo_svg, demo_wav, export_digital_demo

MAX_INPUT_FILE_BYTES = 16 * 1024 * 1024


def _read(path: Path) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("INVALID_INPUT")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("INVALID_INPUT")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_FILE_BYTES + 1)
    if len(raw) > MAX_INPUT_FILE_BYTES:
        raise ValueError("INVALID_INPUT")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant)
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qec-digital")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "fault-battery", "lab"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=Path("artifacts/digital"))
        if name == "run":
            command.add_argument("--input", type=Path, required=True)
        if name == "lab":
            command.add_argument("--receipt", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--expected-input-sha256")
    validate.add_argument("--expected-request-sha256")
    validate.add_argument("--expected-frame-schema-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            value = demo_digital_input().as_dict()
            _write(args.output_dir / "digital_switch_input.json", value)
        elif args.command == "run":
            source = DigitalSwitchInput.from_dict(_read(args.input))
            receipt = execute_digital_switch(source)
            value = validate_digital_receipt(receipt, expected_input_sha256=source.as_dict()["sha256"])
            artifacts = {"digital_switch_input": source.as_dict(), "digital_frame_schema": source.frame_schema.as_dict(),
                         "digital_frame_switch_receipt": receipt, "digital_switch_validation": value,
                         "timeslot_allocation_receipt": receipt["allocation_receipt"],
                         "digital_switch_report": build_digital_report(receipt)}
            for name in ("sync_receipt", "interchange_receipt"):
                artifact_name = {"sync_receipt": "frame_sync_receipt", "interchange_receipt": "timeslot_interchange_receipt"}[name]
                if receipt[name] is not None:
                    artifacts[artifact_name] = receipt[name]
                else:
                    # A rejection must not leave a previous run's successful
                    # optional evidence under a current output filename.
                    (args.output_dir / (artifact_name + ".json")).unlink(missing_ok=True)
            for name, artifact in artifacts.items():
                _write(args.output_dir / (name + ".json"), artifact)
        elif args.command == "fault-battery":
            battery = run_digital_fault_battery()
            value = validate_fault_battery(battery)
            _write(args.output_dir / "digital_fault_battery.json", battery)
            _write(args.output_dir / "digital_fault_battery_validation.json", value)
        elif args.command == "lab":
            receipt = _read(args.receipt)
            value = export_digital_demo(receipt)
            _write(args.output_dir / "digital_lab_demo.json", value)
            (args.output_dir / "digital_lab_demo.svg").write_text(demo_svg(receipt), encoding="utf-8")
            (args.output_dir / "digital_lab_demo.wav").write_bytes(demo_wav(receipt))
        else:
            value = validate_digital_receipt(_read(args.receipt), expected_input_sha256=args.expected_input_sha256,
                                             expected_request_sha256=args.expected_request_sha256,
                                             expected_frame_schema_sha256=args.expected_frame_schema_sha256)
        print(canonical_json(value))
        return 0
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
