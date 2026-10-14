# SPDX-License-Identifier: MPL-2.0
"""Offline canonical artifact entry points: python -m qec.routing.packet."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json
from .core import PacketInput, create_packet_report, demo_packet_input, execute_packet_fabric, validate_packet_receipt
from .lab import create_packet_demo_export
from .softswitch import create_softswitch_adapter_receipt

MAX_JSON_BYTES = 256 * 1024 * 1024


def _read(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("INVALID_INPUT")
            result[key] = value
        return result
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("INVALID_INPUT")
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique)


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qec-packet")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="write a deterministic closed batch")
    demo.add_argument("--scenario", choices=("success", "failover", "overflow", "corrupt", "drop", "recovery"), default="failover")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/packet"))
    run = sub.add_parser("run", help="write canonical replayed software evidence and separate demo export")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/packet"))
    validate = sub.add_parser("validate", help="replay complete source and optionally pin trusted identities")
    validate.add_argument("--receipt", type=Path, required=True)
    for pin in ("input", "topology", "queue-policy", "fault-plan", "receipt"):
        validate.add_argument("--expected-" + pin + "-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo_packet_input(args.scenario).as_dict()
            _write(args.output_dir / "packet_fabric_input.json", result)
        elif args.command == "run":
            source = PacketInput.from_dict(_read(args.input))
            receipt = execute_packet_fabric(source)
            result = validate_packet_receipt(receipt)
            artifacts = {
                "packet_fabric_input": source.as_dict(), "canonical_packet_schema": receipt["canonical_packet_schema"],
                "route_table_manifest": source.topology.as_dict(), "queue_policy_manifest": source.queue_policy.as_dict(),
                "packet_hop_receipt_chain": receipt["hop_receipt_chain"], "packet_failover_receipt": receipt["failover_receipt"],
                "packet_fabric_receipt": receipt, "packet_fabric_validation": result,
                "packet_fabric_report": create_packet_report(receipt),
                "softswitch_adapter_receipt": create_softswitch_adapter_receipt((), packet_receipt=receipt),
                "packet_demo": create_packet_demo_export(receipt),
            }
            for name, value in artifacts.items():
                _write(args.output_dir / (name + ".json"), value)
        else:
            result = validate_packet_receipt(_read(args.receipt), **{
                "expected_" + name + "_sha256": getattr(args, "expected_" + name + "_sha256")
                for name in ("input", "topology", "queue_policy", "fault_plan", "receipt")})
        print(canonical_json(result))
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
