# SPDX-License-Identifier: MPL-2.0
"""CLI for the deterministic Panel separated-control exchange."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from qec.sonify.canonical import canonical_json
from .core import PanelExchange, PanelFaultPlan, PanelRequest, build_claim_validation, build_fault_battery, demo_topology, demo_translation, validate_route_receipt

def csv_ints(text: str) -> tuple[int, ...]:
    try: values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values: raise argparse.ArgumentTypeError("digits may not be empty")
    return values

def csv_names(text: str) -> tuple[str, ...]:
    return tuple(sorted(x.strip() for x in text.split(",") if x.strip()))

def write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qec-panel")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("route")
    r.add_argument("--digits", required=True, type=csv_ints); r.add_argument("--request-id", default="panel-demo")
    r.add_argument("--destination", default="correction/demo"); r.add_argument("--epoch", type=int, default=0)
    r.add_argument("--payload-text", default="qec-correction-request"); r.add_argument("--translation-version", default="1")
    r.add_argument("--busy-banks", type=csv_names, default=()); r.add_argument("--stalled-motor-groups", type=csv_names, default=())
    r.add_argument("--translation-corruption", action="store_true"); r.add_argument("--sender-disagreement", action="store_true")
    r.add_argument("--output-dir", type=Path, default=Path("artifacts/panel"))
    v = sub.add_parser("validate"); v.add_argument("--receipt", required=True, type=Path)
    return p

def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            value = json.loads(args.receipt.read_text(encoding="utf-8"))
            if not isinstance(value, dict): raise ValueError("receipt must be a JSON object")
            print(canonical_json(validate_route_receipt(value))); return 0
        request = PanelRequest(args.request_id, args.digits, args.epoch, args.destination, args.payload_text.encode())
        exchange = PanelExchange(demo_topology(args.destination), demo_translation(args.digits, args.destination, version=args.translation_version))
        faults = PanelFaultPlan(args.busy_banks, args.stalled_motor_groups, args.translation_corruption, args.sender_disagreement)
        result = exchange.route(request, faults=faults)
        claims = build_claim_validation(result.receipt); battery = build_fault_battery(exchange, request)
        files = {"panel_topology.json": result.receipt["topology"], "panel_digit_register.json": result.receipt["digit_register"], "panel_sender_program.json": result.receipt["sender_program"], "panel_route_receipt.json": result.receipt, "panel_fault_battery.json": battery, "panel_claim_validation.json": claims}
        for name, value in files.items(): write(args.output_dir / name, value)
        print(canonical_json({"outcome": result.outcome, "panel_sender_register_receipt_hash": result.receipt["panel_sender_register_receipt_hash"], "panel_separated_control_receipt_hash": result.receipt["sha256"], "claim_validation_passed": claims["all_passed"]}))
        return 0 if result.outcome == "committed" else 2
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"qec-panel: {exc}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
