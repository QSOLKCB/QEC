"""CLI for fail-closed validation of ququart battery report claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qec.sonify.canonical import canonical_json

from .claims import derive_evidence_facts, validate_report_claims


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Validate a machine-readable report-claims declaration against "
            "generated ququart FER evidence."
        )
    )
    result.add_argument("--claims", type=Path, required=True)
    result.add_argument("--evidence", type=Path, required=True)
    result.add_argument("--test-receipt", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    test_receipt = (
        json.loads(args.test_receipt.read_text(encoding="utf-8"))
        if args.test_receipt
        else None
    )
    lane_path = args.evidence / "lane_symmetry_certificate.json"
    lane = json.loads(lane_path.read_text(encoding="utf-8")) if lane_path.exists() else None
    facts = derive_evidence_facts(
        _csv_rows(args.evidence / "harmonic_fault_matrix.csv"),
        _csv_rows(args.evidence / "harmonic_end_to_end.csv"),
        lane_symmetry=lane,
    )
    receipt = validate_report_claims(claims, facts, test_receipt=test_receipt)
    text = canonical_json(receipt) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
