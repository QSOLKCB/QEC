"""CLI for validating the published NEXUS v4.0.1 qBraid evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qec.sonify.canonical import canonical_json
from .replication import (
    EXPECTED_ARCHIVE_SHA256,
    validate_qbraid_replication_archive,
    validate_replication_receipt,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Validate NEXUS qBraid replication evidence and receipts."
    )
    sub = root.add_subparsers(dest="command", required=True)

    archive = sub.add_parser("archive")
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument(
        "--expected-sha256",
        default=EXPECTED_ARCHIVE_SHA256,
    )
    archive.add_argument("--output", type=Path, required=True)

    receipt = sub.add_parser("receipt")
    receipt.add_argument("--receipt", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "archive":
        result = validate_qbraid_replication_archive(
            args.archive,
            expected_archive_sha256=args.expected_sha256,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            canonical_json(result) + "\n",
            encoding="utf-8",
        )
    else:
        payload = json.loads(args.receipt.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("replication receipt must contain a JSON object")
        result = validate_replication_receipt(payload)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
