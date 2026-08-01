"""Fail-closed NEXUS subprocess adapter and artifact writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from qec.sonify.canonical import canonical_json
from .attestation import validate_build_attestation
from .contract import NexusInvocation
from .evidence import (
    build_execution_receipt,
    parse_csv,
    validate_execution_receipt,
)


class NexusExecutionError(RuntimeError):
    pass


def run_nexus(
    invocation: NexusInvocation,
    *,
    binary: Path,
    attestation_path: Path,
    output_dir: Path,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be a positive exact integer"
        )
    binary = binary.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        attestation = json.loads(
            attestation_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NexusExecutionError(
            f"cannot read NEXUS build attestation: {exc}"
        ) from exc
    if not isinstance(attestation, dict):
        raise NexusExecutionError(
            "NEXUS build attestation must be a JSON object"
        )
    attestation_result = validate_build_attestation(
        attestation,
        profile=invocation.profile,
        binary=binary,
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
    }
    try:
        completed = subprocess.run(
            invocation.argv(binary),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NexusExecutionError(f"NEXUS execution failed: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise NexusExecutionError(
            f"NEXUS exited with {completed.returncode}: {stderr}"
        )
    if completed.stderr.strip():
        raise NexusExecutionError(
            "successful NEXUS execution emitted unexpected stderr"
        )
    try:
        stdout_text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NexusExecutionError(
            "NEXUS stdout must be UTF-8 CSV"
        ) from exc
    rows = parse_csv(stdout_text)
    receipt = build_execution_receipt(
        invocation,
        rows=rows,
        stdout_bytes=completed.stdout,
        binary_sha256=str(attestation_result["binary_sha256"]),
        build_attestation_sha256=str(attestation_result["sha256"]),
    )
    (output_dir / "nexus_output.csv").write_bytes(completed.stdout)
    (output_dir / "request.json").write_text(
        canonical_json(invocation.as_dict()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_identity.json").write_text(
        canonical_json(invocation.source.as_dict()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "build_attestation.json").write_text(
        canonical_json(attestation) + "\n",
        encoding="utf-8",
    )
    (output_dir / "execution_receipt.json").write_text(
        canonical_json(receipt) + "\n",
        encoding="utf-8",
    )
    return receipt


def validate_receipt_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NexusExecutionError(
            f"cannot read NEXUS execution receipt: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise NexusExecutionError(
            "receipt file must contain a JSON object"
        )
    return validate_execution_receipt(payload)
