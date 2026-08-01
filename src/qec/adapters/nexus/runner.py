"""Fail-closed NEXUS subprocess adapter and artifact writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import BinaryIO

from qec.sonify.canonical import canonical_json
from .attestation import validate_build_attestation
from .contract import NexusInvocation
from .evidence import (
    build_execution_receipt,
    validate_execution_bundle,
)

MAX_BINARY_BYTES = 64 * 1024 * 1024
MAX_STDOUT_BYTES = 64 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class NexusExecutionError(RuntimeError):
    pass


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NexusExecutionError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NexusExecutionError(f"{label} must contain a JSON object")
    return payload


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NexusExecutionError(f"cannot read {label}: {exc}") from exc


def _copy_private_binary(source: Path, destination: Path) -> None:
    total = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output:
            while True:
                chunk = input_stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BINARY_BYTES:
                    raise NexusExecutionError(
                        "NEXUS binary exceeds the adapter byte limit"
                    )
                output.write(chunk)
        destination.chmod(0o700)
    except NexusExecutionError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise NexusExecutionError(
            f"cannot create private NEXUS executable copy: {exc}"
        ) from exc
    if total == 0:
        destination.unlink(missing_ok=True)
        raise NexusExecutionError("NEXUS binary may not be empty")


def _capture_stream(
    stream: BinaryIO,
    *,
    limit: int,
    label: str,
    process: subprocess.Popen[bytes],
    output: dict[str, bytes],
    failures: list[BaseException],
    overflows: list[str],
) -> None:
    data = bytearray()
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            if len(data) + len(chunk) > limit:
                overflows.append(label)
                try:
                    process.kill()
                except OSError:
                    pass
                break
            data.extend(chunk)
    except BaseException as exc:  # pragma: no cover - defensive I/O boundary
        failures.append(exc)
        try:
            process.kill()
        except OSError:
            pass
    finally:
        try:
            stream.close()
        except OSError:
            pass
        output[label] = bytes(data)


def _run_capped(
    argv: list[str],
    *,
    environment: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, bytes, bytes]:
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise NexusExecutionError(f"NEXUS execution failed: {exc}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    output: dict[str, bytes] = {}
    failures: list[BaseException] = []
    overflows: list[str] = []
    threads = [
        threading.Thread(
            target=_capture_stream,
            kwargs={
                "stream": process.stdout,
                "limit": MAX_STDOUT_BYTES,
                "label": "stdout",
                "process": process,
                "output": output,
                "failures": failures,
                "overflows": overflows,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_capture_stream,
            kwargs={
                "stream": process.stderr,
                "limit": MAX_STDERR_BYTES,
                "label": "stderr",
                "process": process,
                "output": output,
                "failures": failures,
                "overflows": overflows,
            },
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads):
        process.kill()
        raise NexusExecutionError("NEXUS output readers did not terminate")
    if timed_out:
        raise NexusExecutionError(
            f"NEXUS execution timed out after {timeout_seconds} seconds"
        )
    if overflows:
        labels = ", ".join(sorted(set(overflows)))
        raise NexusExecutionError(
            f"NEXUS {labels} exceeded the adapter byte limit"
        )
    if failures:
        raise NexusExecutionError(
            f"cannot read bounded NEXUS output: {failures[0]}"
        ) from failures[0]
    return (
        process.returncode,
        output.get("stdout", b""),
        output.get("stderr", b""),
    )


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
    attestation = _load_json_object(
        attestation_path,
        "NEXUS build attestation",
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
    }
    with tempfile.TemporaryDirectory(prefix="qec-nexus-") as temporary:
        private_binary = Path(temporary) / "nexus"
        _copy_private_binary(binary, private_binary)
        attestation_result = validate_build_attestation(
            attestation,
            profile=invocation.profile,
            binary=private_binary,
        )
        returncode, stdout_bytes, stderr_bytes = _run_capped(
            invocation.argv(private_binary),
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    if returncode != 0:
        stderr = stderr_bytes.decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise NexusExecutionError(
            f"NEXUS exited with {returncode}: {stderr}"
        )
    if stderr_bytes.strip():
        raise NexusExecutionError(
            "successful NEXUS execution emitted unexpected stderr"
        )
    receipt = build_execution_receipt(
        invocation,
        stdout_bytes=stdout_bytes,
        binary_sha256=str(attestation_result["binary_sha256"]),
        build_attestation_sha256=str(attestation_result["sha256"]),
    )
    (output_dir / "nexus_output.csv").write_bytes(stdout_bytes)
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
    receipt = _load_json_object(path, "NEXUS execution receipt")
    directory = path.parent
    request = _load_json_object(
        directory / "request.json",
        "NEXUS request",
    )
    source_identity = _load_json_object(
        directory / "source_identity.json",
        "NEXUS source identity",
    )
    build_attestation = _load_json_object(
        directory / "build_attestation.json",
        "NEXUS build attestation",
    )
    stdout_bytes = _read_bytes(
        directory / "nexus_output.csv",
        "NEXUS CSV evidence",
    )
    return validate_execution_bundle(
        receipt,
        request=request,
        source_identity=source_identity,
        build_attestation=build_attestation,
        stdout_bytes=stdout_bytes,
    )
