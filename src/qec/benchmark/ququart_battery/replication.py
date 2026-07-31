"""Canonical receipts for independent ququart-battery replications."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qec.sonify.canonical import canonical_json, canonical_sha256

REPLICATION_SCHEMA = "qec.ququart-fer-replication.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREFIX_RE = re.compile(r"^[0-9a-f]{8,63}$")

V170_1_0_RELEASE_MANIFEST_SHA256 = (
    "138a3554de455cc9055bf5e83ea36e9024b023c3700d2d92e4d7dd4c196de0ce"
)
V170_1_0_ARTIFACTS = {
    "exact_weight_enumerator.csv": (
        "f273cd3ea0a43357ff0c979a109042451ca5926643490b164b5ab742b869551d"
    ),
    "exact_fer_curve.csv": (
        "c93dc243fb94606dc8b55e7a960ecebfd5814f1af6dfe5f6fbd6de2d6f760c0a"
    ),
    "monte_carlo_fer.csv": (
        "5054c22df53cf9c170e345bbbc7d762f52cac7d694ed49d2e39cd347672e44dd"
    ),
    "harmonic_fault_matrix.csv": (
        "c6f744bacb07f2dfc96fbd578d0f222ab8218da5372216e1e356bfbfd009a9b2"
    ),
    "harmonic_end_to_end.csv": (
        "57c0b7e71955750136bd6eaceae8d3d93754e93a51e8572d5c90f51e81c8a06b"
    ),
    "methodology.json": (
        "6ea9b0d8957c15e181f951f402010c49968413d53f794d3675b8a772172a9565"
    ),
    "report.js": (
        "7bf5787b38ac2ff775d9745ae55bd6e5aa5dc4d046c3e5621147c02943f842ac"
    ),
}


class ReplicationReceiptError(ValueError):
    """Raised when a replication declaration overstates its verification."""


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReplicationReceiptError(f"{field} must be a full lowercase SHA-256")
    return value


def _prefix(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _PREFIX_RE.fullmatch(value):
        raise ReplicationReceiptError(
            f"{field} must be an 8-to-63-character lowercase SHA-256 prefix"
        )
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplicationReceiptError(f"{field} must be a non-empty string")
    return value


def _artifact_status(observation: Mapping[str, Any]) -> dict[str, object]:
    name = observation.get("name")
    if not isinstance(name, str) or not name:
        raise ReplicationReceiptError("artifact observation needs a name")
    kind = observation.get("kind")
    if kind not in {"deterministic", "sampled", "parameter_bound"}:
        raise ReplicationReceiptError(
            f"artifact {name} kind must be deterministic, sampled, or parameter_bound"
        )
    release_sha = _sha256(observation.get("release_sha256"), f"{name}.release_sha256")
    observed_sha = observation.get("observed_sha256")
    observed_prefix = observation.get("observed_sha256_prefix")
    parameters_match = observation.get("parameters_match") is True

    if observed_sha is not None:
        full = _sha256(observed_sha, f"{name}.observed_sha256")
        status = "full_match" if full == release_sha else "mismatch"
        verified_characters = 64
    elif observed_prefix is not None:
        prefix = _prefix(observed_prefix, f"{name}.observed_sha256_prefix")
        status = "prefix_consistent" if release_sha.startswith(prefix) else "mismatch"
        verified_characters = len(prefix)
    elif kind in {"sampled", "parameter_bound"} and not parameters_match:
        status = "parameter_variant_expected"
        verified_characters = 0
    else:
        status = "unverified"
        verified_characters = 0

    return {
        "name": name,
        "kind": kind,
        "release_sha256": release_sha,
        "parameters_match": parameters_match,
        "status": status,
        "verified_sha256_characters": verified_characters,
    }


def build_replication_receipt(
    declaration: Mapping[str, Any],
) -> dict[str, object]:
    """Validate and hash-bind an independent replication declaration."""

    target = declaration.get("target", {})
    environment = declaration.get("environment", {})
    parameters = declaration.get("parameters", {})
    source = declaration.get("source", {})
    observations = declaration.get("artifacts", [])
    if not isinstance(target, Mapping):
        raise ReplicationReceiptError("target must be an object")
    if not isinstance(environment, Mapping):
        raise ReplicationReceiptError("environment must be an object")
    if not isinstance(parameters, Mapping):
        raise ReplicationReceiptError("parameters must be an object")
    if not isinstance(source, Mapping):
        raise ReplicationReceiptError("source must be an object")
    if isinstance(observations, str) or not isinstance(observations, Sequence):
        raise ReplicationReceiptError("artifacts must be a sequence")

    target_release = _nonempty_text(target.get("release"), "target.release")
    target_commit = _nonempty_text(target.get("commit"), "target.commit")
    release_manifest = _sha256(
        target.get("release_manifest_sha256"),
        "target.release_manifest_sha256",
    )
    _sha256(source.get("document_sha256"), "source.document_sha256")
    if source.get("reported_manifest_sha256") is not None:
        _sha256(
            source.get("reported_manifest_sha256"),
            "source.reported_manifest_sha256",
        )
    if source.get("reported_methodology_sha256") is not None:
        _sha256(
            source.get("reported_methodology_sha256"),
            "source.reported_methodology_sha256",
        )

    statuses = tuple(_artifact_status(item) for item in observations)
    if any(item["status"] == "mismatch" for item in statuses):
        raise ReplicationReceiptError("one or more artifact observations mismatch")

    deterministic = tuple(
        item for item in statuses if item["kind"] == "deterministic"
    )
    if deterministic and all(item["status"] == "full_match" for item in deterministic):
        deterministic_status = "full_hash_match"
    elif deterministic and all(
        item["status"] in {"full_match", "prefix_consistent"}
        for item in deterministic
    ):
        deterministic_status = "prefix_consistent"
    else:
        deterministic_status = "unverified"

    sampled = tuple(item for item in statuses if item["kind"] != "deterministic")
    sampled_status = (
        "parameter_variant_expected"
        if sampled
        and all(
            item["status"] in {
                "parameter_variant_expected",
                "prefix_consistent",
                "full_match",
            }
            for item in sampled
        )
        else "unverified"
    )

    receipt: dict[str, object] = {
        "schema": REPLICATION_SCHEMA,
        "target": {
            "release": target_release,
            "commit": target_commit,
            "release_manifest_sha256": release_manifest,
        },
        "environment": dict(environment),
        "parameters": dict(parameters),
        "source": dict(source),
        "artifact_verification": list(statuses),
        "verification": {
            "deterministic_artifacts": deterministic_status,
            "sampled_artifacts": sampled_status,
            "full_cross_environment_hash_match_claimed": (
                deterministic_status == "full_hash_match"
            ),
            "parameter_bound_replication": True,
        },
        "claim": (
            "Independent software replication under declared parameters. "
            "Prefix consistency is not promoted to a full-hash match."
        ),
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return receipt


def qbraid_v170_1_0_receipt() -> dict[str, object]:
    """Return the corrected parameter-bound receipt for the supplied qBraid run."""

    report_prefixes = {
        "exact_weight_enumerator.csv": "f273cd3e",
        "exact_fer_curve.csv": "c93dc243",
        "monte_carlo_fer.csv": "198c2f31",
        "harmonic_fault_matrix.csv": "c6f744ba",
        "harmonic_end_to_end.csv": "7a71f59a",
        "methodology.json": "7e6ebd59",
        "report.js": "7fef4f98",
    }
    deterministic_names = {
        "exact_weight_enumerator.csv",
        "exact_fer_curve.csv",
        "harmonic_fault_matrix.csv",
    }
    artifacts = []
    for name, release_sha in V170_1_0_ARTIFACTS.items():
        kind = "deterministic" if name in deterministic_names else "parameter_bound"
        item: dict[str, object] = {
            "name": name,
            "kind": kind,
            "release_sha256": release_sha,
            "parameters_match": kind == "deterministic",
        }
        prefix = report_prefixes[name]
        if kind == "deterministic":
            item["observed_sha256_prefix"] = prefix
        artifacts.append(item)

    return build_replication_receipt({
        "target": {
            "release": "170.1.0",
            "commit": "32fdf9c88f4ac1b873c8acc4d3d54c50f87d6e0a",
            "release_manifest_sha256": V170_1_0_RELEASE_MANIFEST_SHA256,
        },
        "environment": {
            "platform": "qBraid Lab",
            "operating_system": "Ubuntu 24.04",
            "python": "3.12",
            "execution_date": "2026-07-31",
        },
        "parameters": {
            "seed": 1701001,
            "monte_carlo_trials_per_cell": 1000,
            "harmonic_trials_per_cell": 500,
            "release_monte_carlo_trials_per_cell": 10000,
            "release_harmonic_trials_per_cell": 4000,
        },
        "source": {
            "document": "QEC_FER_BATTERY_REPORT.md",
            "document_sha256": (
                "22f9d219d88f6dfe923d0d14d8a1ef387b24c60649541fd912e408a846e20ccf"
            ),
            "reported_manifest_sha256": (
                "16a40ff502aafff7896c1102a1f92af8c744a5f0cfdcd9c361eb3a1e0a4b9426"
            ),
            "reported_methodology_sha256": (
                "752fc50abd331406e70d91d1a1eb9bbd52033ef397e16f5dbf6ed20284d4d0ff"
            ),
            "hash_visibility": "eight-character prefixes for artifact table",
        },
        "artifacts": artifacts,
    })


def write_replication_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(dict(receipt)) + "\n", encoding="utf-8")
