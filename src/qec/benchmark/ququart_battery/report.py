"""Canonical report assembly for the v170.1.1 ququart evidence layer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.demo import certificate
from qec.sonify.canonical import canonical_json, canonical_sha256

from .channels import CHANNELS, monte_carlo_rows
from .claims import (
    HARDWARE_DECLARATION_FIELDS,
    PERMITTED_CURVE_LANGUAGE,
    derive_evidence_facts,
    derived_report_claims,
    validate_report_claims,
)
from .exact_channels import (
    exact_channel_fer_curve,
    exact_channel_weight_enumerator,
    lane_symmetry_certificate,
)
from .harmonic import (
    DEFAULT_NOISE_SIGMAS,
    harmonic_end_to_end_rows,
    harmonic_fault_rows,
    receiver_operating_rows,
)
from .oracle import DEFAULT_ERROR_RATES, exact_fer_curve, exact_weight_enumerator
from .replication import qbraid_v170_1_0_receipt

SCHEMA = "qec.ququart-fer-battery.v170.1.1"
VERSION = "170.1.1"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    output_dir: Path,
    *,
    error_rates: Iterable[str] = DEFAULT_ERROR_RATES,
    monte_carlo_trials: int = 5000,
    harmonic_trials: int = 2000,
    seed: int = 1701001,
) -> dict[str, object]:
    """Build deterministic evidence, replication, and claim-validation artifacts."""

    rates = tuple(error_rates)
    if not rates:
        raise ValueError("at least one physical error rate is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    code = packed_five_ququart_code()
    weight_rows = [dict(row) for row in exact_weight_enumerator()]
    exact_rows = [dict(row) for row in exact_fer_curve(rates)]
    exact_channel_weight_rows = [
        dict(row)
        for channel in CHANNELS
        for row in exact_channel_weight_enumerator(channel)
    ]
    exact_channel_rows = [
        dict(row)
        for row in exact_channel_fer_curve(rates, channels=tuple(CHANNELS))
    ]
    monte_rows = [
        dict(row)
        for row in monte_carlo_rows(
            error_rates=rates,
            trials=monte_carlo_trials,
            seed=seed,
        )
    ]
    fault_rows = [dict(row) for row in harmonic_fault_rows()]
    harmonic_rates = tuple(
        rate for rate in rates
        if rate in {"0.001", "0.003", "0.01", "0.03"}
    )
    if not harmonic_rates:
        harmonic_rates = rates[: min(4, len(rates))]
    harmonic_rows = [
        dict(row)
        for row in harmonic_end_to_end_rows(
            physical_error_rates=harmonic_rates,
            noise_sigmas=DEFAULT_NOISE_SIGMAS,
            trials=harmonic_trials,
            seed=seed,
        )
    ]
    operating_rows = [dict(row) for row in receiver_operating_rows(harmonic_rows)]

    tables: dict[str, list[dict[str, object]]] = {
        "exact_weight_enumerator.csv": weight_rows,
        "exact_fer_curve.csv": exact_rows,
        "exact_channel_weight_enumerator.csv": exact_channel_weight_rows,
        "exact_channel_fer.csv": exact_channel_rows,
        "monte_carlo_fer.csv": monte_rows,
        "harmonic_fault_matrix.csv": fault_rows,
        "harmonic_end_to_end.csv": harmonic_rows,
        "receiver_operating_curve.csv": operating_rows,
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    base_certificate = certificate()
    lane_certificate = lane_symmetry_certificate()
    facts = derive_evidence_facts(
        fault_rows,
        harmonic_rows,
        lane_symmetry=lane_certificate,
    )
    claims = derived_report_claims(facts)
    claim_validation = validate_report_claims(claims, facts)
    replication_receipt = qbraid_v170_1_0_receipt()

    _write_json(output_dir / "lane_symmetry_certificate.json", lane_certificate)
    _write_json(output_dir / "report_claims.json", claims)
    _write_json(output_dir / "claim_validation.json", claim_validation)
    _write_json(
        output_dir / "qbraid_replication_receipt.json",
        replication_receipt,
    )

    methodology: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "analysis_scope": "finite_code_capacity",
        "code": {
            "name": code.name,
            "physical_ququarts": code.n,
            "logical_ququarts": code.k,
            "distance": code.distance_hint,
            "local_pauli_basis_size": 16,
            "nonidentity_local_errors": 15,
            "exact_patterns": 16 ** code.n,
            "v170_0_certificate_sha256": base_certificate["sha256"],
        },
        "exact_oracle": {
            "method": "two-lane exhaustive packed-Pauli coset classification",
            "weight_probability": "(p/m)^w*(1-p)^(n-w)",
            "full_channel_local_nonidentity_operators": 15,
            "restricted_channel_local_nonidentity_operators": 3,
            "channels": list(CHANNELS),
            "frame_error_definition": (
                "detected_uncorrectable + accepted_nonstabilizer_residual"
            ),
            "small_p_leading_order_full_channel": "10*p^2 + O(p^3)",
            "lane_symmetry_certificate_sha256": lane_certificate["sha256"],
        },
        "monte_carlo": {
            "site_process": "independent Bernoulli(p) per physical ququart",
            "channels": list(CHANNELS),
            "trials_per_cell": monte_carlo_trials,
            "seed": seed,
            "cell_seed": "sha256(seed|battery|channel|parameter_cell)",
            "interval": "Wilson 95 percent",
        },
        "harmonic_receiver": {
            "roles": {
                "H1_H3": "redundant full-state identification",
                "H2": "parity validation",
                "H4": "state-dark distortion reference",
            },
            "deterministic_fault_cases": [
                row["fault_case"] for row in fault_rows
            ],
            "end_to_end_trials_per_cell": harmonic_trials,
            "noise_model": "independent complex Gaussian noise per harmonic sample",
            "noise_sigmas": list(DEFAULT_NOISE_SIGMAS),
            "receiver_policy": "fail_closed",
            "telemetry_layers": [
                "receiver_rejection",
                "decoder_rejection",
                "trusted_correct_syndrome",
                "receiver_false_trust",
                "accepted_logical_residual",
            ],
        },
        "report_claim_validation": {
            "claims_schema": claims["schema"],
            "validation_schema": claim_validation["schema"],
            "validation_sha256": claim_validation["sha256"],
            "controlled_curve_language": list(PERMITTED_CURVE_LANGUAGE),
            "threshold_claim_permitted": False,
            "test_pass_claim_requires_receipt": True,
            "artifact_match_requires_full_sha256": True,
        },
        "replication": {
            "receipt_schema": replication_receipt["schema"],
            "qbraid_receipt_sha256": replication_receipt["sha256"],
            "cross_environment_status": replication_receipt["verification"][
                "deterministic_artifacts"
            ],
        },
        "hardware_claim_contract": {
            "hardware_claim": False,
            "required_declarations": list(HARDWARE_DECLARATION_FIELDS),
        },
        "error_rates": list(rates),
        "claim_scope": (
            "Exact finite code-capacity Pauli analysis plus deterministic "
            "classical harmonic-readout simulation. No circuit-level, "
            "hardware-threshold, break-even, pulse-fidelity, leakage, SPAM, "
            "or universal quantum-advantage claim."
        ),
    }
    methodology["sha256"] = canonical_sha256(methodology)
    _write_json(output_dir / "methodology.json", methodology)

    browser_payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "summary": {
            "exact_patterns": 16 ** code.n,
            "single_ququart_errors": 75,
            "decoder_table_size": 76,
            "distance": code.distance_hint,
            "small_p_scaling": "FER ≈ 10p²",
            "seed": seed,
            "claim_validation": "passed",
            "lane_symmetry": "verified",
        },
        "exact_weight_enumerator": weight_rows,
        "exact_fer_curve": exact_rows,
        "exact_channel_weight_enumerator": exact_channel_weight_rows,
        "exact_channel_fer": exact_channel_rows,
        "monte_carlo_fer": monte_rows,
        "harmonic_fault_matrix": fault_rows,
        "harmonic_end_to_end": harmonic_rows,
        "receiver_operating_curve": operating_rows,
        "evidence_facts": facts,
        "report_claims": claims,
        "claim_validation": claim_validation,
        "lane_symmetry_certificate": lane_certificate,
        "qbraid_replication_receipt": replication_receipt,
        "methodology_sha256": methodology["sha256"],
        "certificate_sha256": base_certificate["sha256"],
    }
    (output_dir / "report.js").write_text(
        "window.QEC_QUQUART_REPORT="
        + json.dumps(
            browser_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + ";\n",
        encoding="utf-8",
    )

    json_artifacts = (
        "lane_symmetry_certificate.json",
        "report_claims.json",
        "claim_validation.json",
        "qbraid_replication_receipt.json",
        "methodology.json",
        "report.js",
    )
    generated = tuple(sorted((*tables, *json_artifacts)))
    hashes = {name: _sha256(output_dir / name) for name in generated}
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "deterministic": True,
        "files": hashes,
        "methodology_sha256": methodology["sha256"],
        "claim_validation_sha256": claim_validation["sha256"],
        "lane_symmetry_sha256": lane_certificate["sha256"],
        "replication_receipt_sha256": replication_receipt["sha256"],
        "v170_0_certificate_sha256": base_certificate["sha256"],
        "seed": seed,
    }
    manifest["sha256"] = canonical_sha256(manifest)
    _write_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest
