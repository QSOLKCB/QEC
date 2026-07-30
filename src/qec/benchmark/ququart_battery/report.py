"""Canonical report assembly for the v170.1.0 ququart battery."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from qec.decoder.ququart.demo import certificate
from qec.sonify.canonical import canonical_json, canonical_sha256

from .channels import CHANNELS, monte_carlo_rows
from .harmonic import (
    DEFAULT_NOISE_SIGMAS,
    harmonic_end_to_end_rows,
    harmonic_fault_rows,
)
from .oracle import DEFAULT_ERROR_RATES, exact_fer_curve, exact_weight_enumerator

SCHEMA = "qec.ququart-fer-battery.v170.1.0"
VERSION = "170.1.0"


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
    """Build deterministic CSV/JSON/browser artifacts for v170.1.0."""

    rates = tuple(error_rates)
    if not rates:
        raise ValueError("at least one physical error rate is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_rows = [dict(row) for row in exact_weight_enumerator()]
    exact_rows = [dict(row) for row in exact_fer_curve(rates)]
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

    tables: dict[str, list[dict[str, object]]] = {
        "exact_weight_enumerator.csv": weight_rows,
        "exact_fer_curve.csv": exact_rows,
        "monte_carlo_fer.csv": monte_rows,
        "harmonic_fault_matrix.csv": fault_rows,
        "harmonic_end_to_end.csv": harmonic_rows,
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    base_certificate = certificate()
    methodology: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "code": {
            "name": "packed-[[5,1,3]]_4",
            "physical_ququarts": 5,
            "logical_ququarts": 1,
            "distance": 3,
            "local_pauli_basis_size": 16,
            "nonidentity_local_errors": 15,
            "exact_patterns": 16 ** 5,
            "v170_0_certificate_sha256": base_certificate["sha256"],
        },
        "exact_oracle": {
            "method": "two-lane exhaustive packed-Pauli coset classification",
            "weight_probability": "(p/15)^w*(1-p)^(5-w)",
            "frame_error_definition": (
                "detected_uncorrectable + accepted_nonstabilizer_residual"
            ),
            "small_p_leading_order": "10*p^2 + O(p^3)",
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
            "exact_patterns": 16 ** 5,
            "single_ququart_errors": 75,
            "decoder_table_size": 76,
            "distance": 3,
            "small_p_scaling": "FER ≈ 10p²",
            "seed": seed,
        },
        "exact_weight_enumerator": weight_rows,
        "exact_fer_curve": exact_rows,
        "monte_carlo_fer": monte_rows,
        "harmonic_fault_matrix": fault_rows,
        "harmonic_end_to_end": harmonic_rows,
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

    generated = tuple(sorted((*tables, "methodology.json", "report.js")))
    hashes = {name: _sha256(output_dir / name) for name in generated}
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "deterministic": True,
        "files": hashes,
        "methodology_sha256": methodology["sha256"],
        "v170_0_certificate_sha256": base_certificate["sha256"],
        "seed": seed,
    }
    manifest["sha256"] = canonical_sha256(manifest)
    _write_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest
