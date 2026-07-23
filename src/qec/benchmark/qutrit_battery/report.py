"""Canonical CSV and JSON report assembly."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from qec.sonify.canonical import canonical_json, canonical_sha256

from .codes import benchmark_models
from .curves import (
    DEFAULT_ERROR_RATES,
    bound_codes,
    decoded_curve_rows,
    radius_bound_rows,
)
from .historical import (
    lineage_record,
    load_v3_baseline,
    numeric_delta_rows,
    overlay_rows,
)
from .harmonic import harmonic_fault_rows
from .sources import published_evidence_rows, research_watch_rows
from .stress import CORPUS_LIMIT_PER_WEIGHT, stress_rows

SCHEMA = "qec.qutrit-decoder-benchmark.v1"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty benchmark table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _wide_rows(
    rows: list[dict[str, object]],
    *,
    value_field: str,
    error_rates: Iterable[str],
) -> list[dict[str, object]]:
    code_ids = tuple(dict.fromkeys(str(row["code_id"]) for row in rows))
    lookup = {
        (str(row["error_rate"]), str(row["code_id"])): row[value_field]
        for row in rows
    }
    return [
        {
            "error_rate": error_rate,
            **{
                code_id: lookup.get((error_rate, code_id), "")
                for code_id in code_ids
            },
        }
        for error_rate in error_rates
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def build_report(
    output_dir: Path,
    *,
    v3_baseline_path: Path,
    error_rates: tuple[str, ...] = DEFAULT_ERROR_RATES,
    stress_limit_per_weight: int = CORPUS_LIMIT_PER_WEIGHT,
) -> dict[str, object]:
    """Build a byte-reproducible benchmark report directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = benchmark_models()
    decoded = decoded_curve_rows(models, error_rates)
    bounds = radius_bound_rows(bound_codes(models), error_rates)
    stress = stress_rows(models, limit_per_weight=stress_limit_per_weight)
    v3_bytes, v3_rows = load_v3_baseline(v3_baseline_path)
    overlays = overlay_rows(v3_rows, decoded, bounds)

    tables: dict[str, list[dict[str, object]]] = {
        "decoded_logical_error_long.csv": decoded,
        "decoded_logical_error_wide.csv": _wide_rows(
            decoded,
            value_field="logical_failure_rate",
            error_rates=error_rates,
        ),
        "guaranteed_radius_tail_long.csv": bounds,
        "guaranteed_radius_tail_wide.csv": _wide_rows(
            bounds,
            value_field="uncorrectable_weight_tail",
            error_rates=error_rates,
        ),
        "deterministic_stress_corpus.csv": stress,
        "harmonic_fault_injection.csv": list(harmonic_fault_rows()),
        "v3_overlay.csv": overlays,
        "v3_numeric_deltas.csv": numeric_delta_rows(overlays),
        "published_evidence.csv": published_evidence_rows(),
        "research_watch.csv": research_watch_rows(),
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)
    (output_dir / "historical_v3_baseline.csv").write_bytes(v3_bytes)

    model_records = [
        {
            "code_id": model.code_id,
            "label": model.label,
            "origin": model.origin,
            "alphabet_dimension": model.modulus,
            "n": model.n,
            "k": model.k,
            "distance": model.distance,
            "decoder_radius": model.radius,
            "decoder_table_size": model.decoder_table_size,
            "stabilizer_group_size": model.stabilizer_group_size,
            "check_matrix_sha256": canonical_sha256(model.checks),
            "source_url": model.source_url,
        }
        for model in models
    ]
    methodology: dict[str, object] = {
        "schema": SCHEMA,
        "error_rates": list(error_rates),
        "models": model_records,
        "noise_model": {
            "name": "iid_depolarizing_code_capacity_per_site",
            "identity_probability": "1-p",
            "nonidentity_probability": "p/(q^2-1)",
            "syndrome_measurement": "perfect",
        },
        "exact_curve": {
            "method": "successful_decoder_coset_weight_enumerator",
            "formula": (
                "P_fail=1-sum_w A_w*(p/(q^2-1))^w*(1-p)^(n-w)"
            ),
            "operation_limit": 1_000_000,
            "omission_policy": (
                "codes above the exact operation limit appear in the "
                "rigorous radius bound and stress corpus, not an estimate"
            ),
        },
        "radius_bound": {
            "formula": "P(W>t)=1-sum_{w=0}^t C(n,w)p^w(1-p)^(n-w)",
            "role": "rigorous logical-failure upper bound",
        },
        "stress_corpus": {
            "limit_per_weight": stress_limit_per_weight,
            "selection": "exact_or_evenly_spaced_ordinal_v1",
            "statistical_claim": "none",
        },
        "v3_lineage": lineage_record(),
        "claim_scope": (
            "Finite code-capacity simulation and algebraic bounds only. "
            "No circuit-level, threshold, hardware, or universal QEC "
            "advantage claim."
        ),
    }
    methodology["sha256"] = canonical_sha256(methodology)
    _write_json(output_dir / "methodology.json", methodology)

    generated = tuple(sorted((
        *tables,
        "historical_v3_baseline.csv",
        "methodology.json",
    )))
    hashes = {
        name: _sha256(output_dir / name)
        for name in generated
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "files": hashes,
        "methodology_sha256": methodology["sha256"],
        "historical_v3_sha256": lineage_record()["file_sha256"],
        "deterministic": True,
    }
    manifest["sha256"] = canonical_sha256(manifest)
    _write_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest
