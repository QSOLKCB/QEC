from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from qec.benchmark.qutrit_battery.historical import (
    V3_FILE_SHA256,
    load_v3_baseline,
)
from qec.benchmark.qutrit_battery.report import build_report
from qec.sonify.canonical import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
V3_BASELINE = ROOT / "qec_data_prepared.csv"


def test_v3_baseline_is_the_immutable_tagged_artifact(tmp_path):
    payload, rows = load_v3_baseline(V3_BASELINE)
    assert hashlib.sha256(payload).hexdigest() == V3_FILE_SHA256
    assert rows[0] == {
        "error_rate": "1e-06",
        "steane": "1e-10",
        "surface": "1e-12",
        "reed_muller": "1e-09",
        "fusion_qec_photonic": "5e-07",
    }

    changed = tmp_path / "changed.csv"
    changed.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_v3_baseline(changed)


def test_report_is_byte_deterministic_and_hash_bound(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_report(
        first,
        v3_baseline_path=V3_BASELINE,
        stress_limit_per_weight=16,
    )
    second_manifest = build_report(
        second,
        v3_baseline_path=V3_BASELINE,
        stress_limit_per_weight=16,
    )
    assert first_manifest == second_manifest
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    manifest = json.loads((first / "benchmark_manifest.json").read_text())
    claimed_hash = manifest.pop("sha256")
    assert claimed_hash == canonical_sha256(manifest)
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((first / filename).read_bytes()).hexdigest() == digest
    assert (
        first / "historical_v3_baseline.csv"
    ).read_bytes() == V3_BASELINE.read_bytes()


def test_external_results_are_not_mixed_into_simulated_wide_table(tmp_path):
    output = tmp_path / "report"
    build_report(
        output,
        v3_baseline_path=V3_BASELINE,
        stress_limit_per_weight=4,
    )
    decoded_header = (
        output / "decoded_logical_error_wide.csv"
    ).read_text().splitlines()[0]
    evidence = (output / "published_evidence.csv").read_text()
    assert "google_surface_d7_2025" not in decoded_header
    assert "fusion_erasure_threshold_2023" not in decoded_header
    assert "google_surface_d7_2025" in evidence
    assert "fusion_erasure_threshold_2023" in evidence


def test_golay_is_bound_not_fabricated_exact_curve(tmp_path):
    output = tmp_path / "report"
    build_report(
        output,
        v3_baseline_path=V3_BASELINE,
        stress_limit_per_weight=4,
    )
    exact = (output / "decoded_logical_error_wide.csv").read_text()
    bounds = (output / "guaranteed_radius_tail_wide.csv").read_text()
    assert "qutrit_golay_11" not in exact.splitlines()[0]
    assert "qutrit_golay_11" in bounds.splitlines()[0]


def test_harmonic_fault_table_is_exhaustive_and_fail_closed(tmp_path):
    output = tmp_path / "report"
    build_report(
        output,
        v3_baseline_path=V3_BASELINE,
        stress_limit_per_weight=4,
    )
    rows = list(csv.DictReader(
        (output / "harmonic_fault_injection.csv").open()
    ))
    expected_counts = {
        "qutrit_cyclic_5": "40",
        "qutrit_shor_9": "72",
        "qutrit_golay_11": "3608",
    }
    for row in rows:
        assert row["errors_tested"] == expected_counts[row["code_id"]]
        assert row["false_accepts"] == "0"
        if row["fault_case"] == "clean":
            assert row["accepted"] == row["errors_tested"]
            assert row["successful"] == row["errors_tested"]
        else:
            assert row["accepted"] == "0"
            assert row["successful"] == "0"
