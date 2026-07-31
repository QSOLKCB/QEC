import hashlib
import json

from qec.benchmark.ququart_battery.report import build_report


EXPECTED_NEW_ARTIFACTS = {
    "exact_channel_weight_enumerator.csv",
    "exact_channel_fer.csv",
    "receiver_operating_curve.csv",
    "lane_symmetry_certificate.json",
    "report_claims.json",
    "claim_validation.json",
    "qbraid_replication_receipt.json",
}


def test_report_hashes_every_declared_artifact(tmp_path):
    manifest = build_report(
        tmp_path,
        error_rates=("0.001", "0.01"),
        monte_carlo_trials=40,
        harmonic_trials=30,
        seed=31,
    )
    assert manifest["version"] == "170.1.1"
    assert manifest["deterministic"] is True
    assert EXPECTED_NEW_ARTIFACTS <= set(manifest["files"])
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected

    on_disk = json.loads((tmp_path / "benchmark_manifest.json").read_text())
    assert on_disk == manifest
    assert (tmp_path / "report.js").read_text().startswith(
        "window.QEC_QUQUART_REPORT="
    )

    methodology = json.loads((tmp_path / "methodology.json").read_text())
    assert "accepted_incorrect_syndrome" in (
        methodology["harmonic_receiver"]["telemetry_layers"]
    )

    validation = json.loads((tmp_path / "claim_validation.json").read_text())
    assert validation["passed"] is True
    replication = json.loads(
        (tmp_path / "qbraid_replication_receipt.json").read_text()
    )
    assert replication["verification"]["deterministic_artifacts"] == "prefix_consistent"
