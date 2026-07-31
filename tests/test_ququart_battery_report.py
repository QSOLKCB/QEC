import hashlib
import json

from qec.benchmark.ququart_battery.report import build_report


def test_report_hashes_every_declared_artifact(tmp_path):
    manifest = build_report(
        tmp_path,
        error_rates=("0.001", "0.01"),
        monte_carlo_trials=40,
        harmonic_trials=30,
        seed=31,
    )
    assert manifest["version"] == "170.1.0"
    assert manifest["deterministic"] is True
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected

    on_disk = json.loads((tmp_path / "benchmark_manifest.json").read_text())
    assert on_disk == manifest
    assert (tmp_path / "report.js").read_text().startswith(
        "window.QEC_QUQUART_REPORT="
    )
