"""Backend-owned declarations; importing these does not run scientific code."""
from pathlib import Path

from .cli_contract import CommandSpec, ScalarOption

QUQUART_ERROR_RATES = ("0.00001", "0.00003", "0.0001", "0.0003", "0.001",
                      "0.003", "0.01", "0.03", "0.1", "0.2")
QUTRIT_STRESS_LIMIT = 2048

QUQUART_BATTERY = CommandSpec(
    "qec.ququart.benchmark", "qec.benchmark.ququart_battery.cli", "QEC ququart evidence battery",
    "Build exact channel FER oracles, corrected Monte Carlo evidence, harmonic receiver telemetry, replication receipts, and validated report claims.",
    (ScalarOption("output", default=Path("benchmarks/ququart_fer_v170_1_1"), path_role="write-directory"),
     ScalarOption("trials", "integer", 5000, minimum=1, help="Monte Carlo trials per physical-channel/error-rate cell."),
     ScalarOption("harmonic_trials", "integer", 2000, minimum=1, help="Trials per harmonic physical-rate/noise-sigma cell."),
     ScalarOption("seed", "integer", 1701001),
     ScalarOption("error_rates", default=",".join(QUQUART_ERROR_RATES), help="Comma-separated independent per-site physical error rates.")),
    "writes-artifacts", {"format": "json", "schema": "qec.ququart-fer-battery.v170.1.1", "view": "artifact-manifest", "directory_field": "output"})

QUQUART_VALIDATION = CommandSpec(
    "qec.ququart.validate", "qec.benchmark.ququart_battery.validate_cli", "Validate QEC ququart report",
    "Validate a machine-readable report-claims declaration against generated ququart FER evidence. Optionally write the validation receipt.",
    (ScalarOption("claims", required=True, path_role="read-file"),
     ScalarOption("evidence", required=True, path_role="read-directory"),
     ScalarOption("test_receipt", path_role="read-file"),
     ScalarOption("output", path_role="write-file")),
    "writes-artifacts", {"format": "json", "schema": "qec.ququart-report-claim-validation.v1", "view": "validation-receipt", "success_field": "passed"})

QUTRIT_BATTERY = CommandSpec(
    "qec.qutrit.benchmark", "qec.benchmark.qutrit_battery.cli", "QEC qutrit evidence battery",
    "Build deterministic qutrit QEC benchmark artifacts from the immutable historical v3 baseline.",
    (ScalarOption("output", default=Path("benchmarks/qutrit_decoder_v1"), path_role="write-directory"),
     ScalarOption("v3_baseline", default=Path("qec_data_prepared.csv"), path_role="read-file"),
     ScalarOption("stress_limit", "integer", QUTRIT_STRESS_LIMIT, minimum=1,
                  help="Maximum deterministic corpus patterns per code and weight.")),
    "writes-artifacts", {"format": "json", "schema": "qec.qutrit-decoder-benchmark.v1", "view": "artifact-manifest", "directory_field": "output"})

COMMANDS = (QUQUART_BATTERY, QUQUART_VALIDATION, QUTRIT_BATTERY)
