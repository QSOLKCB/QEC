from __future__ import annotations

from copy import deepcopy

import pytest

from qec.analysis.experiment_dsl import (
    build_experiment_receipt,
    compile_experiment_spec,
    stable_experiment_hash,
)


def _raw_spec() -> dict[str, object]:
    return {
        "schema_version": "v137.10.1",
        "title": "  Evaluate depolarizing channel threshold  ",
        "objective": "  Confirm stabilizer success rate remains above threshold. ",
        "linked_hypothesis_ids": ["hyp-2", "hyp-1"],
        "variables": [
            {"variable_id": "p_noise", "name": "Physical Error Rate", "role": "independent"},
            {"variable_id": "code_distance", "name": "Distance", "role": "control"},
        ],
        "controls": ["decoder_schedule", "syndrome_rounds"],
        "procedure_steps": [
            {
                "step_order": 2,
                "step_id": "run",
                "action": "execute deterministic sweep",
                "variable_ids": ["p_noise", "code_distance"],
                "measurement_ids": ["logical_error_rate"],
            },
            {
                "step_order": 1,
                "step_id": "setup",
                "action": "prepare fixed code family",
                "variable_ids": ["code_distance"],
                "measurement_ids": [],
            },
        ],
        "measurements": [
            {
                "measurement_id": "logical_error_rate",
                "observable_proxy": "fraction of failed logical recovery outcomes",
                "units": "dimensionless",
                "measurement_method": "count logical failures over fixed-shot trials",
                "variable_ids": ["p_noise"],
            }
        ],
        "acceptance_criteria": [
            {
                "criterion_id": "c-1",
                "measurement_id": "logical_error_rate",
                "comparator": "le",
                "target_value": 0.05,
            }
        ],
        "tags": ["threshold", "stability"],
        "notes": "Deterministic layer-4 experiment specification.",
        "provenance": {"source": "unit-test", "owner": "analysis"},
    }


def test_canonical_outputs_hash_and_ordering_are_stable() -> None:
    raw = _raw_spec()
    artifact_a = compile_experiment_spec(raw)
    artifact_b = compile_experiment_spec(raw)

    assert artifact_a.to_canonical_json() == artifact_b.to_canonical_json()
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert stable_experiment_hash(artifact_a) == stable_experiment_hash(artifact_b)

    reordered = _raw_spec()
    reordered["linked_hypothesis_ids"] = ["hyp-1", "hyp-2"]
    reordered["variables"] = list(reversed(reordered["variables"]))
    reordered["controls"] = list(reversed(reordered["controls"]))
    reordered["measurements"] = list(reversed(reordered["measurements"]))
    reordered["acceptance_criteria"] = list(reversed(reordered["acceptance_criteria"]))

    artifact_reordered = compile_experiment_spec(reordered)
    assert artifact_a.to_canonical_json() == artifact_reordered.to_canonical_json()


def test_validation_rejects_duplicates_and_malformed_criteria() -> None:
    duplicate_variables = _raw_spec()
    duplicate_variables["variables"] = [
        {"variable_id": "p_noise", "name": "Physical Error Rate"},
        {"variable_id": "p_noise", "name": "Duplicate"},
    ]
    with pytest.raises(ValueError, match="duplicate variable IDs"):
        compile_experiment_spec(duplicate_variables)

    duplicate_measurements = _raw_spec()
    duplicate_measurements["measurements"] = [
        {
            "measurement_id": "logical_error_rate",
            "observable_proxy": "x",
            "units": "dimensionless",
            "measurement_method": "m",
            "variable_ids": ["p_noise"],
        },
        {
            "measurement_id": "logical_error_rate",
            "observable_proxy": "y",
            "units": "dimensionless",
            "measurement_method": "m2",
            "variable_ids": ["p_noise"],
        },
    ]
    with pytest.raises(ValueError, match="duplicate measurement IDs"):
        compile_experiment_spec(duplicate_measurements)

    malformed_criterion = _raw_spec()
    malformed_criterion["acceptance_criteria"] = [
        {
            "criterion_id": "c-1",
            "measurement_id": "logical_error_rate",
            "comparator": "between",
            "lower_bound": 1.0,
            "upper_bound": 0.0,
        }
    ]
    with pytest.raises(ValueError, match="malformed range bounds"):
        compile_experiment_spec(malformed_criterion)

    invalid_bounds = _raw_spec()
    invalid_bounds["acceptance_criteria"] = [
        {
            "criterion_id": "c-1",
            "measurement_id": "logical_error_rate",
            "comparator": "between",
            "lower_bound": 0.0,
            "upper_bound": float("inf"),
        }
    ]
    with pytest.raises(ValueError, match="upper_bound must be finite"):
        compile_experiment_spec(invalid_bounds)


def test_validation_rejects_missing_measurement_grounding_and_unknown_references() -> None:
    missing_proxy = _raw_spec()
    missing_proxy["measurements"] = [
        {
            "measurement_id": "logical_error_rate",
            "units": "dimensionless",
            "measurement_method": "count failures",
            "variable_ids": ["p_noise"],
        }
    ]
    with pytest.raises(ValueError, match="observable_proxy"):
        compile_experiment_spec(missing_proxy)

    missing_units = _raw_spec()
    missing_units["measurements"] = [
        {
            "measurement_id": "logical_error_rate",
            "observable_proxy": "failure ratio",
            "measurement_method": "count failures",
            "variable_ids": ["p_noise"],
        }
    ]
    with pytest.raises(ValueError, match="units"):
        compile_experiment_spec(missing_units)

    missing_method = _raw_spec()
    missing_method["measurements"] = [
        {
            "measurement_id": "logical_error_rate",
            "observable_proxy": "failure ratio",
            "units": "dimensionless",
            "variable_ids": ["p_noise"],
        }
    ]
    with pytest.raises(ValueError, match="measurement_method"):
        compile_experiment_spec(missing_method)

    unknown_reference = _raw_spec()
    unknown_reference["acceptance_criteria"] = [
        {
            "criterion_id": "c-1",
            "measurement_id": "unknown-measurement",
            "comparator": "le",
            "target_value": 0.1,
        }
    ]
    with pytest.raises(ValueError, match="unknown measurement_id"):
        compile_experiment_spec(unknown_reference)


def test_defensive_copy_and_receipt_stability() -> None:
    raw = _raw_spec()
    source = deepcopy(raw)
    artifact = compile_experiment_spec(source)
    receipt_a = build_experiment_receipt(artifact)
    receipt_b = build_experiment_receipt(artifact)

    source["title"] = "MUTATED TITLE"
    source["variables"].append({"variable_id": "injected"})  # type: ignore[index,union-attr]

    assert artifact.title == "Evaluate depolarizing channel threshold"
    assert all(variable.variable_id != "injected" for variable in artifact.variables)
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
