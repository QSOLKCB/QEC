from __future__ import annotations

import pytest

from qec.analysis.deterministic_coprocessor_kernel import (
    CoProcessorDescriptor,
    build_coprocessor_receipt,
    compile_coprocessor_report,
    normalize_coprocessor_descriptor,
    run_coprocessor_kernel,
)


def _base_input(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbols": ["three", "one", "zero", "missing"],
    }
    base: dict[str, object] = {
        "descriptor_id": "desc-lookup-001",
        "operation_type": "lookup_table",
        "payload": payload,
        "lane_id": "cpu_sidecar",
        "epoch_id": "epoch-7",
        "scratchpad_size": 32,
        "schema_version": 1,
    }
    base.update(overrides)
    return base


def test_identity_pass_determinism() -> None:
    payload = {"z": [2, 1], "a": {"k": 1}}
    report_a = compile_coprocessor_report(
        _base_input(descriptor_id="id-pass-1", operation_type="identity_pass", payload=payload)
    )
    report_b = compile_coprocessor_report(
        _base_input(descriptor_id="id-pass-1", operation_type="identity_pass", payload=payload)
    )

    assert report_a.result.output_payload == report_b.result.output_payload
    assert report_a.result.output_hash == report_b.result.output_hash
    assert report_a.result.output_payload == {"a": {"k": 1}, "z": (2, 1)}


def test_lookup_table_determinism() -> None:
    report = compile_coprocessor_report(_base_input())
    assert report.result.output_payload["values"] == (3, 1, 0, -1)


def test_integer_matrix_determinism() -> None:
    report = compile_coprocessor_report(
        _base_input(
            descriptor_id="desc-mat-1",
            operation_type="integer_matrix",
            payload={
                "matrix_a": [[1, 2], [3, 4]],
                "matrix_b": [[5, 6], [7, 8]],
            },
        )
    )
    assert report.result.output_payload == {"matrix": ((19, 22), (43, 50))}


def test_fixed_point_mac_determinism() -> None:
    report = compile_coprocessor_report(
        _base_input(
            descriptor_id="desc-mac-1",
            operation_type="fixed_point_mac",
            payload={
                "vector_a": [10, 20, 30],
                "vector_b": [2, 3, 4],
                "accumulator": 5,
                "scale": 3,
            },
        )
    )
    assert report.result.output_payload["mac"] == 205
    assert report.result.output_payload["fixed_point"] == 68


def test_stable_cycle_count_and_output_hashes() -> None:
    descriptor = normalize_coprocessor_descriptor(_base_input())
    result_a = run_coprocessor_kernel(descriptor)
    result_b = run_coprocessor_kernel(descriptor)

    assert result_a.cycle_count == result_b.cycle_count
    assert result_a.output_hash == result_b.output_hash


def test_ordering_independence() -> None:
    report_a = compile_coprocessor_report(
        _base_input(descriptor_id="order-1", operation_type="identity_pass", payload={"b": [2], "a": [1]})
    )
    report_b = compile_coprocessor_report(
        _base_input(descriptor_id="order-1", operation_type="identity_pass", payload={"a": [1], "b": [2]})
    )
    assert report_a.descriptor.to_canonical_bytes() == report_b.descriptor.to_canonical_bytes()
    assert report_a.result.to_canonical_bytes() == report_b.result.to_canonical_bytes()


def test_receipt_stability() -> None:
    descriptor = normalize_coprocessor_descriptor(_base_input(descriptor_id="receipt-1"))
    result = run_coprocessor_kernel(descriptor)
    receipt_a = build_coprocessor_receipt(result)
    receipt_b = build_coprocessor_receipt(result)
    assert receipt_a == receipt_b
    assert receipt_a.receipt_hash == receipt_a.stable_hash()


def test_schema_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_coprocessor_report(_base_input(schema_version=2))


def test_unsupported_operation_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported operation_type"):
        compile_coprocessor_report(_base_input(operation_type="gpu_magic"))


def test_lane_validation() -> None:
    with pytest.raises(ValueError, match="unsupported lane_id"):
        compile_coprocessor_report(_base_input(lane_id="gpu_lane"))


def test_repeated_runs_produce_byte_identical_output() -> None:
    artifacts = tuple(compile_coprocessor_report(_base_input(descriptor_id="repeat-1")).to_canonical_bytes() for _ in range(10))
    assert len(set(artifacts)) == 1


def test_empty_descriptor_rejected() -> None:
    with pytest.raises(ValueError, match="descriptor_id must be non-empty"):
        compile_coprocessor_report(_base_input(descriptor_id=""))


def test_duplicate_descriptor_ids_rejected_when_batch_present() -> None:
    with pytest.raises(ValueError, match="duplicate descriptor IDs"):
        normalize_coprocessor_descriptor(
            {
                "descriptors": [
                    {"descriptor_id": "dup-1"},
                    {"descriptor_id": "dup-1"},
                ]
            }
        )


def test_callable_payload_rejected() -> None:
    with pytest.raises(ValueError, match="callable leakage"):
        compile_coprocessor_report(
            _base_input(operation_type="identity_pass", payload={"bad": test_callable_payload_rejected})
        )


def test_descriptor_dataclass_roundtrip() -> None:
    descriptor = normalize_coprocessor_descriptor(_base_input(descriptor_id="dc-1"))
    explicit = CoProcessorDescriptor(**descriptor.to_dict())
    assert explicit.to_canonical_bytes() == descriptor.to_canonical_bytes()
