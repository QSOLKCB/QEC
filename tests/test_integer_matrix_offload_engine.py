from __future__ import annotations

import pytest

from qec.analysis.integer_matrix_offload_engine import (
    compile_matrix_offload_report,
    normalize_matrix_offload_descriptor,
    run_matrix_offload_engine,
)


def _base_input(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "descriptor_id": "mat-desc-001",
        "operation_type": "matmul",
        "matrix_a": [[1, 2], [3, 4]],
        "matrix_b": [[5, 6], [7, 8]],
        "scale_factor": 3,
        "fixed_point_shift": 2,
        "saturating": False,
        "epoch_id": "epoch-1",
        "lane_id": "integer_lane_0",
        "schema_version": 1,
    }
    base.update(overrides)
    return base


def test_matmul_determinism() -> None:
    report_a = compile_matrix_offload_report(_base_input())
    report_b = compile_matrix_offload_report(_base_input())
    assert report_a.result.output_matrix == ((19, 22), (43, 50))
    assert report_a.result.output_matrix == report_b.result.output_matrix
    assert report_a.result.output_hash == report_b.result.output_hash


def test_matvec_determinism() -> None:
    report = compile_matrix_offload_report(
        _base_input(
            operation_type="matvec",
            matrix_a=[[2, 3], [4, 5]],
            matrix_b=[[7], [11]],
        )
    )
    assert report.result.output_matrix == ((47,), (83,))


def test_fixed_point_mac_determinism() -> None:
    report = compile_matrix_offload_report(
        _base_input(
            operation_type="fixed_point_mac",
            lane_id="fixed_point_lane",
            matrix_a=[[10, 20, 30], [1, 2, 3]],
            matrix_b=[[2, 3, 4], [4, 5, 6]],
            scale_factor=3,
            fixed_point_shift=2,
        )
    )
    assert report.result.output_matrix == ((150,), (24,))


def test_saturating_arithmetic_and_overflow_detection() -> None:
    report = compile_matrix_offload_report(
        _base_input(
            operation_type="accumulate",
            matrix_a=[[2**31 - 1]],
            matrix_b=[[10]],
            saturating=True,
        )
    )
    assert report.result.output_matrix == ((2**31 - 1,),)
    assert report.result.overflow_detected is True


def test_transpose_determinism() -> None:
    report = compile_matrix_offload_report(
        _base_input(
            operation_type="transpose",
            matrix_a=[[1, 2, 3], [4, 5, 6]],
            matrix_b=[[1]],
        )
    )
    assert report.result.output_matrix == ((1, 4), (2, 5), (3, 6))


def test_stable_cycle_count_and_hashes() -> None:
    descriptor = normalize_matrix_offload_descriptor(_base_input())
    result_a = run_matrix_offload_engine(descriptor)
    result_b = run_matrix_offload_engine(descriptor)
    assert result_a.cycle_count == result_b.cycle_count
    assert result_a.output_hash == result_b.output_hash


def test_repeated_runs_byte_identical() -> None:
    artifacts = tuple(compile_matrix_offload_report(_base_input(descriptor_id="repeat-1")).to_canonical_bytes() for _ in range(8))
    assert len(set(artifacts)) == 1


def test_schema_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_matrix_offload_report(_base_input(schema_version=2))


def test_malformed_dimensions_rejection() -> None:
    with pytest.raises(ValueError, match="matrix_a must be rectangular"):
        compile_matrix_offload_report(
            _base_input(
                matrix_a=[[1, 2], [3]],
            )
        )


def test_ordering_independence() -> None:
    report_a = compile_matrix_offload_report(_base_input(descriptor_id="order-1"))
    report_b = compile_matrix_offload_report(
        {
            "schema_version": 1,
            "lane_id": "integer_lane_0",
            "epoch_id": "epoch-1",
            "saturating": False,
            "fixed_point_shift": 2,
            "scale_factor": 3,
            "matrix_b": [[5, 6], [7, 8]],
            "matrix_a": [[1, 2], [3, 4]],
            "operation_type": "matmul",
            "descriptor_id": "order-1",
        }
    )
    assert report_a.descriptor.to_canonical_bytes() == report_b.descriptor.to_canonical_bytes()
    assert report_a.result.to_canonical_bytes() == report_b.result.to_canonical_bytes()


def test_empty_descriptor_rejected() -> None:
    with pytest.raises(ValueError, match="descriptor_id must be non-empty"):
        compile_matrix_offload_report(_base_input(descriptor_id=""))


def test_bad_shift_rejected() -> None:
    with pytest.raises(ValueError, match="malformed shift parameters"):
        compile_matrix_offload_report(_base_input(fixed_point_shift=-1))


def test_non_rectangular_b_rejected() -> None:
    with pytest.raises(ValueError, match="matrix_b must be rectangular"):
        compile_matrix_offload_report(_base_input(matrix_b=[[5], [6, 7]]))
