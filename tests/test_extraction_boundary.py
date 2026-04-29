from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from qec.analysis.extraction_boundary import (
    ExtractionConfigContract,
    ExtractedField,
    ExtractionInput,
    ExtractionReceipt,
    ExtractionResult,
    run_extraction_boundary,
)


def _h(ch: str) -> str:
    return ch * 64


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _build_valid() -> tuple[ExtractionInput, ExtractionConfigContract, ExtractionResult]:
    query_fields = ("invoice_id", "amount")
    config_hash = _stable_hash(
        {
            "contract_version": "v1",
            "backend_name": "oracle",
            "backend_version": "2026.04",
            "schema_hash": _h("2"),
            "locale": "en-US",
            "query_fields": query_fields,
        }
    )
    config = ExtractionConfigContract(
        contract_version="v1",
        backend_name="oracle",
        backend_version="2026.04",
        schema_hash=_h("2"),
        locale="en-US",
        query_fields=query_fields,
        config_hash=config_hash,
    )
    input_hash = _stable_hash(
        {
            "raw_bytes_hash": _h("1"),
            "source_type": "pdf",
            "extraction_config_hash": config_hash,
            "query_fields": query_fields,
        }
    )
    extraction_input = ExtractionInput(
        raw_bytes_hash=_h("1"),
        source_type="pdf",
        extraction_config_hash=config_hash,
        query_fields=query_fields,
        input_hash=input_hash,
    )
    extracted_fields = (ExtractedField("invoice_id", "INV-1"), ExtractedField("amount", "10.00"))
    result_hash = _stable_hash({"extracted_fields": tuple(field.to_dict() for field in extracted_fields)})
    result = ExtractionResult(extracted_fields=extracted_fields, extraction_hash=result_hash)
    return extraction_input, config, result


def test_valid_boundary_and_serialization_and_determinism() -> None:
    extraction_input, config, result = _build_valid()
    receipt = run_extraction_boundary(extraction_input, config, result)

    assert receipt.determinism_status == "CONSISTENT"
    assert receipt.computed_stable_hash() == receipt.stable_hash
    assert json.dumps(receipt.to_dict())
    assert receipt.to_canonical_json() == receipt.to_canonical_json()

    receipt_again = run_extraction_boundary(extraction_input, config, result)
    assert receipt.stable_hash == receipt_again.stable_hash
    assert receipt.query_fields == extraction_input.query_fields


def test_determinism_status_none_previous_is_baseline_consistent() -> None:
    extraction_input, config, result = _build_valid()
    receipt = run_extraction_boundary(extraction_input, config, result, previous_extraction_hash=None)
    assert receipt.determinism_status == "CONSISTENT"


def test_determinism_violation_detected() -> None:
    extraction_input, config, result = _build_valid()
    receipt = run_extraction_boundary(extraction_input, config, result, previous_extraction_hash=_h("4"))
    assert receipt.determinism_status == "DETERMINISM_VIOLATION"


def test_query_field_alignment_errors() -> None:
    extraction_input, config, result = _build_valid()
    mismatch_query_fields = ("amount", "invoice_id")
    mismatch_config_hash = _stable_hash(
        {
            "contract_version": config.contract_version,
            "backend_name": config.backend_name,
            "backend_version": config.backend_version,
            "schema_hash": config.schema_hash,
            "locale": config.locale,
            "query_fields": mismatch_query_fields,
        }
    )
    bad_config = ExtractionConfigContract(
        contract_version=config.contract_version,
        backend_name=config.backend_name,
        backend_version=config.backend_version,
        schema_hash=config.schema_hash,
        locale=config.locale,
        query_fields=mismatch_query_fields,
        config_hash=mismatch_config_hash,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, bad_config, result)

    wrong_order_fields = (ExtractedField("amount", "10.00"), ExtractedField("invoice_id", "INV-1"))
    wrong_order_result = ExtractionResult(
        extracted_fields=wrong_order_fields,
        extraction_hash=_stable_hash({"extracted_fields": tuple(field.to_dict() for field in wrong_order_fields)}),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, config, wrong_order_result)

    missing_fields = (ExtractedField("invoice_id", "INV-1"),)
    missing_field_result = ExtractionResult(
        extracted_fields=missing_fields,
        extraction_hash=_stable_hash({"extracted_fields": tuple(field.to_dict() for field in missing_fields)}),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, config, missing_field_result)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ExtractionInput(_h("1"), "pdf", _h("3"), ("x", "x"), _h("6"))




def test_extraction_config_hash_mismatch_rejected() -> None:
    extraction_input, config, result = _build_valid()
    bad_input_hash = _stable_hash(
        {
            "raw_bytes_hash": extraction_input.raw_bytes_hash,
            "source_type": extraction_input.source_type,
            "extraction_config_hash": _h("3"),
            "query_fields": extraction_input.query_fields,
        }
    )
    bad_input = ExtractionInput(
        raw_bytes_hash=extraction_input.raw_bytes_hash,
        source_type=extraction_input.source_type,
        extraction_config_hash=_h("3"),
        query_fields=extraction_input.query_fields,
        input_hash=bad_input_hash,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(bad_input, config, result)

def test_hash_validation_errors() -> None:
    extraction_input, config, result = _build_valid()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, config, result, previous_extraction_hash="a" * 63)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, config, result, previous_extraction_hash="A" * 64)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_extraction_boundary(extraction_input, config, result, previous_extraction_hash="g" * 64)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ExtractionResult(
            extracted_fields=result.extracted_fields,
            extraction_hash=_h("9"),
        )


def test_immutability() -> None:
    extraction_input, _, _ = _build_valid()
    with pytest.raises(FrozenInstanceError):
        extraction_input.source_type = "docx"  # type: ignore[misc]


def test_receipt_frozen() -> None:
    extraction_input, config, result = _build_valid()
    receipt = run_extraction_boundary(extraction_input, config, result)
    assert isinstance(receipt, ExtractionReceipt)
    with pytest.raises(FrozenInstanceError):
        receipt.version = "vX"  # type: ignore[misc]
