from dataclasses import FrozenInstanceError
import pytest

from qec.analysis.canonicalization_engine import (
    CanonicalDocument,
    CanonicalFieldSpec,
    CanonicalizationReceipt,
    CanonicalizationSchema,
    LocaleContract,
    run_canonicalization_engine,
)
from qec.analysis.extraction_boundary import ExtractedField, ExtractionResult
from qec.analysis.canonical_hashing import sha256_hex


def _field(name, typ, required=True, min_value=None, max_value=None):
    base = {"field_name": name, "field_type": typ, "required": required, "format_pattern": None, "min_value": min_value, "max_value": max_value}
    return CanonicalFieldSpec(**base, field_hash=sha256_hex(base))


def _schema(fields):
    base = {"schema_version": "1", "document_type": "invoice", "fields": tuple(fields)}
    hash_base = {"schema_version": "1", "document_type": "invoice", "fields": tuple(field.to_dict() for field in fields)}
    return CanonicalizationSchema(**base, schema_hash=sha256_hex(hash_base))


def _locale():
    base = {
        "locale_id": "en_US",
        "date_format": "%m/%d/%Y",
        "decimal_separator": ".",
        "thousands_separator": ",",
        "currency_code": "USD",
        "currency_minor_unit_exponent": 2,
        "rounding_mode": "HALF_EVEN",
    }
    return LocaleContract(**base, locale_hash=sha256_hex(base))


def _extraction(fields):
    items = tuple(ExtractedField(field_name=k, raw_value=v) for k, v in fields)
    return ExtractionResult(extracted_fields=items, extraction_hash=sha256_hex({"extracted_fields": tuple(item.to_dict() for item in items)}))


def test_valid_canonicalization_and_hash_stability():
    schema = _schema([_field("name", "STRING"), _field("count", "INTEGER"), _field("price", "DECIMAL"), _field("active", "BOOLEAN"), _field("date", "DATE"), _field("money", "MONEY"), _field("meta", "JSON")])
    rec1 = run_canonicalization_engine(_extraction([("name", " A\u00A0 B "), ("count", "1,234"), ("price", "12.3400"), ("active", "YES"), ("date", "12/31/2024"), ("money", "USD 12.345"), ("meta", "```json\n{\"a\":1,\"b\":[2]}\n```")]), schema, _locale())
    rec2 = run_canonicalization_engine(_extraction([("name", " A\u00A0 B "), ("count", "1,234"), ("price", "12.3400"), ("active", "YES"), ("date", "12/31/2024"), ("money", "USD 12.345"), ("meta", "```json\n{\"a\":1,\"b\":[2]}\n```")]), schema, _locale())
    assert rec1.status == "CANONICALIZED"
    assert rec1.canonical_hash == rec1.canonical_document.canonical_hash
    assert rec1.stable_hash == rec1.computed_stable_hash()
    assert rec1.canonical_document.canonical_json == rec2.canonical_document.canonical_json
    assert rec1.canonical_hash == rec2.canonical_hash
    assert rec1.stable_hash == rec2.stable_hash


def test_alignment_and_required_missing_and_extra():
    locale = _locale()
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("b", "1"), ("a", "x")]), _schema([_field("a", "STRING"), _field("b", "INTEGER")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("a", None)]), _schema([_field("a", "STRING", required=True)]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("a", "x"), ("b", "y")]), _schema([_field("a", "STRING")]), locale)


def test_optional_null_and_string_normalization():
    rec = run_canonicalization_engine(_extraction([("a", "  e\u0301  "), ("b", None), ("c", "   ")]), _schema([_field("a", "STRING"), _field("b", "STRING", required=False), _field("c", "STRING", required=False)]), _locale())
    assert rec.canonical_document.canonical_payload["a"] == "é"
    assert rec.canonical_document.canonical_payload["b"] is None
    assert rec.canonical_document.canonical_payload["c"] is None


def test_date_number_boolean_money_json_invalid_paths():
    locale = _locale()
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("d", "31/12/2024")]), _schema([_field("d", "DATE")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("i", True)]), _schema([_field("i", "INTEGER")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("x", "maybe")]), _schema([_field("x", "BOOLEAN")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("m", "EUR 1.00")]), _schema([_field("m", "MONEY")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("m", "USD -1.00")]), _schema([_field("m", "MONEY")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("j", '{"a":1,"a":2}')]), _schema([_field("j", "JSON")]), locale)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_canonicalization_engine(_extraction([("j", '{"a":1} {"b":2}')]), _schema([_field("j", "JSON")]), locale)


def test_json_span_and_nested_structures_and_immutability_and_scope_guard():
    rec = run_canonicalization_engine(
        _extraction([("j", "prefix {\"k\": [1, {\"z\": false}]} ")]),
        _schema([_field("j", "JSON")]),
        _locale(),
    )
    assert rec.canonical_document.canonical_payload["j"] == {"k": (1, {"z": False})}
    with pytest.raises(FrozenInstanceError):
        rec.status = "X"
    as_dict = rec.to_dict()
    assert isinstance(as_dict["canonical_document"]["canonical_payload"], dict)
    assert "RES_RAG_DIVERGENCE" not in rec.to_canonical_json()
    assert isinstance(rec.canonical_document, CanonicalDocument)
    assert isinstance(rec, CanonicalizationReceipt)
