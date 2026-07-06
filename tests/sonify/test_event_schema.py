import json
from pathlib import Path

import pytest

from qec.sonify.canonical import canonical_sha256
from qec.sonify.events import ALLOWED_EVENT_TYPES, build_symbolic_event
from qec.sonify.mapping import default_symbolic_mapping_schema, validate_symbolic_mapping_schema

FIXTURE = Path("tests/fixtures/sonify/event_schema_v167.json")
EXPECTED_EVENT_SCHEMA_HASH = "e5908ab517fbd2e3725728998277cdbcd3c74be97a0f8ce6cb33bef51d7f84d8"


def load_schema():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_event_schema_fixture_exists_and_is_stable():
    assert FIXTURE.exists()
    first = load_schema()
    second = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert canonical_sha256(first) == EXPECTED_EVENT_SCHEMA_HASH
    assert canonical_sha256(second) == EXPECTED_EVENT_SCHEMA_HASH
    assert FIXTURE.read_text(encoding="utf-8").strip() == json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_schema_fixture_declares_v167_boundaries():
    schema = load_schema()
    assert schema["schema_version"] == "v167.0"
    assert schema["schema_kind"] == "SYMBOLIC_SONIFICATION_EVENT_STREAM_SCHEMA"
    assert tuple(schema["allowed_event_types"]) == tuple(sorted(ALLOWED_EVENT_TYPES))
    assert schema["creative_status"] == "SYMBOLIC_CREATIVE_ARTIFACT"
    assert schema["claim_scope"] == "NO_SCIENTIFIC_MEDICAL_BIOLOGICAL_COSMOLOGICAL_OR_QEC_CLAIM"
    assert schema["authority_allowed"] is False
    forbidden = " ".join(schema["forbidden_scopes"]).lower()
    assert "no midi export" in forbidden
    assert "no mapping packs" in forbidden
    assert "no audio rendering" in forbidden
    assert "no llm calls" in forbidden
    assert "no network calls" in forbidden
    assert "no audio-device dependency" in forbidden


def test_default_symbolic_mapping_schema_validates_and_aligns_with_event_schema():
    schema = validate_symbolic_mapping_schema(default_symbolic_mapping_schema())
    fixture = load_schema()
    assert set(schema.required_event_fields) == set(fixture["event_required_fields"])
    assert tuple(schema.allowed_event_types) == tuple(fixture["allowed_event_types"])


def test_bad_event_type_and_empty_event_id_rejected():
    with pytest.raises(ValueError):
        build_symbolic_event("e1", "NOT_ALLOWED", "DIAG", 0, 1, "lane")
    with pytest.raises(ValueError):
        build_symbolic_event("", "SYMBOLIC_MARKER", "DIAG", 0, 1, "lane")


def test_bool_as_int_and_floats_rejected():
    with pytest.raises(TypeError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "DIAG", True, 1, "lane")
    with pytest.raises(TypeError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "DIAG", 0, False, "lane")
    with pytest.raises(TypeError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "DIAG", 0.5, 1, "lane")
    with pytest.raises(TypeError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "DIAG", 0, 1, "lane", {"x": 1.2})


def test_authority_allowed_and_forbidden_claim_phrases_rejected():
    with pytest.raises(ValueError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "DIAG", 0, 1, "lane", authority_allowed=True)
    with pytest.raises(ValueError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "E8 physics-truth", 0, 1, "lane")
    with pytest.raises(ValueError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "HPV16", 0, 1, "lane", {"note": "medical\nclaim"})
    with pytest.raises(ValueError):
        build_symbolic_event("e1", "SYMBOLIC_MARKER", "HPV16", 0, 1, "lane", tags=("decoder_correctness",))
