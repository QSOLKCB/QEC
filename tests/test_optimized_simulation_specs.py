from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_dependency_reduction_receipts import _chain
from qec.analysis.dependency_reduction_receipts import build_dependency_reduction_receipt_from_implementation
from qec.analysis.optimized_simulation_specs import (
    build_optimized_simulation_spec_from_dependency_reduction,
    build_simulation_backend_declaration,
    validate_optimized_simulation_spec,
    validate_optimized_simulation_spec_matches_inputs,
)


def _spec_chain():
    c = _chain()
    full = (*c, build_dependency_reduction_receipt_from_implementation(*c))
    return full, build_optimized_simulation_spec_from_dependency_reduction(*full)


def test_optimized_simulation_child_hash_stability():
    _, spec = _spec_chain()
    b = spec.backend_declarations[0]
    r = build_simulation_backend_declaration(**{**b.to_dict(), "simulation_backend_declaration_hash": ""})
    assert r.simulation_backend_declaration_hash == b.simulation_backend_declaration_hash
    assert b.to_canonical_json() == r.to_canonical_json()


def test_optimized_simulation_spec_hash_stability():
    c, s1 = _spec_chain(); s2 = build_optimized_simulation_spec_from_dependency_reduction(*c)
    assert s1.optimized_simulation_spec_hash == s2.optimized_simulation_spec_hash


def test_optimized_simulation_from_dependency_reduction_lineage():
    c, s = _spec_chain(); assert validate_optimized_simulation_spec_matches_inputs(s, *c)


def test_optimized_simulation_lineage_mismatch():
    """Test that mismatched lineage hashes are rejected."""
    c, s = _spec_chain()
    # Mutate the source_dependency_reduction_receipt_hash to simulate a mismatch
    bad_spec = replace(s, source_dependency_reduction_receipt_hash="0" * 64)
    # Recompute the hash so the spec itself is valid but lineage is wrong
    from qec.analysis.optimized_simulation_specs import _hash_payload, _base_payload
    bad_spec = replace(bad_spec, optimized_simulation_spec_hash=_hash_payload(_base_payload(bad_spec, "optimized_simulation_spec_hash")))
    with pytest.raises(ValueError, match="DEPENDENCY_REDUCTION_RECEIPT_MISMATCH"):
        validate_optimized_simulation_spec_matches_inputs(bad_spec, *c)


def test_optimized_simulation_hash_validation():
    _, s = _spec_chain()
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_optimized_simulation_spec(replace(s, optimized_simulation_spec_hash="x"))


def test_optimized_simulation_empty_hash_rejected():
    """Test that empty spec hash is rejected."""
    _, s = _spec_chain()
    with pytest.raises(ValueError, match="MISSING_SPEC_HASH"):
        validate_optimized_simulation_spec(replace(s, optimized_simulation_spec_hash=""))


def test_optimized_simulation_counts_and_ordering():
    _, s = _spec_chain()
    with pytest.raises(ValueError, match="INDEX_ORDER_MISMATCH"):
        bad = replace(s, backend_declarations=(replace(s.backend_declarations[0], backend_index=1),))
        validate_optimized_simulation_spec(bad)


def test_optimized_simulation_status_semantics():
    _, s = _spec_chain()
    with pytest.raises(ValueError, match="INVALID_SPEC_STATUS"):
        validate_optimized_simulation_spec(replace(s, spec_status="X"))


def test_optimized_simulation_source_scan_and_decoder_boundary():
    """Verify no heavy imports and no decoder layer imports of OptimizedSimulationSpec."""
    root = Path(__file__).parent.parent
    text = (root / "src/qec/analysis/optimized_simulation_specs.py").read_text(encoding="utf-8")
    assert "import numpy" not in text
    # Check that decoder layer does not import from optimized_simulation_specs module
    for p in (root / "src/qec/decoder").glob("**/*.py"):
        t = p.read_text(encoding="utf-8")
        assert "from qec.analysis.optimized_simulation_specs import" not in t
        assert "from qec.analysis import optimized_simulation_specs" not in t


def test_optimized_simulation_spec_has_valid_hash():
    """Test that built specs always have a valid non-empty hash."""
    _, s = _spec_chain()
    assert s.optimized_simulation_spec_hash
    assert len(s.optimized_simulation_spec_hash) == 64
    assert validate_optimized_simulation_spec(s)
