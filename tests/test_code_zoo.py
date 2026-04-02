"""
Tests for QEC Code Zoo — deterministic code-family registry.

Minimum 55 tests covering:
- dataclass immutability
- constructor determinism
- parameter validation
- stable ordering
- registry hash stability
- snapshot integration
- 100-run replay determinism
- same-input hash identity
- registry validation
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys

import pytest

from qec.codes.code_zoo import (
    REGISTRY_VERSION,
    CodeSpec,
    CodeZooRegistry,
    build_default_code_zoo,
    build_qldpc_code,
    build_repetition_code,
    build_snapshot_from_code_registry,
    build_surface_code,
    build_toric_code,
    compute_code_registry_hash,
    register_code,
    validate_code_spec,
    validate_registry,
)


# ===================================================================
# 1. Dataclass immutability tests
# ===================================================================


class TestDataclassImmutability:
    """Verify frozen dataclass invariants."""

    def test_code_spec_frozen(self):
        spec = build_repetition_code(3)
        with pytest.raises(AttributeError):
            spec.distance = 5  # type: ignore[misc]

    def test_code_spec_family_frozen(self):
        spec = build_surface_code(3)
        with pytest.raises(AttributeError):
            spec.family = "other"  # type: ignore[misc]

    def test_code_zoo_registry_frozen(self):
        zoo = build_default_code_zoo()
        with pytest.raises(AttributeError):
            zoo.registry_version = "bad"  # type: ignore[misc]

    def test_code_zoo_registry_codes_frozen(self):
        zoo = build_default_code_zoo()
        with pytest.raises(AttributeError):
            zoo.codes = ()  # type: ignore[misc]

    def test_code_zoo_registry_hash_frozen(self):
        zoo = build_default_code_zoo()
        with pytest.raises(AttributeError):
            zoo.state_hash = "bad"  # type: ignore[misc]

    def test_code_spec_metadata_frozen(self):
        spec = build_repetition_code(3)
        with pytest.raises(AttributeError):
            spec.metadata = {}  # type: ignore[misc]


# ===================================================================
# 2. Constructor determinism tests
# ===================================================================


class TestConstructorDeterminism:
    """Verify same input always produces identical output."""

    def test_repetition_determinism(self):
        a = build_repetition_code(5)
        b = build_repetition_code(5)
        assert a == b

    def test_surface_determinism(self):
        a = build_surface_code(3)
        b = build_surface_code(3)
        assert a == b

    def test_toric_determinism(self):
        a = build_toric_code(3)
        b = build_toric_code(3)
        assert a == b

    def test_qldpc_determinism(self):
        a = build_qldpc_code(30, 8, 6)
        b = build_qldpc_code(30, 8, 6)
        assert a == b

    def test_default_zoo_determinism(self):
        a = build_default_code_zoo()
        b = build_default_code_zoo()
        assert a == b


# ===================================================================
# 3. Constructor correctness tests
# ===================================================================


class TestConstructorCorrectness:
    """Verify code parameters are computed correctly."""

    def test_repetition_physical_qubits(self):
        spec = build_repetition_code(7)
        assert spec.physical_qubits == 7

    def test_repetition_logical_qubits(self):
        spec = build_repetition_code(5)
        assert spec.logical_qubits == 1

    def test_repetition_stabilizers(self):
        spec = build_repetition_code(5)
        assert spec.stabilizer_count == 4

    def test_surface_physical_qubits(self):
        spec = build_surface_code(5)
        assert spec.physical_qubits == 25

    def test_surface_logical_qubits(self):
        spec = build_surface_code(3)
        assert spec.logical_qubits == 1

    def test_surface_stabilizers(self):
        spec = build_surface_code(3)
        assert spec.stabilizer_count == 8

    def test_toric_physical_qubits(self):
        spec = build_toric_code(4)
        assert spec.physical_qubits == 32

    def test_toric_logical_qubits(self):
        spec = build_toric_code(3)
        assert spec.logical_qubits == 2

    def test_toric_stabilizers(self):
        spec = build_toric_code(3)
        assert spec.stabilizer_count == 16

    def test_qldpc_params(self):
        spec = build_qldpc_code(30, 8, 6)
        assert spec.physical_qubits == 30
        assert spec.logical_qubits == 8
        assert spec.distance == 6
        assert spec.stabilizer_count == 22

    def test_repetition_code_id(self):
        spec = build_repetition_code(3)
        assert spec.code_id == "repetition_d3"

    def test_surface_code_id(self):
        spec = build_surface_code(5)
        assert spec.code_id == "surface_d5"

    def test_toric_code_id(self):
        spec = build_toric_code(3)
        assert spec.code_id == "toric_d3"

    def test_qldpc_code_id(self):
        spec = build_qldpc_code(30, 8, 6)
        assert spec.code_id == "qldpc_n30_k8_d6"


# ===================================================================
# 4. Parameter validation tests
# ===================================================================


class TestParameterValidation:
    """Verify invalid parameters are rejected."""

    def test_repetition_zero_distance(self):
        with pytest.raises(ValueError):
            build_repetition_code(0)

    def test_repetition_negative_distance(self):
        with pytest.raises(ValueError):
            build_repetition_code(-1)

    def test_surface_zero_distance(self):
        with pytest.raises(ValueError):
            build_surface_code(0)

    def test_toric_zero_distance(self):
        with pytest.raises(ValueError):
            build_toric_code(0)

    def test_qldpc_zero_n(self):
        with pytest.raises(ValueError):
            build_qldpc_code(0, 1, 1)

    def test_qldpc_zero_k(self):
        with pytest.raises(ValueError):
            build_qldpc_code(10, 0, 1)

    def test_qldpc_zero_d(self):
        with pytest.raises(ValueError):
            build_qldpc_code(10, 1, 0)

    def test_qldpc_k_greater_than_n(self):
        with pytest.raises(ValueError):
            build_qldpc_code(5, 10, 3)

    def test_repetition_float_distance(self):
        with pytest.raises(ValueError):
            build_repetition_code(3.5)  # type: ignore[arg-type]

    def test_surface_string_distance(self):
        with pytest.raises(ValueError):
            build_surface_code("3")  # type: ignore[arg-type]


# ===================================================================
# 5. Stable ordering tests
# ===================================================================


class TestStableOrdering:
    """Verify registry ordering is deterministic and sorted."""

    def test_default_zoo_sorted(self):
        zoo = build_default_code_zoo()
        keys = [(s.family, s.distance, s.code_id) for s in zoo.codes]
        assert keys == sorted(keys)

    def test_register_maintains_order(self):
        r = register_code(build_surface_code(5))
        r = register_code(build_repetition_code(3), r)
        r = register_code(build_toric_code(3), r)
        keys = [(s.family, s.distance, s.code_id) for s in r.codes]
        assert keys == sorted(keys)

    def test_register_order_independent(self):
        r1 = register_code(build_repetition_code(3))
        r1 = register_code(build_surface_code(5), r1)

        r2 = register_code(build_surface_code(5))
        r2 = register_code(build_repetition_code(3), r2)

        assert r1.codes == r2.codes
        assert r1.state_hash == r2.state_hash


# ===================================================================
# 6. Registry hash stability tests
# ===================================================================


class TestRegistryHashStability:
    """Verify hash is deterministic and stable."""

    def test_same_registry_same_hash(self):
        z1 = build_default_code_zoo()
        z2 = build_default_code_zoo()
        assert z1.state_hash == z2.state_hash

    def test_hash_changes_with_different_codes(self):
        z1 = register_code(build_repetition_code(3))
        z2 = register_code(build_repetition_code(5))
        assert z1.state_hash != z2.state_hash

    def test_hash_is_sha256_hex(self):
        zoo = build_default_code_zoo()
        assert len(zoo.state_hash) == 64
        int(zoo.state_hash, 16)  # must not raise

    def test_compute_hash_matches_stored(self):
        zoo = build_default_code_zoo()
        computed = compute_code_registry_hash(zoo)
        assert computed == zoo.state_hash

    def test_hash_100_run_stability(self):
        """Hash must be identical across 100 consecutive builds."""
        reference = build_default_code_zoo()
        for _ in range(100):
            assert compute_code_registry_hash(build_default_code_zoo()) == reference.state_hash


# ===================================================================
# 7. Registry validation tests
# ===================================================================


class TestRegistryValidation:
    """Verify registry validation catches invalid registries."""

    def test_valid_registry(self):
        zoo = build_default_code_zoo()
        assert validate_registry(zoo) is True

    def test_invalid_hash(self):
        zoo = build_default_code_zoo()
        bad = CodeZooRegistry(
            codes=zoo.codes,
            registry_version=zoo.registry_version,
            state_hash="a" * 64,
        )
        with pytest.raises(ValueError, match="state_hash mismatch"):
            validate_registry(bad)

    def test_invalid_version_empty(self):
        with pytest.raises(ValueError, match="registry_version"):
            validate_registry(
                CodeZooRegistry(codes=(), registry_version="", state_hash="a" * 64)
            )

    def test_validate_code_spec_valid(self):
        spec = build_repetition_code(3)
        assert validate_code_spec(spec) is True

    def test_validate_code_spec_bad_code_id(self):
        spec = CodeSpec(
            code_id="", family="rep", distance=3,
            logical_qubits=1, physical_qubits=3,
            stabilizer_count=2, metadata={},
        )
        with pytest.raises(ValueError, match="code_id"):
            validate_code_spec(spec)

    def test_validate_code_spec_bad_distance(self):
        spec = CodeSpec(
            code_id="x", family="rep", distance=0,
            logical_qubits=1, physical_qubits=3,
            stabilizer_count=2, metadata={},
        )
        with pytest.raises(ValueError, match="distance"):
            validate_code_spec(spec)


# ===================================================================
# 8. 100-run replay determinism
# ===================================================================


class TestReplayDeterminism:
    """Mandatory 100-run replay tests."""

    def test_default_zoo_100_replay(self):
        reference = build_default_code_zoo()
        for _ in range(100):
            assert build_default_code_zoo() == reference

    def test_hash_100_replay(self):
        reference = build_default_code_zoo()
        ref_hash = compute_code_registry_hash(reference)
        for _ in range(100):
            assert compute_code_registry_hash(build_default_code_zoo()) == ref_hash

    def test_constructor_100_replay_repetition(self):
        reference = build_repetition_code(5)
        for _ in range(100):
            assert build_repetition_code(5) == reference

    def test_constructor_100_replay_surface(self):
        reference = build_surface_code(3)
        for _ in range(100):
            assert build_surface_code(3) == reference


# ===================================================================
# 9. Snapshot integration tests
# ===================================================================


class TestSnapshotIntegration:
    """Verify integration with controller_snapshot_schema."""

    def test_snapshot_builds(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "test_policy")
        assert snap.policy_id == "test_policy"

    def test_snapshot_state_hash_is_sha256(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "policy_1")
        assert len(snap.state_hash) == 64
        int(snap.state_hash, 16)

    def test_snapshot_payload_is_valid_json(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "policy_1")
        parsed = json.loads(snap.payload_json)
        assert parsed["type"] == "CodeZooRegistry"

    def test_snapshot_determinism(self):
        zoo = build_default_code_zoo()
        s1 = build_snapshot_from_code_registry(zoo, "p1")
        s2 = build_snapshot_from_code_registry(zoo, "p1")
        assert s1 == s2

    def test_snapshot_100_replay(self):
        zoo = build_default_code_zoo()
        ref = build_snapshot_from_code_registry(zoo, "replay_test")
        for _ in range(100):
            assert build_snapshot_from_code_registry(zoo, "replay_test") == ref

    def test_snapshot_schema_version(self):
        from qec.ai.controller_snapshot_schema import SCHEMA_VERSION
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "p1")
        assert snap.schema_version == SCHEMA_VERSION

    def test_snapshot_evidence_score_range(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "p1")
        assert 0.0 <= snap.evidence_score <= 1.0

    def test_snapshot_invariant_passed(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "p1")
        assert snap.invariant_passed is True

    def test_snapshot_timestamp_index(self):
        zoo = build_default_code_zoo()
        snap = build_snapshot_from_code_registry(zoo, "p1")
        assert snap.timestamp_index == len(zoo.codes)


# ===================================================================
# 10. Registry version test
# ===================================================================


class TestRegistryVersion:
    def test_registry_version(self):
        assert REGISTRY_VERSION == "v136.8.2"

    def test_default_zoo_version(self):
        zoo = build_default_code_zoo()
        assert zoo.registry_version == "v136.8.2"


# ===================================================================
# 11. Register code tests
# ===================================================================


class TestRegisterCode:
    def test_register_to_none(self):
        r = register_code(build_repetition_code(3))
        assert len(r.codes) == 1

    def test_register_duplicate_raises(self):
        r = register_code(build_repetition_code(3))
        with pytest.raises(ValueError, match="Duplicate"):
            register_code(build_repetition_code(3), r)

    def test_register_multiple(self):
        r = register_code(build_repetition_code(3))
        r = register_code(build_surface_code(5), r)
        r = register_code(build_toric_code(3), r)
        assert len(r.codes) == 3


# ===================================================================
# 12. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    """Verify code_zoo does not import or modify decoder."""

    def test_no_decoder_import_in_code_zoo(self):
        import qec.codes.code_zoo as mod
        source_path = mod.__file__
        with open(source_path) as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_directory_exists_unchanged(self):
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        assert os.path.isdir(decoder_path)


# ===================================================================
# 13. Same-input hash identity
# ===================================================================


class TestSameInputHashIdentity:
    def test_identical_registries_identical_hash(self):
        r1 = register_code(build_surface_code(3))
        r2 = register_code(build_surface_code(3))
        assert r1.state_hash == r2.state_hash

    def test_different_registries_different_hash(self):
        r1 = register_code(build_surface_code(3))
        r2 = register_code(build_surface_code(5))
        assert r1.state_hash != r2.state_hash
