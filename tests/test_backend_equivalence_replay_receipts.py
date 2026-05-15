from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_dependency_reduction_receipts import _chain as _base_chain
from qec.analysis.dependency_reduction_receipts import build_dependency_reduction_receipt_from_implementation
from qec.analysis.optimized_simulation_specs import build_optimized_simulation_spec_from_dependency_reduction
from qec.analysis.backend_equivalence_replay_receipts import (
    BackendReplayScenario,
    BackendReplayObservation,
    BackendReplayComparisonCase,
    BackendReplayComparisonResult,
    BackendEquivalenceReplayReceipt,
    build_backend_replay_scenario,
    build_backend_replay_observation,
    build_backend_replay_comparison_case,
    build_backend_replay_comparison_result,
    build_backend_equivalence_replay_receipt,
    validate_backend_replay_scenario,
    validate_backend_replay_observation,
    validate_backend_replay_comparison_case,
    validate_backend_replay_comparison_result,
    validate_backend_equivalence_replay_receipt,
    validate_backend_equivalence_replay_receipt_matches_inputs,
    _canonical_json,
    _normalise_hash_tuple,
    _evaluate_comparison_case,
)


def _chain():
    """Build the full chain up to OptimizedSimulationSpec."""
    c = _base_chain()
    dr = build_dependency_reduction_receipt_from_implementation(*c)
    full = (*c, dr)
    return full, build_optimized_simulation_spec_from_dependency_reduction(*full)


def _scenario(idx: int = 0, **kw):
    """Build a test scenario."""
    defaults = dict(
        scenario_index=idx,
        scenario_name=f"scenario_{idx}",
        scenario_status="REPLAY_SCENARIO_READY",
        dependency_name="qiskit",
        dependency_class="QUANTUM_BACKEND",
        optimization_scope="BACKEND_EQUIVALENCE",
        source_optimized_simulation_spec_hash="a" * 64,
        source_backend_declaration_hash="b" * 64,
        source_operation_declaration_hash="c" * 64,
        source_input_boundary_hashes=("d" * 64,),
        source_output_boundary_hashes=("e" * 64,),
        source_fallback_declaration_hash=None,
        reference_backend_declaration_hash="f" * 64,
        candidate_backend_declaration_hash="0" * 64,
        replay_requirement="REPLAY_REQUIRED",
        benchmark_requirement="BENCHMARK_NOT_ALLOWED_IN_REPLAY",
        equivalence_policy="EXACT_CANONICAL_JSON",
        scenario_input_hash="1" * 64,
        expected_output_boundary_hashes=("2" * 64,),
        reason="Test scenario",
    )
    defaults.update(kw)
    return build_backend_replay_scenario(**defaults)


def _observation(idx: int = 0, scenario_hash: str = "a" * 64, role: str = "REFERENCE_BACKEND", kind: str = "CANONICAL_JSON", **kw):
    """Build a test observation."""
    defaults = dict(
        observation_index=idx,
        source_scenario_hash=scenario_hash,
        observation_role=role,
        observation_kind=kind,
        dependency_name="qiskit",
        dependency_class="QUANTUM_BACKEND",
        optimization_scope="BACKEND_EQUIVALENCE",
        source_backend_declaration_hash="b" * 64,
        source_operation_declaration_hash="c" * 64,
        source_input_boundary_hashes=("d" * 64,),
        source_output_boundary_hashes=("e" * 64,),
        canonical_payload={"x": 1} if kind == "CANONICAL_JSON" else None,
        payload_hash="f" * 64 if kind == "HASH_ONLY" else None,
        shape=(2, 3) if kind == "STRUCTURAL_SHAPE_DTYPE" else None,
        dtype="float64" if kind == "STRUCTURAL_SHAPE_DTYPE" else None,
        ordered_sequence=(1, 2, 3) if kind == "ORDERED_SEQUENCE" else None,
        set_like_sequence=(3, 2, 1) if kind == "SET_LIKE_SEQUENCE" else None,
        unavailable_reason="unavailable" if kind == "DECLARED_UNAVAILABLE" else None,
        error_code="E001" if kind == "DECLARED_ERROR" else None,
        reason="Test observation",
    )
    defaults.update(kw)
    return build_backend_replay_observation(**defaults)


def _comparison_case(idx: int = 0, scenario_hash: str = "a" * 64, ref_hash: str = "b" * 64, cand_hash: str = "c" * 64, **kw):
    """Build a test comparison case."""
    defaults = dict(
        case_index=idx,
        source_scenario_hash=scenario_hash,
        case_name=f"case_{idx}",
        equivalence_policy="EXACT_CANONICAL_JSON",
        reference_observation_hash=ref_hash,
        candidate_observation_hash=cand_hash,
        source_optimized_simulation_spec_hash="d" * 64,
        source_backend_declaration_hashes=("e" * 64,),
        source_operation_declaration_hash="f" * 64,
        source_input_boundary_hashes=("0" * 64,),
        source_output_boundary_hashes=("1" * 64,),
        reason="Test case",
    )
    defaults.update(kw)
    return build_backend_replay_comparison_case(**defaults)


def test_backend_equivalence_replay_child_hash_stability():
    """Test that child dataclass hashes are stable across rebuilds."""
    s = _scenario()
    rebuilt = build_backend_replay_scenario(**{**s.to_dict(), "backend_replay_scenario_hash": ""})
    assert s.backend_replay_scenario_hash == rebuilt.backend_replay_scenario_hash
    assert s.to_canonical_json() == rebuilt.to_canonical_json()


def test_backend_equivalence_replay_receipt_hash_stability():
    """Test that receipt hashes are stable across rebuilds."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec1 = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    rec2 = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    assert rec1.backend_equivalence_replay_receipt_hash == rec2.backend_equivalence_replay_receipt_hash


def test_backend_equivalence_replay_from_optimized_spec_lineage():
    """Test that receipt lineage matches inputs."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    assert validate_backend_equivalence_replay_receipt_matches_inputs(
        rec, discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec
    )


def test_backend_equivalence_replay_policy_pass_and_fail():
    """Test equivalence policy evaluation for pass and fail cases."""
    o1 = _observation(0, "a" * 64, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2_pass = _observation(1, "a" * 64, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2_fail = _observation(1, "a" * 64, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 2})
    passed, code, status = _evaluate_comparison_case("EXACT_CANONICAL_JSON", o1, o2_pass)
    assert passed is True
    assert code is None
    assert status == "BACKEND_REPLAY_COMPARISON_PASSED"
    passed, code, status = _evaluate_comparison_case("EXACT_CANONICAL_JSON", o1, o2_fail)
    assert passed is False
    assert code == "CANONICAL_JSON_MISMATCH"
    assert status == "BACKEND_REPLAY_COMPARISON_FAILED"


def test_backend_replay_scenario_validation():
    """Test scenario validation rejects invalid inputs."""
    s = _scenario()
    assert validate_backend_replay_scenario(s)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_backend_replay_scenario(replace(s, backend_replay_scenario_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_backend_replay_scenario(replace(s, backend_replay_scenario_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_backend_replay_scenario(**{**s.to_dict(), "scenario_index": -1, "backend_replay_scenario_hash": ""})


def test_backend_replay_observation_validation():
    """Test observation validation rejects invalid inputs."""
    o = _observation()
    assert validate_backend_replay_observation(o)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_backend_replay_observation(replace(o, backend_replay_observation_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_backend_replay_observation(replace(o, backend_replay_observation_hash="0" * 64))
    with pytest.raises(ValueError):
        _canonical_json({"a": math.nan})


def test_backend_replay_comparison_case_validation():
    """Test comparison case validation rejects invalid inputs."""
    c = _comparison_case()
    assert validate_backend_replay_comparison_case(c)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_backend_replay_comparison_case(replace(c, backend_replay_comparison_case_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_backend_replay_comparison_case(replace(c, backend_replay_comparison_case_hash="0" * 64))


def test_backend_replay_comparison_result_re_evaluation():
    """Test that comparison results are re-evaluated during validation."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 2})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_FAILED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    assert rec.comparison_results[0].equivalence_passed is False
    assert rec.comparison_results[0].failure_code == "CANONICAL_JSON_MISMATCH"
    bad_result = replace(rec.comparison_results[0], equivalence_passed=True, failure_code=None)
    from qec.analysis.backend_equivalence_replay_receipts import _hash_payload, _base_payload
    bad_result = replace(bad_result, backend_replay_comparison_result_hash=_hash_payload(_base_payload(bad_result, "backend_replay_comparison_result_hash")))
    bad_rec = replace(rec, comparison_results=(bad_result,), first_comparison_result_hash=bad_result.backend_replay_comparison_result_hash, final_comparison_result_hash=bad_result.backend_replay_comparison_result_hash)
    with pytest.raises(ValueError, match="RESULT_EVALUATION_MISMATCH"):
        validate_backend_equivalence_replay_receipt(bad_rec)


def test_backend_equivalence_replay_lineage_mismatch_rejection():
    """Test that mismatched lineage hashes are rejected."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    from qec.analysis.backend_equivalence_replay_receipts import _hash_payload, _base_payload
    bad_rec = replace(rec, source_optimized_simulation_spec_hash="0" * 64)
    bad_rec = replace(bad_rec, backend_equivalence_replay_receipt_hash=_hash_payload(_base_payload(bad_rec, "backend_equivalence_replay_receipt_hash")))
    with pytest.raises(ValueError, match="RECEIPT_SPEC_MISMATCH"):
        validate_backend_equivalence_replay_receipt_matches_inputs(
            bad_rec, discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
            opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
            cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
            dependency_reduction_receipt=dr, optimized_simulation_spec=spec
        )


def test_backend_equivalence_replay_hash_validation():
    """Test that invalid hashes are rejected."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_backend_equivalence_replay_receipt(replace(rec, backend_equivalence_replay_receipt_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_backend_equivalence_replay_receipt(replace(rec, backend_equivalence_replay_receipt_hash="0" * 64))


def test_backend_equivalence_replay_counts_and_ordering():
    """Test that count and ordering mismatches are rejected."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    with pytest.raises(ValueError, match="COUNT_MISMATCH"):
        validate_backend_equivalence_replay_receipt(replace(rec, scenario_count=999))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_backend_replay_scenario(**{**s.to_dict(), "scenario_index": True, "backend_replay_scenario_hash": ""})


def test_backend_equivalence_replay_status_semantics():
    """Test that invalid status values are rejected."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    with pytest.raises(ValueError, match="INVALID_REPLAY_STATUS"):
        validate_backend_equivalence_replay_receipt(replace(rec, replay_status="INVALID_STATUS"))


def test_backend_equivalence_replay_source_scan_and_decoder_boundary():
    """Verify no heavy imports and no decoder layer imports."""
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "src/qec/analysis/backend_equivalence_replay_receipts.py").read_text(encoding="utf-8")
    forbidden_tokens = [
        "import numpy", "import scipy", "import pandas", "import matplotlib",
        "import qutip", "import qiskit", "import qiskit_aer", "import stim",
        "import pymatching", "import mido", "import qldpc", "import requests",
        "urllib", "subprocess", "importlib.import_module", "eval(", "exec(",
        "os.system", "shell=True", "pip", "time.time", "datetime.now", "random."
    ]
    for token in forbidden_tokens:
        assert token not in text, f"Forbidden token '{token}' found in source"
    for p in (repo / "src/qec/decoder").glob("**/*.py"):
        t = p.read_text(encoding="utf-8")
        assert "BackendEquivalenceReplayReceipt" not in t
        assert "backend_equivalence_replay_receipt_hash" not in t


def test_backend_equivalence_replay_no_scope_creep():
    """Test that receipt JSON doesn't contain scope-creep tokens."""
    chain, spec = _chain()
    m, hr, ir, eq, idx, contract, adapter, cache, fp, imp, dr = chain
    s = _scenario(source_optimized_simulation_spec_hash=spec.optimized_simulation_spec_hash)
    o1 = _observation(0, s.backend_replay_scenario_hash, "REFERENCE_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    o2 = _observation(1, s.backend_replay_scenario_hash, "OPTIMIZED_BACKEND", "CANONICAL_JSON", canonical_payload={"x": 1})
    c = _comparison_case(0, s.backend_replay_scenario_hash, o1.backend_replay_observation_hash, o2.backend_replay_observation_hash)
    rec = build_backend_equivalence_replay_receipt(
        discovery_manifest=m, hotpath_receipt=hr, invariant_receipt=ir, cross_backend_receipt=eq,
        opportunity_index=idx, optimization_contract=contract, adapter_spec=adapter,
        cached_kernel_receipt=cache, fast_path_receipt=fp, implementation_receipt=imp,
        dependency_reduction_receipt=dr, optimized_simulation_spec=spec,
        replay_mode="DECLARATIVE_BACKEND_REPLAY", replay_status="BACKEND_EQUIVALENCE_REPLAY_PASSED",
        scenarios=(s,), observations=(o1, o2), comparison_cases=(c,)
    )
    txt = rec.to_canonical_json().lower()
    for token in ["speedup", "benchmark proven", "runtime cache", "memoization", "fast path implemented", "implementation complete"]:
        assert token not in txt


def test_normalise_hash_tuple_rejects_string():
    """Test that _normalise_hash_tuple rejects plain strings."""
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _normalise_hash_tuple("abc")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _normalise_hash_tuple(b"abc")


def test_backend_replay_observation_hash_only():
    """Test observation with HASH_ONLY kind."""
    o = _observation(0, "a" * 64, "REFERENCE_BACKEND", "HASH_ONLY", payload_hash="f" * 64)
    assert validate_backend_replay_observation(o)
    rebuilt = build_backend_replay_observation(**{**o.to_dict(), "backend_replay_observation_hash": ""})
    assert o.backend_replay_observation_hash == rebuilt.backend_replay_observation_hash


def test_backend_replay_observation_structural_shape_dtype():
    """Test observation with STRUCTURAL_SHAPE_DTYPE kind."""
    o = _observation(0, "a" * 64, "REFERENCE_BACKEND", "STRUCTURAL_SHAPE_DTYPE", shape=(2, 3), dtype="float64")
    assert validate_backend_replay_observation(o)


def test_backend_replay_observation_ordered_sequence():
    """Test observation with ORDERED_SEQUENCE kind."""
    o = _observation(0, "a" * 64, "REFERENCE_BACKEND", "ORDERED_SEQUENCE", ordered_sequence=(1, 2, 3))
    assert validate_backend_replay_observation(o)


def test_backend_replay_observation_set_like_sequence():
    """Test observation with SET_LIKE_SEQUENCE kind."""
    o = _observation(0, "a" * 64, "REFERENCE_BACKEND", "SET_LIKE_SEQUENCE", set_like_sequence=(3, 2, 1))
    assert validate_backend_replay_observation(o)


def test_backend_replay_observation_declared_unavailable():
    """Test observation with DECLARED_UNAVAILABLE kind."""
    o = _observation(0, "a" * 64, "DECLARED_UNAVAILABLE_BACKEND", "DECLARED_UNAVAILABLE", unavailable_reason="not available")
    assert validate_backend_replay_observation(o)


def test_backend_replay_observation_declared_error():
    """Test observation with DECLARED_ERROR kind."""
    o = _observation(0, "a" * 64, "DECLARED_ERROR_BACKEND", "DECLARED_ERROR", error_code="E001")
    assert validate_backend_replay_observation(o)
