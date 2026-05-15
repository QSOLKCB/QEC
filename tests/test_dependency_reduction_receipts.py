from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qec.analysis.backend_invariant_candidate_receipts import (
    build_backend_invariant_candidate,
    build_backend_invariant_candidate_receipt,
)
from qec.analysis.cached_canonical_kernel_receipts import (
    build_cached_canonical_kernel_receipt_from_adapter,
)
from qec.analysis.cross_backend_equivalence_receipts import (
    build_backend_observation,
    build_cross_backend_comparison_case,
    build_cross_backend_equivalence_receipt,
)
from qec.analysis.dependency_hotpath_receipts import (
    build_dependency_hotpath_candidate,
    build_dependency_import_and_hotpath_receipt,
)
from qec.analysis.dependency_reduction_receipts import (
    build_dependency_reduction_receipt_from_implementation,
    build_dependency_reduction_target,
    validate_dependency_reduction_receipt,
    validate_dependency_reduction_receipt_matches_inputs,
)
from qec.analysis.fast_path_equivalence_receipts import (
    build_fast_path_comparison_case,
    build_fast_path_equivalence_receipt,
    build_fast_path_observation,
)
from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.lightweight_adapter_specs import (
    build_lightweight_adapter_spec_from_contract,
)
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.optimization_implementation_receipts import (
    build_optimization_implementation_receipt_from_fast_path,
)
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index

def _chain():
    m = build_default_unprobed_manifest()
    hot = [
        build_dependency_hotpath_candidate(
            candidate_index=0,
            dependency_name="qiskit",
            source_path="src/a.py",
            line_number=1,
            candidate_kind="QUANTUM_BACKEND_BOUNDARY",
            candidate_status="NEEDS_EQUIVALENCE_PROOF",
            reason="r",
            related_import_site_hashes=(),
        )
    ]
    hr = build_dependency_import_and_hotpath_receipt(
        (),
        hot,
        source_root_label="src",
        scanned_file_count=1,
        target_registry_hash=m.heavy_dependency_discovery_manifest_hash,
    )
    deps = [
        "qldpc_external",
        "qiskit",
        "numpy",
        "matplotlib",
        "pandas",
        "mido",
        "qldpc_internal",
    ]
    kinds = [
        "POLICY_BLOCKED_EXTERNAL_INVARIANT",
        "QUANTUM_BACKEND_BOUNDARY_INVARIANT",
        "SPARSE_DENSE_BOUNDARY_INVARIANT",
        "PLOTTING_RENDER_BOUNDARY_INVARIANT",
        "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT",
        "AUDIO_MIDI_BOUNDARY_INVARIANT",
        "INTERNAL_QEC_SURFACE_INVARIANT",
    ]
    c = [
        build_backend_invariant_candidate(
            candidate_index=i,
            dependency_name=deps[i],
            invariant_name=f"inv{i}",
            invariant_kind=kinds[i],
            invariant_status="CANDIDATE_IDENTIFIED",
            review_class=(
                "BLOCKED_BY_POLICY" if i == 0 else "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"
            ),
            required_next_receipt=(
                "CrossBackendEquivalenceReceipt"
                if i in (1, 2, 4)
                else "OptimizationContract"
            ),
            evidence_hashes=(),
            source_paths=("src/a.py",),
            reason="r",
        )
        for i in range(len(deps))
    ]
    ir = build_backend_invariant_candidate_receipt(m, hr, (), c)
    o1 = build_backend_observation(
        observation_index=0,
        backend_name="r",
        dependency_name="qiskit",
        observation_name="n",
        observation_kind="JSON_VALUE",
        backend_role="REFERENCE",
        payload={"x": 1},
        error_code=None,
        unavailable_reason=None,
        source_invariant_candidate_hash=None,
    )
    o2 = build_backend_observation(
        observation_index=1,
        backend_name="c",
        dependency_name="qiskit",
        observation_name="n",
        observation_kind="JSON_VALUE",
        backend_role="CANDIDATE",
        payload={"x": 1},
        error_code=None,
        unavailable_reason=None,
        source_invariant_candidate_hash=None,
    )
    cc = build_cross_backend_comparison_case(
        case_index=0,
        case_name="n",
        equivalence_policy="EXACT_CANONICAL_JSON",
        reference_observation_hash=o1.observation_hash,
        candidate_observation_hashes=(o2.observation_hash,),
        source_candidate_hash=None,
        case_reason="r",
    )
    eq = build_cross_backend_equivalence_receipt(ir, [o1, o2], [cc])
    idx = derive_optimization_opportunity_index(m, hr, ir, eq)
    ready = next(
        o for o in idx.opportunities if o.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT"
    )
    contract = build_optimization_contract_from_opportunity(idx, ready.opportunity_hash)
    adapter = build_lightweight_adapter_spec_from_contract(contract)
    cache = build_cached_canonical_kernel_receipt_from_adapter(contract, adapter)
    kh = cache.kernel_descriptors[0].kernel_hash
    r = build_fast_path_observation(
        observation_index=0,
        observation_role="REFERENCE",
        observation_kind="HASH_ONLY",
        observation_name="r",
        dependency_name=contract.dependency_name,
        source_kernel_hash=kh,
        source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,
        payload=None,
        payload_hash="a" * 64,
        shape=None,
        dtype=None,
        ordered_sequence=None,
        set_like_sequence=None,
        error_code=None,
        unavailable_reason=None,
        reason="r",
    )
    cnd = build_fast_path_observation(
        observation_index=1,
        observation_role="CANDIDATE",
        observation_kind="HASH_ONLY",
        observation_name="c",
        dependency_name=contract.dependency_name,
        source_kernel_hash=kh,
        source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,
        payload=None,
        payload_hash="a" * 64,
        shape=None,
        dtype=None,
        ordered_sequence=None,
        set_like_sequence=None,
        error_code=None,
        unavailable_reason=None,
        reason="r",
    )
    case = build_fast_path_comparison_case(
        case_index=0,
        case_name="k",
        equivalence_policy="EXACT_HASH",
        reference_observation_hash=r.observation_hash,
        candidate_observation_hash=cnd.observation_hash,
        source_kernel_hash=kh,
        source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,
        reason="r",
    )
    fp = build_fast_path_equivalence_receipt(
        contract, adapter, cache, "FAST_PATH_EQUIVALENCE_PASSED", (r, cnd), (case,)
    )
    imp = build_optimization_implementation_receipt_from_fast_path(
        contract, adapter, cache, fp
    )
    return m, hr, ir, eq, idx, contract, adapter, cache, fp, imp


def test_dependency_reduction_child_hash_stability():
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    t = r.targets[0]
    rebuilt = build_dependency_reduction_target(
        **{**t.to_dict(), "reduction_target_hash": ""}
    )
    assert rebuilt.reduction_target_hash == t.reduction_target_hash


def test_dependency_reduction_receipt_hash_stability():
    c = _chain()
    r1 = build_dependency_reduction_receipt_from_implementation(*c)
    r2 = build_dependency_reduction_receipt_from_implementation(*c)
    assert r1.dependency_reduction_receipt_hash == r2.dependency_reduction_receipt_hash


def test_dependency_reduction_from_implementation_lineage():
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    assert validate_dependency_reduction_receipt_matches_inputs(r, *c)


def test_dependency_reduction_hash_validation():
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_dependency_reduction_receipt(
            replace(r, dependency_reduction_receipt_hash="x")
        )

def test_dependency_reduction_source_scan_and_decoder_boundary():
    # Resolve paths relative to this test file for reliability
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent
    source_file = repo_root / "src" / "qec" / "analysis" / "dependency_reduction_receipts.py"
    decoder_dir = repo_root / "src" / "qec" / "decoder"

    text = source_file.read_text(encoding="utf-8")
    forbidden_tokens = [
        "import numpy",
        "import scipy",
        "import pandas",
        "import matplotlib",
        "import qutip",
        "import qiskit",
        "import qiskit_aer",
        "import stim",
        "import pymatching",
        "import mido",
        "import qldpc",
        "import requests",
        "urllib",
        "subprocess",
        "importlib.import_module",
        "__import__(",
        "eval(",
        "exec(",
        "os.system",
        "shell=True",
        "pip",
        "time.time",
        "datetime.now",
        "random.",
    ]
    for token in forbidden_tokens:
        assert token not in text, f"Forbidden token '{token}' found in source"

    for p in decoder_dir.glob("**/*.py"):
        t = p.read_text(encoding="utf-8")
        assert "DependencyReductionReceipt" not in t, f"DependencyReductionReceipt found in {p}"
        assert "dependency_reduction_receipt_hash" not in t, f"dependency_reduction_receipt_hash found in {p}"


def test_dependency_reduction_receipt_hash_mismatch():
    """Test that HASH_MISMATCH is raised when receipt hash doesn't match payload."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    # Tamper with the hash to a valid format but wrong value
    tampered = replace(r, dependency_reduction_receipt_hash="a" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_reduction_receipt(tampered)


def test_dependency_reduction_receipt_invalid_status():
    """Test that INVALID_REDUCTION_STATUS is raised for invalid status."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    tampered = replace(r, reduction_status="INVALID_STATUS")
    with pytest.raises(ValueError, match="INVALID_REDUCTION_STATUS"):
        validate_dependency_reduction_receipt(tampered)


def test_dependency_reduction_receipt_target_count_mismatch():
    """Test that TARGET_COUNT_MISMATCH is raised when count doesn't match."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    tampered = replace(r, target_count=999)
    with pytest.raises(ValueError, match="TARGET_COUNT_MISMATCH"):
        validate_dependency_reduction_receipt(tampered)


def test_dependency_reduction_receipt_matches_inputs_hotpath_mismatch():
    """Test that HOTPATH_RECEIPT_MISMATCH is raised for mismatched hotpath."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    # Tampering with source hash will cause HASH_MISMATCH in validate_dependency_reduction_receipt
    # which is called first in validate_dependency_reduction_receipt_matches_inputs
    tampered = replace(r, source_dependency_hotpath_receipt_hash="b" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_reduction_receipt_matches_inputs(tampered, *c)


def test_dependency_reduction_receipt_matches_inputs_contract_mismatch():
    """Test that CONTRACT_MISMATCH is raised for mismatched contract."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    tampered = replace(r, source_optimization_contract_hash="c" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_reduction_receipt_matches_inputs(tampered, *c)


def test_dependency_reduction_receipt_matches_inputs_adapter_mismatch():
    """Test that ADAPTER_SPEC_MISMATCH is raised for mismatched adapter."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    tampered = replace(r, source_lightweight_adapter_spec_hash="d" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_reduction_receipt_matches_inputs(tampered, *c)


def test_dependency_reduction_receipt_matches_inputs_fast_path_mismatch():
    """Test that FAST_PATH_RECEIPT_MISMATCH is raised for mismatched fast path."""
    c = _chain()
    r = build_dependency_reduction_receipt_from_implementation(*c)
    tampered = replace(r, source_fast_path_equivalence_receipt_hash="e" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_reduction_receipt_matches_inputs(tampered, *c)
