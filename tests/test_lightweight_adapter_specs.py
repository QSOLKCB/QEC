import json
from pathlib import Path

import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_evidence, build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.cross_backend_equivalence_receipts import build_backend_observation, build_equivalence_receipt_from_observations
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.lightweight_adapter_specs import (
    _derive_adapter_kind, build_adapter_boundary_spec, build_adapter_capability_spec, build_adapter_operation_spec,
    build_lightweight_adapter_spec_from_contract, validate_adapter_boundary_spec, validate_adapter_capability_spec,
    validate_adapter_operation_spec, validate_adapter_spec_matches_contract, validate_lightweight_adapter_spec,
)


def _contract():
    manifest = build_default_unprobed_manifest()
    hot = [build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="src/a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=())]
    hot_receipt = build_dependency_import_and_hotpath_receipt((), hot, source_root_label="src", scanned_file_count=1, target_registry_hash=manifest.heavy_dependency_discovery_manifest_hash)
    ev = [build_backend_invariant_evidence(evidence_index=0, dependency_name="qiskit", evidence_kind="HOTPATH_CANDIDATE_EVIDENCE", source_path="src/a.py", line_number=1, import_site_hash=None, hotpath_candidate_hash=hot[0].candidate_hash, probe_hash=manifest.probe_results[1].probe_hash, reason="r")]
    cand = build_backend_invariant_candidate(candidate_index=0, dependency_name="qiskit", invariant_name="inv", invariant_kind="QUANTUM_BACKEND_BOUNDARY_INVARIANT", invariant_status="CANDIDATE_IDENTIFIED", review_class="SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW", required_next_receipt="CrossBackendEquivalenceReceipt", evidence_hashes=(ev[0].evidence_hash,), source_paths=("src/a.py",), reason="r")
    inv = build_backend_invariant_candidate_receipt(manifest, hot_receipt, ev, (cand,))
    obs = (
        build_backend_observation(observation_index=0, backend_name="ref", dependency_name="qiskit", observation_name="obs", observation_kind="JSON_VALUE", backend_role="REFERENCE", payload={"x": 1}, payload_hash="", error_code=None, unavailable_reason=None, source_invariant_candidate_hash=cand.candidate_hash),
        build_backend_observation(observation_index=1, backend_name="cand", dependency_name="qiskit", observation_name="obs", observation_kind="JSON_VALUE", backend_role="CANDIDATE", payload={"x": 1}, payload_hash="", error_code=None, unavailable_reason=None, source_invariant_candidate_hash=cand.candidate_hash),
    )
    eq = build_equivalence_receipt_from_observations(inv, obs, equivalence_policy="EXACT_CANONICAL_JSON")
    idx = derive_optimization_opportunity_index(manifest, hot_receipt, inv, eq)
    op = next(o for o in idx.opportunities if o.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT")
    return build_optimization_contract_from_opportunity(idx, op.opportunity_hash)


def test_adapter_specs_behavior_and_stability():
    c = _contract()
    b = build_adapter_boundary_spec(boundary_index=0, boundary_kind="INPUT_BOUNDARY", boundary_name="in", dependency_name="qiskit", allowed_payload_kind="CANONICAL_JSON", validation_policy="EXACT_CANONICAL_JSON", reason="r")
    o = build_adapter_operation_spec(operation_index=0, operation_kind="NORMALIZE_INPUT", operation_name="n", dependency_name="qiskit", input_boundary_hashes=(b.boundary_hash,), output_boundary_hashes=(b.boundary_hash,), required=True, reason="r")
    k = build_adapter_capability_spec(capability_index=0, capability_kind="READ_ONLY", dependency_name="qiskit", capability_name="ro", enabled=True, reason="r")
    assert b.boundary_hash == build_adapter_boundary_spec(**{**b.to_dict(), "boundary_hash": ""}).boundary_hash
    assert o.operation_hash == build_adapter_operation_spec(**{**o.to_dict(), "operation_hash": ""}).operation_hash
    assert k.capability_hash == build_adapter_capability_spec(**{**k.to_dict(), "capability_hash": ""}).capability_hash
    spec = build_lightweight_adapter_spec_from_contract(c)
    assert spec.lightweight_adapter_spec_hash == build_lightweight_adapter_spec_from_contract(c).lightweight_adapter_spec_hash
    json.dumps(b.to_dict(), allow_nan=False); json.dumps(o.to_dict(), allow_nan=False); json.dumps(k.to_dict(), allow_nan=False)
    assert spec.to_canonical_json() == build_lightweight_adapter_spec_from_contract(c).to_canonical_json()
    assert spec.to_canonical_bytes() == build_lightweight_adapter_spec_from_contract(c).to_canonical_bytes()
    assert validate_lightweight_adapter_spec(spec) is True
    assert spec.adapter_kind == "QUANTUM_BACKEND_ADAPTER"
    assert {x.boundary_kind for x in spec.boundaries}.issuperset({"INPUT_BOUNDARY", "OUTPUT_BOUNDARY", "ERROR_BOUNDARY", "POLICY_BOUNDARY"})
    assert {x.operation_kind for x in spec.operations}.issuperset({"NORMALIZE_INPUT", "CANONICALIZE_OUTPUT", "EXPORT_CANONICAL_PAYLOAD"})
    assert {x.capability_kind for x in spec.capabilities}.issuperset({"READ_ONLY", "CANONICAL_JSON_OUTPUT", "NO_BACKEND_EXECUTION", "NO_NETWORK_EXECUTION", "NO_RUNTIME_IMPORT"})


def test_validation_errors_and_contract_binding_and_scans():
    c = _contract()
    s = build_lightweight_adapter_spec_from_contract(c)
    assert s.source_optimization_contract_hash == c.optimization_contract_hash
    assert "speedup" not in s.to_canonical_json().lower() and "implementation complete" not in s.to_canonical_json().lower()
    assert _derive_adapter_kind("EXACT_JSON_EQUIVALENCE_REVIEW") == "EXACT_JSON_EQUIVALENCE_ADAPTER"
    with pytest.raises(ValueError, match="INVALID_OPTIMIZATION_SCOPE"): _derive_adapter_kind("BAD")
    with pytest.raises(ValueError, match="INVALID_INPUT"): build_lightweight_adapter_spec_from_contract(object())
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_adapter_boundary_spec(type(s.boundaries[0])(**{**s.boundaries[0].__dict__, "boundary_hash": "zzz"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_adapter_boundary_spec(type(s.boundaries[0])(**{**s.boundaries[0].__dict__, "boundary_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_adapter_operation_spec(type(s.operations[0])(**{**s.operations[0].__dict__, "operation_hash": "zzz"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_adapter_operation_spec(type(s.operations[0])(**{**s.operations[0].__dict__, "operation_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_adapter_capability_spec(type(s.capabilities[0])(**{**s.capabilities[0].__dict__, "capability_hash": "zzz"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_adapter_capability_spec(type(s.capabilities[0])(**{**s.capabilities[0].__dict__, "capability_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "lightweight_adapter_spec_hash": "zzz"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "lightweight_adapter_spec_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "schema_version": "BAD"}))
    with pytest.raises(ValueError, match="INVALID_SPEC_MODE"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "spec_mode": "BAD"}))
    with pytest.raises(ValueError, match="BOUNDARY_ORDER_MISMATCH"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "first_boundary_hash": "0" * 64}))
    with pytest.raises(ValueError, match="ADAPTER_SPEC_COUNT_MISMATCH"): validate_lightweight_adapter_spec(type(s)(**{**s.__dict__, "boundary_count": 999}))
    bad_o = build_adapter_operation_spec(operation_index=s.operations[0].operation_index, operation_kind=s.operations[0].operation_kind, operation_name=s.operations[0].operation_name, dependency_name=s.operations[0].dependency_name, input_boundary_hashes=("0" * 64,), output_boundary_hashes=s.operations[0].output_boundary_hashes, required=s.operations[0].required, reason=s.operations[0].reason)
    bad_ops = (bad_o,) + s.operations[1:]
    bad_spec = type(s)(**{**s.__dict__, "operations": bad_ops, "first_operation_hash": bad_ops[0].operation_hash, "final_operation_hash": bad_ops[-1].operation_hash})
    with pytest.raises(ValueError, match="OPERATION_BOUNDARY_MISMATCH"): validate_lightweight_adapter_spec(bad_spec)
    with pytest.raises(ValueError, match="LIGHTWEIGHT_ADAPTER_SPEC_MISMATCH"): validate_adapter_spec_matches_contract(type(s)(**{**s.__dict__, "adapter_name": "x"}), c)
    src = Path("src/qec/analysis/lightweight_adapter_specs.py").read_text(encoding="utf-8").lower()
    for token in ["import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim", "import pymatching", "import mido", "import requests", "urllib.request", "subprocess", "os.system", "shell=true", "eval(", "exec(", "__import__(", "importlib.import_module", "pip", "time.time", "datetime.now", "random."]:
        assert token not in src
    for f in Path("src/qec/decoder").glob("**/*.py"):
        assert "lightweightadapterspec" not in f.read_text(encoding="utf-8").lower()
