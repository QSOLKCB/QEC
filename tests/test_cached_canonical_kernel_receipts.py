import json
from pathlib import Path

import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_evidence, build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.cross_backend_equivalence_receipts import build_backend_observation, build_equivalence_receipt_from_observations
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.lightweight_adapter_specs import build_lightweight_adapter_spec_from_contract
from qec.analysis.cached_canonical_kernel_receipts import *


def _contract_and_adapter():
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
    c = build_optimization_contract_from_opportunity(idx, op.opportunity_hash)
    return c, build_lightweight_adapter_spec_from_contract(c)


def test_cached_kernel_receipt_behavior_and_determinism():
    c, s = _contract_and_adapter()
    k = build_canonical_kernel_descriptor(kernel_index=0, kernel_kind="HASH_ONLY_KERNEL", kernel_name="k", dependency_name="qiskit", source_adapter_hash=s.lightweight_adapter_spec_hash, canonical_identity_policy="P", replay_identity_fields=("a", "b"), reason="r")
    r = build_cache_eligibility_rule(rule_index=0, rule_kind="HASH_BOUND_REUSE", source_kernel_hash=k.kernel_hash, replay_safe=True, requires_equivalence_receipt=True, requires_benchmark_receipt=False, reason="r")
    i = build_cache_invalidation_rule(invalidation_index=0, invalidation_kind="INVALIDATE_ON_HASH_MISMATCH", source_kernel_hash=k.kernel_hash, trigger_condition="c", fallback_behavior="f")
    assert k.kernel_hash == build_canonical_kernel_descriptor(**{**k.to_dict(), "kernel_hash": ""}).kernel_hash
    assert r.rule_hash == build_cache_eligibility_rule(**{**r.to_dict(), "rule_hash": ""}).rule_hash
    assert i.invalidation_hash == build_cache_invalidation_rule(**{**i.to_dict(), "invalidation_hash": ""}).invalidation_hash
    rec1 = build_cached_canonical_kernel_receipt_from_adapter(c, s)
    rec2 = build_cached_canonical_kernel_receipt_from_adapter(c, s)
    assert rec1.cached_canonical_kernel_receipt_hash == rec2.cached_canonical_kernel_receipt_hash
    json.dumps(k.to_dict(), allow_nan=False); json.dumps(r.to_dict(), allow_nan=False); json.dumps(i.to_dict(), allow_nan=False)
    assert rec1.to_canonical_json() == rec2.to_canonical_json()
    assert rec1.to_canonical_bytes() == rec2.to_canonical_bytes()
    assert rec1.kernel_descriptors[0].kernel_kind == "HASH_ONLY_KERNEL"
    with pytest.raises(ValueError, match="INVALID_ADAPTER_KIND"): build_cached_canonical_kernel_receipt_from_adapter(c, type(s)(**{**s.__dict__, "adapter_kind": "BAD"}))
    assert {x.rule_kind for x in rec1.cache_rules}.issuperset({"REPLAY_SAFE_REUSE", "HASH_BOUND_REUSE", "CONTRACT_BOUND_REUSE"})
    assert {x.invalidation_kind for x in rec1.invalidation_rules}.issuperset({"INVALIDATE_ON_HASH_MISMATCH", "INVALIDATE_ON_EQUIVALENCE_FAILURE", "INVALIDATE_ON_POLICY_CHANGE", "INVALIDATE_ON_CONTRACT_CHANGE"})
    assert validate_cached_canonical_kernel_receipt(rec1) is True


def test_cached_kernel_receipt_validation_and_source_scan():
    c, s = _contract_and_adapter()
    rec = build_cached_canonical_kernel_receipt_from_adapter(c, s)
    k = rec.kernel_descriptors[0]; r = rec.cache_rules[0]; i = rec.invalidation_rules[0]
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_canonical_kernel_descriptor(CanonicalKernelDescriptor(**{**k.__dict__, "kernel_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_canonical_kernel_descriptor(CanonicalKernelDescriptor(**{**k.__dict__, "kernel_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_cache_eligibility_rule(CacheEligibilityRule(**{**r.__dict__, "rule_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_cache_eligibility_rule(CacheEligibilityRule(**{**r.__dict__, "rule_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_cache_invalidation_rule(CacheInvalidationRule(**{**i.__dict__, "invalidation_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_cache_invalidation_rule(CacheInvalidationRule(**{**i.__dict__, "invalidation_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_cached_canonical_kernel_receipt(CachedCanonicalKernelReceipt(**{**rec.__dict__, "cached_canonical_kernel_receipt_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_cached_canonical_kernel_receipt(CachedCanonicalKernelReceipt(**{**rec.__dict__, "cached_canonical_kernel_receipt_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"): validate_cached_canonical_kernel_receipt(CachedCanonicalKernelReceipt(**{**rec.__dict__, "schema_version": "BAD"}))
    with pytest.raises(ValueError, match="INVALID_CACHE_MODE"): validate_cached_canonical_kernel_receipt(CachedCanonicalKernelReceipt(**{**rec.__dict__, "cache_mode": "BAD"}))
    assert rec.kernel_descriptor_count == len(rec.kernel_descriptors) and rec.cache_rule_count == len(rec.cache_rules) and rec.invalidation_rule_count == len(rec.invalidation_rules)
    assert [x.kernel_index for x in rec.kernel_descriptors] == list(range(len(rec.kernel_descriptors)))
    assert [x.rule_index for x in rec.cache_rules] == list(range(len(rec.cache_rules)))
    assert [x.invalidation_index for x in rec.invalidation_rules] == list(range(len(rec.invalidation_rules)))
    assert rec.source_lightweight_adapter_spec_hash == s.lightweight_adapter_spec_hash
    assert rec.source_optimization_contract_hash == c.optimization_contract_hash
    text = rec.to_canonical_json().lower(); assert "runtime cache" not in text and "speedup" not in text and "implementation complete" not in text
    assert validate_cached_kernel_receipt_matches_inputs(rec, c, s) is True
    with pytest.raises(ValueError, match="KERNEL_DEPENDENCY_NAME_MISMATCH"): validate_cached_kernel_receipt_matches_inputs(CachedCanonicalKernelReceipt(**{**rec.__dict__, "dependency_name": "x"}), c, s)
    src = Path("src/qec/analysis/cached_canonical_kernel_receipts.py").read_text(encoding="utf-8").lower()
    for token in ["import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim", "import pymatching", "import mido", "import requests", "urllib.request", "subprocess", "os.system", "shell=true", "eval(", "exec(", "__import__(", "importlib.import_module", "pip", "time.time", "datetime.now", "random."]:
        assert token not in src


def test_cached_kernel_receipt_referential_integrity():
    c, s = _contract_and_adapter()
    rec = build_cached_canonical_kernel_receipt_from_adapter(c, s)
    k = rec.kernel_descriptors[0]; r = rec.cache_rules[0]; i = rec.invalidation_rules[0]
    bad_k = build_canonical_kernel_descriptor(**{**k.to_dict(), "dependency_name": "wrong_dep"})
    bad_rec = CachedCanonicalKernelReceipt(**{**rec.__dict__, "kernel_descriptors": (bad_k,), "first_kernel_hash": bad_k.kernel_hash, "final_kernel_hash": bad_k.kernel_hash})
    with pytest.raises(ValueError, match="KERNEL_DEPENDENCY_NAME_MISMATCH"): validate_cached_canonical_kernel_receipt(bad_rec)
    bad_k2 = build_canonical_kernel_descriptor(**{**k.to_dict(), "source_adapter_hash": "0" * 64})
    bad_rec2 = CachedCanonicalKernelReceipt(**{**rec.__dict__, "kernel_descriptors": (bad_k2,), "first_kernel_hash": bad_k2.kernel_hash, "final_kernel_hash": bad_k2.kernel_hash})
    with pytest.raises(ValueError, match="KERNEL_ADAPTER_HASH_MISMATCH"): validate_cached_canonical_kernel_receipt(bad_rec2)
    bad_r = build_cache_eligibility_rule(**{**r.to_dict(), "source_kernel_hash": "0" * 64})
    bad_rec3 = CachedCanonicalKernelReceipt(**{**rec.__dict__, "cache_rules": (bad_r,), "cache_rule_count": 1, "first_rule_hash": bad_r.rule_hash, "final_rule_hash": bad_r.rule_hash})
    with pytest.raises(ValueError, match="RULE_KERNEL_HASH_NOT_FOUND"): validate_cached_canonical_kernel_receipt(bad_rec3)
    bad_i = build_cache_invalidation_rule(**{**i.to_dict(), "source_kernel_hash": "0" * 64})
    bad_rec4 = CachedCanonicalKernelReceipt(**{**rec.__dict__, "invalidation_rules": (bad_i,), "invalidation_rule_count": 1, "first_invalidation_hash": bad_i.invalidation_hash, "final_invalidation_hash": bad_i.invalidation_hash})
    with pytest.raises(ValueError, match="INVALIDATION_KERNEL_HASH_NOT_FOUND"): validate_cached_canonical_kernel_receipt(bad_rec4)
    assert validate_cached_canonical_kernel_receipt(rec) is True
