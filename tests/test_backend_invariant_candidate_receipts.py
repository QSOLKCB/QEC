import json
from pathlib import Path

import pytest

from qec.analysis.backend_invariant_candidate_receipts import (
    _CANDIDATE_MODE,
    _SCHEMA_VERSION,
    BackendInvariantCandidate,
    BackendInvariantEvidence,
    build_backend_invariant_candidate,
    build_backend_invariant_candidate_receipt,
    build_backend_invariant_evidence,
    derive_backend_invariant_candidates,
    validate_backend_invariant_candidate,
    validate_backend_invariant_candidate_receipt,
    validate_backend_invariant_evidence,
    validate_receipt_matches_inputs,
)
from qec.analysis.dependency_hotpath_receipts import (
    build_dependency_hotpath_candidate,
    build_dependency_import_and_hotpath_receipt,
    build_dependency_import_site,
)
from qec.analysis.heavy_dependency_discovery import (
    build_heavy_dependency_discovery_manifest,
    build_probe_result,
    get_heavy_dependency_targets,
)


def _inputs():
    statuses = {
        "numpy": "AVAILABLE",
        "scipy": "AVAILABLE",
        "pandas": "AVAILABLE",
        "matplotlib": "AVAILABLE",
        "qutip": "UNAVAILABLE",
        "qiskit": "AVAILABLE",
        "qiskit_aer": "NOT_PROBED",
        "stim": "NOT_PROBED",
        "pymatching": "NOT_PROBED",
        "mido": "AVAILABLE",
        "qldpc_internal": "INTERNAL_AVAILABLE",
        "qldpc_external": "BLOCKED_BY_POLICY",
    }
    manifest = build_heavy_dependency_discovery_manifest(
        [build_probe_result(t.dependency_name, statuses[t.dependency_name]) for t in get_heavy_dependency_targets()]
    )
    s1 = build_dependency_import_site(dependency_name="qutip", import_name="qutip", source_path="src/qec/analysis/a.py", line_number=10, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    s2 = build_dependency_import_site(dependency_name="scipy", import_name="scipy.sparse", source_path="src/qec/analysis/b.py", line_number=11, import_kind="IMPORT", import_placement="FUNCTION_BODY", imported_symbol=None, is_heavy_target=True)
    s3 = build_dependency_import_site(dependency_name="matplotlib", import_name="matplotlib", source_path="src/qec/analysis/c.py", line_number=12, import_kind="IMPORT", import_placement="FUNCTION_BODY", imported_symbol=None, is_heavy_target=True)
    s4 = build_dependency_import_site(dependency_name="pandas", import_name="pandas", source_path="src/qec/analysis/d.py", line_number=13, import_kind="IMPORT", import_placement="FUNCTION_BODY", imported_symbol=None, is_heavy_target=True)
    s5 = build_dependency_import_site(dependency_name="mido", import_name="mido", source_path="src/qec/analysis/e.py", line_number=14, import_kind="IMPORT", import_placement="FUNCTION_BODY", imported_symbol=None, is_heavy_target=True)
    s6 = build_dependency_import_site(dependency_name="qiskit", import_name="qiskit", source_path="src/qec/analysis/f.py", line_number=15, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    cands = [
        build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qutip", source_path=s1.source_path, line_number=10, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="top-level import", related_import_site_hashes=(s1.import_site_hash,)),
        build_dependency_hotpath_candidate(candidate_index=1, dependency_name="scipy", source_path=s2.source_path, line_number=11, candidate_kind="DENSE_SPARSE_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="sparse boundary", related_import_site_hashes=(s2.import_site_hash,)),
        build_dependency_hotpath_candidate(candidate_index=2, dependency_name="matplotlib", source_path=s3.source_path, line_number=12, candidate_kind="PLOTTING_RENDER_BOUNDARY", candidate_status="CANDIDATE_ONLY", reason="render boundary", related_import_site_hashes=(s3.import_site_hash,)),
        build_dependency_hotpath_candidate(candidate_index=3, dependency_name="pandas", source_path=s4.source_path, line_number=13, candidate_kind="DATAFRAME_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="df boundary", related_import_site_hashes=(s4.import_site_hash,)),
        build_dependency_hotpath_candidate(candidate_index=4, dependency_name="mido", source_path=s5.source_path, line_number=14, candidate_kind="AUDIO_MIDI_BOUNDARY", candidate_status="CANDIDATE_ONLY", reason="midi boundary", related_import_site_hashes=(s5.import_site_hash,)),
        build_dependency_hotpath_candidate(candidate_index=5, dependency_name="qiskit", source_path=s6.source_path, line_number=15, candidate_kind="REPEATED_IMPORT_REFERENCE", candidate_status="CANDIDATE_ONLY", reason="repeated ref", related_import_site_hashes=(s6.import_site_hash,)),
    ]
    hot = build_dependency_import_and_hotpath_receipt([s1, s2, s3, s4, s5, s6], cands, source_root_label="src", scanned_file_count=6)
    return manifest, hot


def test_hashes_and_canonical_stability_and_derivation():
    m, h = _inputs()
    e1 = build_backend_invariant_evidence(evidence_index=0, dependency_name="numpy", evidence_kind="DISCOVERY_STATUS_EVIDENCE", source_path=None, line_number=None, import_site_hash=None, hotpath_candidate_hash=None, probe_hash=m.probe_results[0].probe_hash, reason="AVAILABLE")
    e2 = build_backend_invariant_evidence(**{**e1.to_dict(), "evidence_hash": ""})
    assert e1.evidence_hash == e2.evidence_hash
    c1 = build_backend_invariant_candidate(candidate_index=0, dependency_name="numpy", invariant_name="n", invariant_kind="AVAILABLE_BACKEND_SURFACE_INVARIANT", invariant_status="CANDIDATE_IDENTIFIED", review_class="DISCOVERY_ONLY", evidence_hashes=(e1.evidence_hash,), source_paths=(), reason="r", required_next_receipt="CrossBackendEquivalenceReceipt")
    c2 = build_backend_invariant_candidate(**{**c1.to_dict(), "candidate_hash": ""})
    assert c1.candidate_hash == c2.candidate_hash
    r1 = derive_backend_invariant_candidates(m, h)
    r2 = derive_backend_invariant_candidates(m, h)
    assert r1.backend_invariant_candidate_receipt_hash == r2.backend_invariant_candidate_receipt_hash
    assert json.loads(e1.to_canonical_json())["evidence_hash"] == e1.evidence_hash
    assert json.loads(c1.to_canonical_json())["candidate_hash"] == c1.candidate_hash
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()
    assert validate_backend_invariant_candidate_receipt(r1)
    assert validate_receipt_matches_inputs(r1, m, h)
    assert all("confidence" not in c.to_dict() for c in r1.candidates)
    assert tuple(e.evidence_index for e in r1.evidence) == tuple(range(len(r1.evidence)))
    assert tuple(c.candidate_index for c in r1.candidates) == tuple(range(len(r1.candidates)))
    assert any(c.invariant_kind == "AVAILABLE_BACKEND_SURFACE_INVARIANT" and c.dependency_name == "numpy" for c in r1.candidates)
    assert any(c.invariant_kind == "AVAILABLE_BACKEND_SURFACE_INVARIANT" and c.dependency_name == "qldpc_internal" for c in r1.candidates)
    assert any(c.invariant_kind == "POLICY_BLOCKED_EXTERNAL_INVARIANT" and c.dependency_name == "qldpc_external" for c in r1.candidates)
    assert any(c.invariant_kind == "QUANTUM_BACKEND_BOUNDARY_INVARIANT" and c.dependency_name in {"qutip", "qiskit"} for c in r1.candidates)
    assert any(c.invariant_kind == "SPARSE_DENSE_BOUNDARY_INVARIANT" for c in r1.candidates)
    assert any(c.invariant_kind == "PLOTTING_RENDER_BOUNDARY_INVARIANT" for c in r1.candidates)
    assert any(c.invariant_kind == "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT" for c in r1.candidates)
    assert any(c.invariant_kind == "AUDIO_MIDI_BOUNDARY_INVARIANT" for c in r1.candidates)
    assert any(c.invariant_kind == "REPEATED_IMPORT_SURFACE_INVARIANT" for c in r1.candidates)
    assert any(c.invariant_kind == "TOP_LEVEL_IMPORT_BOUNDARY_INVARIANT" and c.required_next_receipt == "OptimizedQECBenchmarkReceipt" for c in r1.candidates)


def test_validation_errors_and_constraints_and_source_scan():
    m, h = _inputs(); r = derive_backend_invariant_candidates(m, h)
    dup_e = [r.evidence[0], build_backend_invariant_evidence(**{**r.evidence[1].to_dict(), "evidence_index": 0})]
    with pytest.raises(ValueError, match="EVIDENCE_ORDER_MISMATCH"):
        build_backend_invariant_candidate_receipt(m, h, dup_e, r.candidates)
    dup_c = [r.candidates[0], build_backend_invariant_candidate(**{**r.candidates[1].to_dict(), "candidate_index": 0})]
    with pytest.raises(ValueError, match="CANDIDATE_ORDER_MISMATCH"):
        build_backend_invariant_candidate_receipt(m, h, r.evidence, dup_c)
    bad_e = BackendInvariantEvidence(**{**r.evidence[0].to_dict(), "evidence_hash": "abc"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_backend_invariant_evidence(bad_e)
    bad_e2 = BackendInvariantEvidence(**{**r.evidence[0].to_dict(), "evidence_hash": "0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_backend_invariant_evidence(bad_e2)
    bad_c = BackendInvariantCandidate(**{**r.candidates[0].to_dict(), "candidate_hash": "abc"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_backend_invariant_candidate(bad_c)
    bad_c2 = BackendInvariantCandidate(**{**r.candidates[0].to_dict(), "candidate_hash": "0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_backend_invariant_candidate(bad_c2)
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"): validate_backend_invariant_candidate_receipt(type(r)(**{**r.to_dict(), "evidence": r.evidence, "candidates": r.candidates, "schema_version": "X"}))
    with pytest.raises(ValueError, match="INVALID_CANDIDATE_MODE"): validate_backend_invariant_candidate_receipt(type(r)(**{**r.to_dict(), "evidence": r.evidence, "candidates": r.candidates, "candidate_mode": "X"}))
    bad_r = type(r)(**{**r.to_dict(), "evidence": r.evidence, "candidates": r.candidates, "backend_invariant_candidate_receipt_hash": "abc"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_backend_invariant_candidate_receipt(bad_r)
    bad_r2 = type(r)(**{**r.to_dict(), "evidence": r.evidence, "candidates": r.candidates, "backend_invariant_candidate_receipt_hash": "0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_backend_invariant_candidate_receipt(bad_r2)
    assert r.schema_version == _SCHEMA_VERSION and r.candidate_mode == _CANDIDATE_MODE
    assert r.first_evidence_hash == (r.evidence[0].evidence_hash if r.evidence else "")
    assert r.final_candidate_hash == (r.candidates[-1].candidate_hash if r.candidates else "")
    assert r.candidate_count == len(r.candidates) and r.evidence_count == len(r.evidence)
    assert all(tuple(sorted(c.source_paths)) == c.source_paths for c in r.candidates)
    assert all(c.required_next_receipt in {"CrossBackendEquivalenceReceipt","OptimizedQECBenchmarkReceipt","UpstreamSourceNormalizationReceipt","OptimizationContract","NONE"} for c in r.candidates)
    assert all(c.review_class in {"DISCOVERY_ONLY","NEEDS_EQUIVALENCE_RECEIPT","NEEDS_BENCHMARK_RECEIPT","NEEDS_POLICY_NORMALIZATION","BLOCKED_BY_POLICY","SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"} for c in r.candidates)
    src = Path("src/qec/analysis/backend_invariant_candidate_receipts.py").read_text(encoding="utf-8")
    forbidden = ["import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim", "import pymatching", "import mido", "import requests", "urllib.request", "subprocess", "os.system", "shell=True", "eval(", "exec(", "__import__(", "importlib.import_module", "pip", "time.time", "datetime.now", "random."]
    for token in forbidden:
        assert token not in src
