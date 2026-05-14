from __future__ import annotations

import json
from pathlib import Path

import pytest

from qec.analysis.dependency_hotpath_receipts import (
    build_dependency_hotpath_candidate,
    build_dependency_import_and_hotpath_receipt,
    build_dependency_import_site,
    scan_dependency_imports,
    validate_dependency_hotpath_candidate,
    validate_dependency_import_and_hotpath_receipt,
    validate_dependency_import_site,
    validate_receipt_matches_scan,
)


def _write(tmp_path: Path, rel: str, content: str) -> None:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_hash_determinism_builders_and_receipt():
    s1 = build_dependency_import_site(dependency_name="numpy", import_name="numpy", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    s2 = build_dependency_import_site(dependency_name="numpy", import_name="numpy", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    assert s1.import_site_hash == s2.import_site_hash
    c1 = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="numpy", source_path="a.py", line_number=1, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="x", related_import_site_hashes=(s1.import_site_hash,))
    c2 = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="numpy", source_path="a.py", line_number=1, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="x", related_import_site_hashes=(s1.import_site_hash,))
    assert c1.candidate_hash == c2.candidate_hash
    r1 = build_dependency_import_and_hotpath_receipt((s1,), (c1,), source_root_label="src", scanned_file_count=1)
    r2 = build_dependency_import_and_hotpath_receipt((s2,), (c2,), source_root_label="src", scanned_file_count=1)
    assert r1.dependency_hotpath_receipt_hash == r2.dependency_hotpath_receipt_hash


def test_json_and_canonical_exports():
    s = build_dependency_import_site(dependency_name="numpy", import_name="numpy", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    c = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="numpy", source_path="a.py", line_number=1, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="x", related_import_site_hashes=(s.import_site_hash,))
    r = build_dependency_import_and_hotpath_receipt((s,), (c,), source_root_label="src", scanned_file_count=1)
    json.dumps(s.to_dict())
    json.dumps(c.to_dict())
    assert r.to_canonical_json() == r.to_canonical_json()
    assert r.to_canonical_bytes() == r.to_canonical_bytes()


def test_scan_and_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write(tmp_path, "src/pkg/a.py", "import numpy\nfrom scipy import sparse\nimport matplotlib.pyplot as plt\nimport qutip\nimport qiskit_aer\nimport pandas\nimport mido\n")
    _write(tmp_path, "src/pkg/b.py", "from scipy import linalg\n")
    _write(tmp_path, "src/pkg/c.py", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n import numpy\ndef f():\n import numpy\n")
    _write(tmp_path, "src/pkg/link_target.py", "import numpy\n")
    (tmp_path / "src/pkg/link.py").symlink_to(tmp_path / "src/pkg/link_target.py")

    import importlib
    monkeypatch.setattr(importlib, "import_module", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not import modules")))

    receipt = scan_dependency_imports(tmp_path)
    dep_names = [s.dependency_name for s in receipt.import_sites]
    assert "numpy" in dep_names and "scipy" in dep_names and "matplotlib" in dep_names and "qutip" in dep_names and "qiskit_aer" in dep_names and "pandas" in dep_names and "mido" in dep_names
    placements = {s.import_placement for s in receipt.import_sites if s.dependency_name == "numpy"}
    assert "TYPE_CHECKING_BLOCK" in placements and "FUNCTION_BODY" in placements and "MODULE_TOP_LEVEL" in placements
    assert all("/" in s.source_path or s.source_path.endswith(".py") for s in receipt.import_sites)
    kinds = {c.candidate_kind: c for c in receipt.hotpath_candidates}
    assert "REPEATED_IMPORT_REFERENCE" in kinds
    assert "MODULE_TOP_LEVEL_HEAVY_IMPORT" in kinds
    assert kinds["QUANTUM_BACKEND_BOUNDARY"].candidate_status == "NEEDS_EQUIVALENCE_PROOF"
    assert "PLOTTING_RENDER_BOUNDARY" in kinds
    assert "DATAFRAME_BOUNDARY" in kinds
    assert "DENSE_SPARSE_BOUNDARY" in kinds


def test_validation_failures_and_counts(tmp_path: Path):
    s = build_dependency_import_site(dependency_name="numpy", import_name="numpy", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_dependency_import_site(type(s)(**{**s.to_dict(), "import_site_hash": "abc"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_dependency_import_site(type(s)(**{**s.to_dict(), "import_site_hash": "0" * 64}))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_dependency_import_site(dependency_name="numpy", import_name="numpy", source_path="a.py", line_number=True, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    c = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="numpy", source_path="a.py", line_number=1, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="x", related_import_site_hashes=(s.import_site_hash,))
    c_bad = build_dependency_hotpath_candidate(candidate_index=2, dependency_name="numpy", source_path="a.py", line_number=1, candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT", candidate_status="NEEDS_BENCHMARK_RECEIPT", reason="x", related_import_site_hashes=(s.import_site_hash,))
    with pytest.raises(ValueError, match="CANDIDATE_ORDER_MISMATCH"):
        build_dependency_import_and_hotpath_receipt((s,), (c_bad,), source_root_label="src", scanned_file_count=1)
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE"):
        build_dependency_import_and_hotpath_receipt((s,), (c, c), source_root_label="src", scanned_file_count=1)
    r = build_dependency_import_and_hotpath_receipt((s,), (c,), source_root_label="src", scanned_file_count=1)
    assert r.import_site_count == 1 and r.hotpath_candidate_count == 1
    assert validate_dependency_import_and_hotpath_receipt(r)
    r_bad = type(r)(**{**r.to_dict(), "first_import_site_hash": "f" * 64})
    with pytest.raises(ValueError):
        validate_dependency_import_and_hotpath_receipt(r_bad)

    _write(tmp_path, "src/x.py", "import numpy\n")
    scan = scan_dependency_imports(tmp_path)
    assert validate_receipt_matches_scan(scan, tmp_path)
    _write(tmp_path, "src/x.py", "import scipy\n")
    with pytest.raises(ValueError, match="HOTPATH_RECEIPT_MISMATCH"):
        validate_receipt_matches_scan(scan, tmp_path)


def test_source_scan_guard_tokens():
    text = Path("src/qec/analysis/dependency_hotpath_receipts.py").read_text(encoding="utf-8")
    forbidden = ["import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim", "import pymatching", "import mido", "import requests", "urllib.request", "subprocess", "os.system", "shell=True", "eval(", "exec(", "__import__(", "importlib.import_module", "pip", "time.time", "datetime.now", "random."]
    for token in forbidden:
        assert token not in text


def test_qldpc_internal_mapping_preserved(tmp_path: Path):
    """P1 fix: Ensure qldpc.css_code imports are mapped to qldpc_internal, not qldpc_external."""
    _write(tmp_path, "src/pkg/a.py", "from qldpc.css_code import CSSCode\n")
    _write(tmp_path, "src/pkg/b.py", "import qldpc\n")
    receipt = scan_dependency_imports(tmp_path)
    # qldpc.css_code should map to qldpc_internal
    qldpc_css_sites = [s for s in receipt.import_sites if s.import_name == "qldpc.css_code"]
    assert len(qldpc_css_sites) == 1
    assert qldpc_css_sites[0].dependency_name == "qldpc_internal"
    assert qldpc_css_sites[0].is_heavy_target is True
    # import qldpc should map to qldpc_external (top-level import)
    qldpc_top_sites = [s for s in receipt.import_sites if s.import_name == "qldpc"]
    assert len(qldpc_top_sites) == 1
    assert qldpc_top_sites[0].dependency_name == "qldpc_external"
    assert qldpc_top_sites[0].is_heavy_target is True
    # Check that INTERNAL_QEC_BOUNDARY candidate is generated
    kinds = {c.candidate_kind for c in receipt.hotpath_candidates}
    assert "INTERNAL_QEC_BOUNDARY" in kinds


def test_schema_and_scan_mode_validation():
    """P2 fix: Ensure schema_version and scan_mode are validated."""
    from qec.analysis.dependency_hotpath_receipts import (
        DependencyImportAndHotPathReceipt,
        _SCHEMA_VERSION,
        _SCAN_MODE,
    )
    s = build_dependency_import_site(
        dependency_name="numpy",
        import_name="numpy",
        source_path="a.py",
        line_number=1,
        import_kind="IMPORT",
        import_placement="MODULE_TOP_LEVEL",
        imported_symbol=None,
        is_heavy_target=True,
    )
    c = build_dependency_hotpath_candidate(
        candidate_index=0,
        dependency_name="numpy",
        source_path="a.py",
        line_number=1,
        candidate_kind="MODULE_TOP_LEVEL_HEAVY_IMPORT",
        candidate_status="NEEDS_BENCHMARK_RECEIPT",
        reason="x",
        related_import_site_hashes=(s.import_site_hash,),
    )
    r = build_dependency_import_and_hotpath_receipt(
        (s,), (c,), source_root_label="src", scanned_file_count=1
    )
    
    # Valid receipt should pass
    assert validate_dependency_import_and_hotpath_receipt(r)
    
    # Invalid schema_version should fail
    r_bad_schema = DependencyImportAndHotPathReceipt(
        **{**r.to_dict(), "schema_version": "INVALID_SCHEMA"}
    )
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"):
        validate_dependency_import_and_hotpath_receipt(r_bad_schema)
    
    # Invalid scan_mode should fail
    r_bad_mode = DependencyImportAndHotPathReceipt(
        **{**r.to_dict(), "scan_mode": "INVALID_MODE"}
    )
    with pytest.raises(ValueError, match="INVALID_SCAN_MODE"):
        validate_dependency_import_and_hotpath_receipt(r_bad_mode)
