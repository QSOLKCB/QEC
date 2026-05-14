from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .heavy_dependency_discovery import (
    HeavyDependencyDiscoveryManifest,
    HeavyDependencyTarget,
    build_default_unprobed_manifest,
    get_heavy_dependency_targets,
    validate_heavy_dependency_discovery_manifest,
)

_SCHEMA_VERSION = "DEPENDENCY_IMPORT_HOTPATH_V1"
_SCAN_MODE = "STATIC_AST_IMPORT_SURFACE_SCAN"
_MAX_IMPORT_SITES = 4096
_MAX_HOTPATH_CANDIDATES = 2048
_MAX_SOURCE_PATH_LENGTH = 256
_MAX_IMPORT_NAME_LENGTH = 128
_MAX_SYMBOL_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256

_ALLOWED_IMPORT_PLACEMENTS = {
    "MODULE_TOP_LEVEL",
    "FUNCTION_BODY",
    "CLASS_BODY",
    "TYPE_CHECKING_BLOCK",
    "UNKNOWN_PLACEMENT",
}
_ALLOWED_IMPORT_KINDS = {"IMPORT", "IMPORT_FROM", "DYNAMIC_REFERENCE"}
_ALLOWED_CANDIDATE_KINDS = {
    "MODULE_TOP_LEVEL_HEAVY_IMPORT",
    "REPEATED_IMPORT_REFERENCE",
    "DENSE_SPARSE_BOUNDARY",
    "PLOTTING_RENDER_BOUNDARY",
    "QUANTUM_BACKEND_BOUNDARY",
    "DATAFRAME_BOUNDARY",
    "AUDIO_MIDI_BOUNDARY",
    "INTERNAL_QEC_BOUNDARY",
    "UNKNOWN_CANDIDATE",
}
_ALLOWED_CANDIDATE_STATUS = {
    "CANDIDATE_ONLY",
    "NEEDS_EQUIVALENCE_PROOF",
    "NEEDS_BENCHMARK_RECEIPT",
    "BLOCKED_BY_POLICY",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DependencyImportSite:
    dependency_name: str
    import_name: str
    source_path: str
    line_number: int
    import_kind: str
    import_placement: str
    imported_symbol: str | None
    is_heavy_target: bool
    import_site_hash: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DependencyHotPathCandidate:
    candidate_index: int
    dependency_name: str
    source_path: str
    line_number: int
    candidate_kind: str
    candidate_status: str
    reason: str
    related_import_site_hashes: tuple[str, ...]
    candidate_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["related_import_site_hashes"] = list(self.related_import_site_hashes)
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DependencyImportAndHotPathReceipt:
    schema_version: str
    scan_mode: str
    source_root_label: str
    target_registry_hash: str
    scanned_file_count: int
    import_site_count: int
    hotpath_candidate_count: int
    import_sites: tuple[DependencyImportSite, ...]
    hotpath_candidates: tuple[DependencyHotPathCandidate, ...]
    first_import_site_hash: str
    final_import_site_hash: str
    first_candidate_hash: str
    final_candidate_hash: str
    dependency_hotpath_receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scan_mode": self.scan_mode,
            "source_root_label": self.source_root_label,
            "target_registry_hash": self.target_registry_hash,
            "scanned_file_count": self.scanned_file_count,
            "import_site_count": self.import_site_count,
            "hotpath_candidate_count": self.hotpath_candidate_count,
            "import_sites": [x.to_dict() for x in self.import_sites],
            "hotpath_candidates": [x.to_dict() for x in self.hotpath_candidates],
            "first_import_site_hash": self.first_import_site_hash,
            "final_import_site_hash": self.final_import_site_hash,
            "first_candidate_hash": self.first_candidate_hash,
            "final_candidate_hash": self.final_candidate_hash,
            "dependency_hotpath_receipt_hash": self.dependency_hotpath_receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_hash_format(value: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _norm_relpath(base: Path, path: Path) -> str:
    rel = path.relative_to(base).as_posix()
    if not rel or len(rel) > _MAX_SOURCE_PATH_LENGTH or rel.startswith("../"):
        raise ValueError("INVALID_SOURCE_PATH")
    return rel


def _base_site_payload(site: DependencyImportSite) -> dict[str, Any]:
    d = site.to_dict(); d.pop("import_site_hash"); return d


def _base_candidate_payload(c: DependencyHotPathCandidate) -> dict[str, Any]:
    d = c.to_dict(); d.pop("candidate_hash"); return d


def build_dependency_import_site(**kwargs: Any) -> DependencyImportSite:
    site = DependencyImportSite(import_site_hash="", **kwargs)
    validate_dependency_import_site(site, allow_blank_hash=True)
    return DependencyImportSite(**{**site.to_dict(), "import_site_hash": _hash_payload(_base_site_payload(site))})


def build_dependency_hotpath_candidate(**kwargs: Any) -> DependencyHotPathCandidate:
    c = DependencyHotPathCandidate(candidate_hash="", **kwargs)
    validate_dependency_hotpath_candidate(c, allow_blank_hash=True)
    return DependencyHotPathCandidate(**{**c.to_dict(), "related_import_site_hashes": tuple(c.related_import_site_hashes), "candidate_hash": _hash_payload(_base_candidate_payload(c))})


def build_dependency_import_and_hotpath_receipt(import_sites, hotpath_candidates, *, source_root_label: str, target_registry_hash: str | None = None, scanned_file_count: int) -> DependencyImportAndHotPathReceipt:
    sites = tuple(sorted(tuple(import_sites), key=lambda s: (s.source_path, s.line_number, s.dependency_name, s.import_name, s.import_kind, s.imported_symbol or "")))
    cands = tuple(hotpath_candidates)
    for s in sites:
        validate_dependency_import_site(s)
    for c in cands:
        validate_dependency_hotpath_candidate(c)
    if len({s.import_site_hash for s in sites}) != len(sites):
        raise ValueError("DUPLICATE_IMPORT_SITE")
    if len({c.candidate_hash for c in cands}) != len(cands):
        raise ValueError("DUPLICATE_CANDIDATE")
    if tuple(c.candidate_index for c in cands) != tuple(range(len(cands))):
        raise ValueError("CANDIDATE_ORDER_MISMATCH")
    manifest = build_default_unprobed_manifest()
    validate_heavy_dependency_discovery_manifest(manifest)
    registry_hash = target_registry_hash or manifest.heavy_dependency_discovery_manifest_hash
    receipt = DependencyImportAndHotPathReceipt(
        schema_version=_SCHEMA_VERSION, scan_mode=_SCAN_MODE, source_root_label=source_root_label,
        target_registry_hash=registry_hash, scanned_file_count=scanned_file_count,
        import_site_count=len(sites), hotpath_candidate_count=len(cands), import_sites=sites, hotpath_candidates=cands,
        first_import_site_hash=sites[0].import_site_hash if sites else "", final_import_site_hash=sites[-1].import_site_hash if sites else "",
        first_candidate_hash=cands[0].candidate_hash if cands else "", final_candidate_hash=cands[-1].candidate_hash if cands else "",
        dependency_hotpath_receipt_hash="",
    )
    payload = receipt.to_dict(); payload.pop("dependency_hotpath_receipt_hash")
    return DependencyImportAndHotPathReceipt(
        schema_version=receipt.schema_version,
        scan_mode=receipt.scan_mode,
        source_root_label=receipt.source_root_label,
        target_registry_hash=receipt.target_registry_hash,
        scanned_file_count=receipt.scanned_file_count,
        import_site_count=receipt.import_site_count,
        hotpath_candidate_count=receipt.hotpath_candidate_count,
        import_sites=receipt.import_sites,
        hotpath_candidates=receipt.hotpath_candidates,
        first_import_site_hash=receipt.first_import_site_hash,
        final_import_site_hash=receipt.final_import_site_hash,
        first_candidate_hash=receipt.first_candidate_hash,
        final_candidate_hash=receipt.final_candidate_hash,
        dependency_hotpath_receipt_hash=_hash_payload(payload),
    )


def validate_dependency_import_site(site: DependencyImportSite, allow_blank_hash: bool = False) -> bool:
    if not isinstance(site, DependencyImportSite): raise ValueError("INVALID_INPUT")
    registry = {t.dependency_name for t in get_heavy_dependency_targets()}
    if not isinstance(site.source_path, str) or not site.source_path or len(site.source_path) > _MAX_SOURCE_PATH_LENGTH: raise ValueError("INVALID_SOURCE_PATH")
    if not isinstance(site.line_number, int) or isinstance(site.line_number, bool) or site.line_number <= 0: raise ValueError("INVALID_INPUT")
    if site.import_kind not in _ALLOWED_IMPORT_KINDS: raise ValueError("INVALID_IMPORT_KIND")
    if site.import_placement not in _ALLOWED_IMPORT_PLACEMENTS: raise ValueError("INVALID_IMPORT_PLACEMENT")
    if site.is_heavy_target and site.dependency_name not in registry: raise ValueError("INVALID_DEPENDENCY_NAME")
    expected = _hash_payload(_base_site_payload(site))
    if site.import_site_hash == "" and allow_blank_hash: return True
    _validate_hash_format(site.import_site_hash)
    if site.import_site_hash != expected: raise ValueError("HASH_MISMATCH")
    return True


def validate_dependency_hotpath_candidate(c: DependencyHotPathCandidate, allow_blank_hash: bool = False) -> bool:
    if not isinstance(c, DependencyHotPathCandidate): raise ValueError("INVALID_INPUT")
    if c.candidate_kind not in _ALLOWED_CANDIDATE_KINDS: raise ValueError("INVALID_CANDIDATE_KIND")
    if c.candidate_status not in _ALLOWED_CANDIDATE_STATUS: raise ValueError("INVALID_CANDIDATE_STATUS")
    if not isinstance(c.reason, str) or len(c.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    if not isinstance(c.candidate_index, int) or isinstance(c.candidate_index, bool) or c.candidate_index < 0: raise ValueError("INVALID_INPUT")
    for h in c.related_import_site_hashes: _validate_hash_format(h)
    expected = _hash_payload(_base_candidate_payload(c))
    if c.candidate_hash == "" and allow_blank_hash: return True
    _validate_hash_format(c.candidate_hash)
    if c.candidate_hash != expected: raise ValueError("HASH_MISMATCH")
    return True


def validate_dependency_import_and_hotpath_receipt(receipt: DependencyImportAndHotPathReceipt) -> bool:
    if not isinstance(receipt, DependencyImportAndHotPathReceipt): raise ValueError("INVALID_INPUT")
    if receipt.import_site_count != len(receipt.import_sites) or receipt.hotpath_candidate_count != len(receipt.hotpath_candidates): raise ValueError("HOTPATH_COUNT_MISMATCH")
    for s in receipt.import_sites: validate_dependency_import_site(s)
    for c in receipt.hotpath_candidates: validate_dependency_hotpath_candidate(c)
    if tuple(c.candidate_index for c in receipt.hotpath_candidates) != tuple(range(len(receipt.hotpath_candidates))): raise ValueError("CANDIDATE_ORDER_MISMATCH")
    if receipt.import_sites:
        if receipt.first_import_site_hash != receipt.import_sites[0].import_site_hash or receipt.final_import_site_hash != receipt.import_sites[-1].import_site_hash: raise ValueError("HASH_MISMATCH")
    if receipt.hotpath_candidates:
        if receipt.first_candidate_hash != receipt.hotpath_candidates[0].candidate_hash or receipt.final_candidate_hash != receipt.hotpath_candidates[-1].candidate_hash: raise ValueError("HASH_MISMATCH")
    _validate_hash_format(receipt.dependency_hotpath_receipt_hash)
    payload = receipt.to_dict(); payload.pop("dependency_hotpath_receipt_hash")
    if _hash_payload(payload) != receipt.dependency_hotpath_receipt_hash: raise ValueError("HASH_MISMATCH")
    return True


def _placement(ancestors: list[ast.AST]) -> str:
    for node in reversed(ancestors):
        if isinstance(node, ast.If):
            t = node.test
            if isinstance(t, ast.Name) and t.id == "TYPE_CHECKING": return "TYPE_CHECKING_BLOCK"
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "typing" and t.attr == "TYPE_CHECKING": return "TYPE_CHECKING_BLOCK"
    for node in reversed(ancestors):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): return "FUNCTION_BODY"
        if isinstance(node, ast.ClassDef): return "CLASS_BODY"
    return "MODULE_TOP_LEVEL" if ancestors and isinstance(ancestors[0], ast.Module) else "UNKNOWN_PLACEMENT"


def scan_dependency_imports(source_root: str | Path, *, targets: tuple[HeavyDependencyTarget, ...] | None = None, include_tests: bool = False) -> DependencyImportAndHotPathReceipt:
    root = Path(source_root)
    if not root.exists() or not root.is_dir(): raise ValueError("INVALID_INPUT")
    scan_root = root / "src" if (root / "src").is_dir() else root
    tups = targets if targets is not None else get_heavy_dependency_targets()
    target_by_top = {t.import_name.split(".")[0]: t.dependency_name for t in tups}
    files = []
    for p in sorted(scan_root.rglob("*.py")):
        if p.is_symlink():
            continue
        rel = _norm_relpath(scan_root, p)
        if not include_tests and rel.startswith("tests/"):
            continue
        files.append(p)
    sites = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            raise ValueError("INVALID_INPUT")
        rel = _norm_relpath(scan_root, p)
        stack: list[ast.AST] = [tree]
        def walk(n: ast.AST):
            for child in ast.iter_child_nodes(n):
                stack.append(child)
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        top = alias.name.split(".")[0]
                        dep = target_by_top.get(top, top)
                        is_heavy = top in target_by_top
                        sites.append(build_dependency_import_site(dependency_name=dep, import_name=alias.name, source_path=rel, line_number=child.lineno, import_kind="IMPORT", import_placement=_placement(stack), imported_symbol=None, is_heavy_target=is_heavy))
                elif isinstance(child, ast.ImportFrom) and child.module:
                    top = child.module.split(".")[0]
                    dep = target_by_top.get(top, top)
                    is_heavy = top in target_by_top
                    for alias in child.names:
                        sites.append(build_dependency_import_site(dependency_name=dep, import_name=child.module, source_path=rel, line_number=child.lineno, import_kind="IMPORT_FROM", import_placement=_placement(stack), imported_symbol=alias.name, is_heavy_target=is_heavy))
                walk(child)
                stack.pop()
        walk(tree)
    sites_t = tuple(sorted(sites, key=lambda s: (s.source_path, s.line_number, s.dependency_name, s.import_name, s.import_kind, s.imported_symbol or "")))
    by_dep: dict[str, list[DependencyImportSite]] = {}
    for s in sites_t: by_dep.setdefault(s.dependency_name, []).append(s)
    cand = []
    def add(dep, sp, ln, kind, status, reason, hashes):
        cand.append(build_dependency_hotpath_candidate(candidate_index=len(cand), dependency_name=dep, source_path=sp, line_number=ln, candidate_kind=kind, candidate_status=status, reason=reason, related_import_site_hashes=tuple(sorted(hashes))))
    for dep, arr in by_dep.items():
        if len(arr) > 1: add(dep, arr[0].source_path, arr[0].line_number, "REPEATED_IMPORT_REFERENCE", "NEEDS_BENCHMARK_RECEIPT", "dependency appears in multiple import sites", [x.import_site_hash for x in arr])
        for s in arr:
            if s.import_placement == "MODULE_TOP_LEVEL" and s.is_heavy_target:
                add(dep, s.source_path, s.line_number, "MODULE_TOP_LEVEL_HEAVY_IMPORT", "NEEDS_BENCHMARK_RECEIPT", "module-level heavy import is a hot-path candidate", [s.import_site_hash])
        if dep == "matplotlib": add(dep, arr[0].source_path, arr[0].line_number, "PLOTTING_RENDER_BOUNDARY", "CANDIDATE_ONLY", "plot rendering boundary detected", [x.import_site_hash for x in arr])
        if dep == "pandas": add(dep, arr[0].source_path, arr[0].line_number, "DATAFRAME_BOUNDARY", "CANDIDATE_ONLY", "dataframe boundary detected", [x.import_site_hash for x in arr])
        if dep in {"qutip", "qiskit", "qiskit_aer", "stim", "pymatching"}: add(dep, arr[0].source_path, arr[0].line_number, "QUANTUM_BACKEND_BOUNDARY", "NEEDS_EQUIVALENCE_PROOF", "quantum backend boundary detected", [x.import_site_hash for x in arr])
        if dep == "mido": add(dep, arr[0].source_path, arr[0].line_number, "AUDIO_MIDI_BOUNDARY", "CANDIDATE_ONLY", "audio/MIDI boundary detected", [x.import_site_hash for x in arr])
        if dep == "scipy": add(dep, arr[0].source_path, arr[0].line_number, "DENSE_SPARSE_BOUNDARY", "NEEDS_EQUIVALENCE_PROOF", "dense/sparse boundary detected", [x.import_site_hash for x in arr])
        if dep == "qldpc_internal": add(dep, arr[0].source_path, arr[0].line_number, "INTERNAL_QEC_BOUNDARY", "NEEDS_EQUIVALENCE_PROOF", "internal qec boundary detected", [x.import_site_hash for x in arr])
    return build_dependency_import_and_hotpath_receipt(sites_t, tuple(cand), source_root_label=scan_root.name, scanned_file_count=len(files))


def validate_receipt_matches_scan(receipt: DependencyImportAndHotPathReceipt, source_root: str | Path, *, include_tests: bool = False) -> bool:
    rescan = scan_dependency_imports(source_root, include_tests=include_tests)
    if rescan.to_dict() != receipt.to_dict():
        raise ValueError("HOTPATH_RECEIPT_MISMATCH")
    return True
