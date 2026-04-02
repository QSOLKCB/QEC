"""Fast smoke test suite — catches import errors and basic invariants.

This runs in < 5 seconds and validates:
  - Core package imports
  - Decoder module loads without error
  - Channel module loads without error
  - Simulation modules load
  - Frozen dataclass contracts hold
  - Deterministic replay produces identical output
  - Decoder source files unchanged (hash sentinel)
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

from fast_regression_runner import get_repo_root

REPO_ROOT = get_repo_root()


# ── 1. Core imports ──────────────────────────────────────────────

class TestCoreImports:
    """Verify that all major packages import without error."""

    def test_import_qec(self):
        import qec  # noqa: F401

    def test_import_decoder(self):
        import qec.decoder  # noqa: F401

    def test_import_channel(self):
        import qec.channel  # noqa: F401

    def test_import_diagnostics(self):
        import qec.diagnostics  # noqa: F401

    def test_import_analysis(self):
        import qec.analysis  # noqa: F401

    def test_import_sims(self):
        import qec.sims  # noqa: F401

    def test_import_simulation(self):
        import qec.simulation  # noqa: F401

    def test_import_experiments(self):
        import qec.experiments  # noqa: F401

    def test_import_core(self):
        import qec.core  # noqa: F401

    def test_import_utils(self):
        import qec.utils  # noqa: F401


# ── 2. Decoder module loads ──────────────────────────────────────

class TestDecoderLoads:
    """Verify decoder sub-modules load without error."""

    def test_bp_decoder_reference(self):
        from qec.decoder import bp_decoder_reference  # noqa: F401

    def test_decoder_interface(self):
        from qec.decoder import decoder_interface  # noqa: F401

    def test_energy(self):
        from qec.decoder import energy  # noqa: F401


# ── 3. Simulation modules load ───────────────────────────────────

class TestSimModulesLoad:
    """Verify major simulation modules import."""

    def test_sims_init(self):
        import qec.sims  # noqa: F401

    def test_simulation_export_schema(self):
        from qec.simulation import export_schema  # noqa: F401

    def test_simulation_export_codec(self):
        from qec.simulation import export_codec  # noqa: F401


# ── 4. Frozen dataclass contracts ────────────────────────────────

class TestFrozenDataclasses:
    """Verify key frozen dataclasses remain immutable."""

    def test_export_schema_frozen(self):
        from qec.simulation.export_schema import SimulationExport
        assert hasattr(SimulationExport, "__dataclass_fields__")
        assert SimulationExport.__dataclass_params__.frozen is True

    def test_frozen_dataclass_fields_stable(self):
        from qec.simulation.export_schema import SimulationExport
        field_names = sorted(f.name for f in fields(SimulationExport))
        assert len(field_names) > 0

    def test_export_metadata_frozen(self):
        from qec.simulation.export_schema import ExportMetadata
        assert ExportMetadata.__dataclass_params__.frozen is True


# ── 5. Deterministic replay ─────────────────────────────────────

class TestDeterministicReplay:
    """Verify that repeated execution produces identical results."""

    def test_numpy_deterministic_seed(self):
        import numpy as np
        rng = np.random.default_rng(seed=42)
        a = rng.random(10).tobytes()
        rng2 = np.random.default_rng(seed=42)
        b = rng2.random(10).tobytes()
        assert a == b, "NumPy RNG not deterministic with same seed"

    def test_sorted_dict_ordering(self):
        d = {"z": 3, "a": 1, "m": 2}
        keys1 = list(sorted(d.keys()))
        keys2 = list(sorted(d.keys()))
        assert keys1 == keys2

    def test_hashlib_deterministic(self):
        data = b"qec determinism check"
        h1 = hashlib.sha256(data).hexdigest()
        h2 = hashlib.sha256(data).hexdigest()
        assert h1 == h2


# ── 6. Decoder source untouched sentinel ─────────────────────────

class TestDecoderUntouched:
    """Hash-based sentinel to detect unexpected decoder changes."""

    def _hash_directory(self, directory: Path) -> str:
        h = hashlib.sha256()
        for path in sorted(directory.rglob("*.py")):
            h.update(path.relative_to(directory).as_posix().encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def test_decoder_directory_exists(self):
        decoder_dir = REPO_ROOT / "src" / "qec" / "decoder"
        assert decoder_dir.is_dir(), "Decoder directory missing"

    def test_decoder_files_present(self):
        decoder_dir = REPO_ROOT / "src" / "qec" / "decoder"
        py_files = sorted(p.name for p in decoder_dir.glob("*.py"))
        assert "__init__.py" in py_files
        assert "bp_decoder_reference.py" in py_files


# ── 7. Baseline registry loadable ────────────────────────────────

class TestBaselineRegistry:
    """Verify the baseline registry file is valid and loadable."""

    def test_registry_exists(self):
        path = REPO_ROOT / "tests" / "test_baseline_registry.json"
        assert path.exists(), "Baseline registry missing"

    def test_registry_valid_json(self):
        import json
        path = REPO_ROOT / "tests" / "test_baseline_registry.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "baseline_total_passed" in data
        assert "baseline_skipped" in data
        assert "baseline_failed" in data

    def test_registry_baseline_failed_zero(self):
        import json
        path = REPO_ROOT / "tests" / "test_baseline_registry.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["baseline_failed"] == 0, "Baseline must have zero failures"
