"""
Import hygiene tests for v3.1.2 interop layer.

Ensures that:
1. Core QEC imports do not pull in stim or pymatching.
2. bench/interop modules handle missing optional deps gracefully.
3. The interop layer does not leak into core modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


class TestCoreImportHygiene:
    """Verify that importing core QEC never imports third-party QEC tools."""

    def test_core_import_does_not_import_stim(self):
        """Importing src must not import stim."""
        # Run in a subprocess to get a clean sys.modules state.
        code = textwrap.dedent("""\
            import sys
            import src
            assert "stim" not in sys.modules, (
                "stim was imported by core QEC"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_core_import_does_not_import_pymatching(self):
        """Importing src must not import pymatching."""
        code = textwrap.dedent("""\
            import sys
            import src
            assert "pymatching" not in sys.modules, (
                "pymatching was imported by core QEC"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_bench_schema_does_not_import_stim(self):
        """Importing bench schema must not import stim."""
        code = textwrap.dedent("""\
            import sys
            from bench.schema import validate_result, dumps_result
            assert "stim" not in sys.modules, (
                "stim was imported by bench.schema"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_bench_config_does_not_import_stim(self):
        """Importing bench config must not import stim."""
        code = textwrap.dedent("""\
            import sys
            from bench.config import BenchmarkConfig
            assert "stim" not in sys.modules, (
                "stim was imported by bench.config"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "PASS" in result.stdout


class TestMissingOptionalDeps:
    """Verify graceful behavior when stim/pymatching are absent."""

    def test_interop_imports_module_loads(self):
        """bench/interop/imports.py must load without errors."""
        from bench.interop.imports import HAS_STIM, HAS_PYMATCHING, tool_versions
        # We don't assert HAS_STIM is False because it might be installed
        # in some environments. We just verify the module loads.
        assert isinstance(HAS_STIM, bool)
        assert isinstance(HAS_PYMATCHING, bool)
        tv = tool_versions()
        assert isinstance(tv, dict)
        assert "stim" in tv
        assert "pymatching" in tv

    def test_stim_baseline_skipped_when_unavailable(self):
        """Stim baseline returns 'skipped' record when stim is absent."""
        # Use subprocess to simulate missing stim.
        code = textwrap.dedent("""\
            import sys
            # Block stim import
            sys.modules["stim"] = None
            # Clear any cached imports
            for key in list(sys.modules.keys()):
                if "bench.interop" in key:
                    del sys.modules[key]

            from bench.interop.imports import HAS_STIM
            from bench.interop.runners import run_stim_pymatching_baseline

            result = run_stim_pymatching_baseline(
                code_family="repetition",
                distances=[3],
                p_values=[0.01],
                trials=10,
            )
            assert len(result) == 1
            assert result[0]["status"] == "skipped"
            assert "stim" in result[0]["reason"].lower()
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_tool_versions_with_missing_deps(self):
        """tool_versions() returns None for missing tools."""
        from bench.interop.imports import tool_versions
        tv = tool_versions()
        # If stim is not installed, version should be None
        # We can't guarantee it's missing, but we verify the structure
        for key in ("stim", "pymatching"):
            val = tv[key]
            assert val is None or isinstance(val, str)

    def test_require_stim_raises_when_absent(self):
        """require_stim() raises ImportError when stim is not installed."""
        from bench.interop.imports import HAS_STIM, require_stim
        if not HAS_STIM:
            with pytest.raises(ImportError, match="stim"):
                require_stim("test operation")

    def test_require_pymatching_raises_when_absent(self):
        """require_pymatching() raises ImportError when pymatching is absent."""
        from bench.interop.imports import HAS_PYMATCHING, require_pymatching
        if not HAS_PYMATCHING:
            with pytest.raises(ImportError, match="pymatching"):
                require_pymatching("test operation")
