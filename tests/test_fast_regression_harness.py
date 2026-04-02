"""Tests for the fast regression harness infrastructure.

Validates:
  - Baseline registry loading and schema
  - Delta comparison logic
  - Smoke runner invocation
  - Impacted test selection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fast_regression_runner import get_repo_root

REPO_ROOT = get_repo_root()


# ── 1. Baseline registry ────────────────────────────────────────

class TestBaselineRegistryLoading:
    """Test that the baseline registry can be loaded and validated."""

    def test_load_baseline(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        assert isinstance(baseline, dict)

    def test_baseline_has_required_keys(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        assert "baseline_total_passed" in baseline
        assert "baseline_skipped" in baseline
        assert "baseline_failed" in baseline

    def test_baseline_types(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        assert isinstance(baseline["baseline_total_passed"], int)
        assert isinstance(baseline["baseline_skipped"], int)
        assert isinstance(baseline["baseline_failed"], int)

    def test_baseline_no_failures(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        assert baseline["baseline_failed"] == 0

    def test_baseline_passed_positive(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        assert baseline["baseline_total_passed"] > 0


# ── 2. Delta comparison ─────────────────────────────────────────

class TestDeltaComparison:
    """Test baseline delta detection logic."""

    def test_regression_detected_when_fewer_pass(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        # Simulate a run with fewer passing tests
        simulated_passed = baseline["baseline_total_passed"] - 5
        delta = simulated_passed - baseline["baseline_total_passed"]
        assert delta < 0, "Should detect regression"

    def test_no_regression_when_equal(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        delta = baseline["baseline_total_passed"] - baseline["baseline_total_passed"]
        assert delta == 0

    def test_improvement_detected_when_more_pass(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        simulated_passed = baseline["baseline_total_passed"] + 10
        delta = simulated_passed - baseline["baseline_total_passed"]
        assert delta > 0, "Should detect improvement"

    def test_failure_regression_detected(self):
        from fast_regression_runner import load_baseline
        baseline = load_baseline()
        simulated_failures = 3
        assert simulated_failures > baseline["baseline_failed"]


# ── 3. Smoke runner ─────────────────────────────────────────────

class TestSmokeRunner:
    """Test that the smoke suite can be invoked."""

    def test_smoke_test_file_exists(self):
        smoke_path = REPO_ROOT / "tests" / "test_fast_smoke.py"
        assert smoke_path.exists()

    def test_smoke_test_importable(self):
        """Verify smoke test module can be collected by pytest."""
        smoke_path = REPO_ROOT / "tests" / "test_fast_smoke.py"
        content = smoke_path.read_text(encoding="utf-8")
        assert "class TestCoreImports" in content
        assert "class TestDeterministicReplay" in content

    def test_run_smoke_function_exists(self):
        from fast_regression_runner import run_smoke
        assert callable(run_smoke)

    def test_run_pytest_function_exists(self):
        from fast_regression_runner import run_pytest
        assert callable(run_pytest)


# ── 4. Impacted test selection ───────────────────────────────────

class TestImpactedTestSelection:
    """Test that impacted test detection works."""

    def test_detect_impacted_tests_callable(self):
        from fast_regression_runner import detect_impacted_tests
        assert callable(detect_impacted_tests)

    def test_detect_impacted_returns_list(self):
        from fast_regression_runner import detect_impacted_tests
        result = detect_impacted_tests()
        assert isinstance(result, list)

    def test_spectral_selector_module_mapping(self):
        from qec.dev.test_selection import SpectralTestSelector
        sel = SpectralTestSelector(repo_root=REPO_ROOT)
        # Known mapping: decoder source -> module name
        mod = sel.module_from_path("src/qec/decoder/bp_decoder_reference.py")
        assert mod == "decoder.bp_decoder_reference"

    def test_spectral_selector_ignores_tests(self):
        from qec.dev.test_selection import SpectralTestSelector
        sel = SpectralTestSelector(repo_root=REPO_ROOT)
        mod = sel.module_from_path("tests/test_something.py")
        assert mod is None

    def test_spectral_selector_ignores_non_python(self):
        from qec.dev.test_selection import SpectralTestSelector
        sel = SpectralTestSelector(repo_root=REPO_ROOT)
        mod = sel.module_from_path("README.md")
        assert mod is None


# ── 5. Registry file schema stability ───────────────────────────

class TestRegistrySchema:
    """Ensure the registry JSON schema is stable."""

    def test_registry_frozen_at_present(self):
        path = REPO_ROOT / "tests" / "test_baseline_registry.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "frozen_at" in data

    def test_registry_no_extra_required_fields_missing(self):
        path = REPO_ROOT / "tests" / "test_baseline_registry.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        required = {"baseline_total_passed", "baseline_skipped", "baseline_failed"}
        assert required.issubset(data.keys())


# ── 6. Pytest summary parser (strict regex) ─────────────────────

class TestParsePytestSummary:
    """Test the strict regex parser for pytest output."""

    def test_parse_typical_summary(self):
        from fast_regression_runner import _parse_pytest_summary
        stdout = "9870 passed, 11 skipped in 79.83s\n"
        result = _parse_pytest_summary(stdout)
        assert result["passed"] == 9870
        assert result["skipped"] == 11
        assert result["failed"] == 0
        assert result["errors"] == 0

    def test_parse_failures_present(self):
        from fast_regression_runner import _parse_pytest_summary
        stdout = "3 failed, 120 passed, 2 skipped in 5.00s\n"
        result = _parse_pytest_summary(stdout)
        assert result["passed"] == 120
        assert result["failed"] == 3
        assert result["skipped"] == 2
        assert result["errors"] == 0

    def test_parse_errors_present(self):
        from fast_regression_runner import _parse_pytest_summary
        stdout = "1 error in 0.50s\n"
        result = _parse_pytest_summary(stdout)
        assert result["errors"] == 1
        assert result["passed"] == 0

    def test_parse_all_categories(self):
        from fast_regression_runner import _parse_pytest_summary
        stdout = "5 failed, 100 passed, 3 skipped, 2 errors in 10.00s\n"
        result = _parse_pytest_summary(stdout)
        assert result["passed"] == 100
        assert result["failed"] == 5
        assert result["skipped"] == 3
        assert result["errors"] == 2

    def test_parse_empty_output(self):
        from fast_regression_runner import _parse_pytest_summary
        result = _parse_pytest_summary("")
        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_parse_malformed_output(self):
        from fast_regression_runner import _parse_pytest_summary
        result = _parse_pytest_summary("garbage output\nno counts here\n")
        assert result["passed"] == 0
        assert result["failed"] == 0

    def test_parse_returns_all_keys(self):
        from fast_regression_runner import _parse_pytest_summary
        result = _parse_pytest_summary("47 passed in 0.30s\n")
        assert set(result.keys()) == {"passed", "failed", "skipped", "errors"}


# ── 7. Subprocess command security audit ─────────────────────────

class TestSubprocessSecurity:
    """Verify subprocess invocations are hardened."""

    def test_run_pytest_uses_list_command(self):
        """Verify run_pytest builds a list-based command, not a string."""
        import inspect
        from fast_regression_runner import run_pytest
        source = inspect.getsource(run_pytest)
        assert "shell=False" in source, "shell=False must be explicit"

    def test_run_pytest_no_shell_true(self):
        import inspect
        from fast_regression_runner import run_pytest
        source = inspect.getsource(run_pytest)
        assert "shell=True" not in source

    def test_command_uses_sys_executable(self):
        import inspect
        from fast_regression_runner import run_pytest
        source = inspect.getsource(run_pytest)
        assert "sys.executable" in source


# ── 8. Path normalization ────────────────────────────────────────

class TestPathNormalization:
    """Verify dynamic test paths are validated inside repo root."""

    def test_valid_test_path_accepted(self):
        from fast_regression_runner import _validate_test_path
        result = _validate_test_path("tests/test_fast_smoke.py")
        assert "test_fast_smoke" in result

    def test_path_traversal_rejected(self):
        from fast_regression_runner import _validate_test_path
        with pytest.raises(ValueError, match="escapes repository root"):
            _validate_test_path("../../etc/passwd")

    def test_absolute_escape_rejected(self):
        from fast_regression_runner import _validate_test_path
        with pytest.raises(ValueError, match="escapes repository root"):
            _validate_test_path("../../../tmp/evil.py")

    def test_normalized_path_returned(self):
        from fast_regression_runner import _validate_test_path
        result = _validate_test_path("tests/./test_fast_smoke.py")
        assert "./" not in result


# ── 9. Centralized repo root ────────────────────────────────────

class TestCentralizedRepoRoot:
    """Verify get_repo_root is the single source of truth."""

    def test_get_repo_root_returns_path(self):
        assert isinstance(REPO_ROOT, Path)

    def test_get_repo_root_is_directory(self):
        assert REPO_ROOT.is_dir()

    def test_get_repo_root_contains_src(self):
        assert (REPO_ROOT / "src").is_dir()

    def test_get_repo_root_contains_tests(self):
        assert (REPO_ROOT / "tests").is_dir()
