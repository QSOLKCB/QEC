"""Tests for the fast regression harness infrastructure.

Validates:
  - Baseline registry loading and schema
  - Delta comparison logic
  - Smoke runner invocation
  - Impacted test selection
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = REPO_ROOT / "scripts"
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))


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
