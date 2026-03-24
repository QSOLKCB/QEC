from __future__ import annotations

from pathlib import Path

from qec.dev.coverage_data import (
    CoverageAwareSelector,
    CoverageDataLoader,
    changed_lines,
    parse_changed_lines_from_diff,
)


class _FakeCoverageData:
    def measured_files(self):
        return ["/repo/src/qec/analysis/sample.py"]

    def contexts_by_lineno(self, filename):
        assert filename == "/repo/src/qec/analysis/sample.py"
        return {
            10: ["tests/test_sample.py::test_a", "irrelevant_context"],
            11: ["tests/test_sample.py::test_b"],
        }


class _FakeLoader(CoverageDataLoader):
    def load(self) -> bool:
        self.line_to_tests = {
            ("src/qec/analysis/sample.py", 10): {"tests/test_sample.py::test_a"},
            ("src/qec/analysis/sample.py", 20): {"tests/test_sample.py::test_b"},
        }
        return True


class _MissingLoader(CoverageDataLoader):
    def load(self) -> bool:
        return False


def test_parse_changed_lines_from_diff_extracts_added_lines():
    diff_text = """\
diff --git a/src/qec/analysis/sample.py b/src/qec/analysis/sample.py
index abc..def 100644
--- a/src/qec/analysis/sample.py
+++ b/src/qec/analysis/sample.py
@@ -10,0 +20,3 @@
+line1
+line2
+line3
@@ -40 +50 @@
-line_old
+line_new
"""
    parsed = parse_changed_lines_from_diff(diff_text)
    assert parsed == {
        ("src/qec/analysis/sample.py", 20),
        ("src/qec/analysis/sample.py", 21),
        ("src/qec/analysis/sample.py", 22),
        ("src/qec/analysis/sample.py", 50),
    }


def test_loader_builds_line_to_test_mapping_deterministically():
    loader = CoverageDataLoader(repo_root=Path("/repo"))
    mapping = loader._build_line_to_tests(_FakeCoverageData())
    assert mapping == {
        ("src/qec/analysis/sample.py", 10): {"tests/test_sample.py::test_a"},
        ("src/qec/analysis/sample.py", 11): {"tests/test_sample.py::test_b"},
    }


def test_changed_lines_uses_git_diff_output(monkeypatch):
    class _Result:
        returncode = 0
        stdout = """\
+++ b/src/qec/analysis/sample.py
@@ -0,0 +3,2 @@
+a
+b
"""

    def _fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr("src.qec.dev.coverage_data.subprocess.run", _fake_run)
    assert changed_lines(repo_root=Path("/repo")) == {
        ("src/qec/analysis/sample.py", 3),
        ("src/qec/analysis/sample.py", 4),
    }


def test_coverage_selector_selects_only_covering_tests(monkeypatch):
    monkeypatch.setattr("src.qec.dev.coverage_data.CoverageDataLoader", _FakeLoader)
    monkeypatch.setattr(
        "src.qec.dev.coverage_data.changed_lines",
        lambda **_: {
            ("src/qec/analysis/sample.py", 20),
            ("src/qec/analysis/sample.py", 10),
        },
    )

    selector = CoverageAwareSelector(repo_root=Path("."))
    assert selector.select_tests() == (
        "tests/test_sample.py::test_a",
        "tests/test_sample.py::test_b",
    )


def test_coverage_selector_is_deterministic(monkeypatch):
    monkeypatch.setattr("src.qec.dev.coverage_data.CoverageDataLoader", _FakeLoader)
    monkeypatch.setattr(
        "src.qec.dev.coverage_data.changed_lines",
        lambda **_: {
            ("src/qec/analysis/sample.py", 20),
            ("src/qec/analysis/sample.py", 10),
        },
    )

    selector = CoverageAwareSelector(repo_root=Path("."))
    assert selector.select_tests() == selector.select_tests()


def test_selector_falls_back_when_coverage_unavailable(monkeypatch):
    monkeypatch.setattr("src.qec.dev.coverage_data.CoverageDataLoader", _MissingLoader)
    selector = CoverageAwareSelector(repo_root=Path("."))
    assert selector.select_tests() is None
