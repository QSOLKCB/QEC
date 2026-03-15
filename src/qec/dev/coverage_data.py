"""Coverage-aware deterministic test selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, Iterable, Optional, Set, Tuple


LineRef = Tuple[str, int]


def _normalize_repo_path(path_value: str, repo_root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    return str(path).replace("\\", "/")


def parse_changed_lines_from_diff(diff_text: str) -> Set[LineRef]:
    """Parse git unified diff output and return changed line references."""
    changed: Set[LineRef] = set()
    current_file: Optional[str] = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:]
            continue

        if current_file is None or not raw_line.startswith("@@"):
            continue

        # e.g. @@ -10,0 +20,3 @@ or @@ -4 +5 @@
        try:
            plus_chunk = raw_line.split("+", 1)[1].split(" ", 1)[0]
        except IndexError:
            continue

        if "," in plus_chunk:
            start_str, count_str = plus_chunk.split(",", 1)
            start = int(start_str)
            count = int(count_str)
        else:
            start = int(plus_chunk)
            count = 1

        for line_no in range(start, start + count):
            changed.add((current_file, line_no))

    return changed


def changed_lines(repo_root: Path | str = ".", diff_ref: str = "HEAD") -> Set[LineRef]:
    """Return changed lines from `git diff -U0 <diff_ref>`."""
    root = Path(repo_root)
    result = subprocess.run(
        ["git", "diff", "-U0", diff_ref],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return parse_changed_lines_from_diff(result.stdout)


@dataclass
class CoverageDataLoader:
    """Read `.coverage` and build deterministic line -> tests mapping."""

    coverage_file: Path | str = ".coverage"
    repo_root: Path | str = "."

    def __post_init__(self) -> None:
        self.coverage_file = Path(self.coverage_file)
        self.repo_root = Path(self.repo_root)
        self.line_to_tests: Dict[LineRef, Set[str]] = {}

    def _import_coverage(self):
        try:
            import coverage
        except ImportError:
            return None
        return coverage

    @staticmethod
    def _is_test_context(context: str) -> bool:
        return "::" in context

    def load(self) -> bool:
        coverage = self._import_coverage()
        if coverage is None or not self.coverage_file.exists():
            return False

        cov = coverage.Coverage(data_file=str(self.coverage_file))
        data = cov.get_data()
        if not data.measured_files():
            try:
                if hasattr(data, "read_file"):
                    data.read_file(str(self.coverage_file))
                else:
                    data.read()
            except Exception:
                return False

        self.line_to_tests = self._build_line_to_tests(data)
        return True

    def _build_line_to_tests(self, coverage_data) -> Dict[LineRef, Set[str]]:
        mapping: Dict[LineRef, Set[str]] = {}
        files = sorted(coverage_data.measured_files())

        for filename in files:
            normalized_file = _normalize_repo_path(filename, self.repo_root)
            contexts_by_line = coverage_data.contexts_by_lineno(filename)
            for line_no in sorted(contexts_by_line):
                contexts = sorted(contexts_by_line[line_no])
                tests = {ctx for ctx in contexts if self._is_test_context(ctx)}
                if tests:
                    mapping[(normalized_file, line_no)] = tests

        return mapping

    def tests_for_lines(self, lines: Iterable[LineRef]) -> Set[str]:
        selected: Set[str] = set()
        for line_ref in lines:
            selected.update(self.line_to_tests.get(line_ref, set()))
        return selected


@dataclass
class CoverageAwareSelector:
    """Select tests that executed changed lines using coverage contexts."""

    repo_root: Path | str = "."
    coverage_file: Path | str = ".coverage"
    diff_ref: str = "HEAD"

    def select_tests(self) -> Optional[Tuple[str, ...]]:
        loader = CoverageDataLoader(coverage_file=self.coverage_file, repo_root=self.repo_root)
        if not loader.load():
            return None

        lines = changed_lines(repo_root=self.repo_root, diff_ref=self.diff_ref)
        if not lines:
            return tuple()

        normalized_lines = {
            (_normalize_repo_path(path, Path(self.repo_root)), line_no)
            for path, line_no in lines
        }
        selected = loader.tests_for_lines(normalized_lines)
        return tuple(sorted(selected))
