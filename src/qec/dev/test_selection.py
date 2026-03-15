"""Deterministic spectral test selection utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


class SpectralTestSelector:
    """Select tests affected by modified source modules."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        if repo_root is None:
            self.repo_root = Path(__file__).resolve().parents[3]
        else:
            self.repo_root = Path(repo_root).resolve()

    def detect_changed_files(self) -> list[str]:
        """Return changed repository files from git diff against HEAD."""
        try:
            output = subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.repo_root,
                text=True,
            )
        except Exception:
            return []

        changed: list[str] = []
        for raw_path in output.splitlines():
            path = raw_path.strip()
            if not path:
                continue
            full_path = (self.repo_root / path).resolve()
            if full_path.exists() and self.repo_root in full_path.parents:
                changed.append(path)
        return sorted(set(changed))

    @staticmethod
    def module_from_path(path: str) -> str | None:
        """Map source file paths to dotted module names under ``src/qec``."""
        if not path.endswith(".py"):
            return None
        normalized = path.replace("\\", "/")
        if normalized.startswith("tests/"):
            return None
        if normalized.startswith("docs/"):
            return None
        if not normalized.startswith("src/qec/"):
            return None
        if normalized.endswith("/__init__.py"):
            return None

        rel = normalized[len("src/qec/") : -len(".py")]
        if not rel:
            return None
        return rel.replace("/", ".")

    def changed_modules(self, changed_files: list[str] | None = None) -> list[str]:
        """Return sorted affected modules for changed source files."""
        files = changed_files if changed_files is not None else self.detect_changed_files()
        modules = {
            module
            for module in (self.module_from_path(path) for path in files)
            if module is not None
        }
        return sorted(modules)

    def tests_for_modules(self, modules: list[str]) -> list[str]:
        """Return sorted test file paths affected by changed modules."""
        tests_dir = self.repo_root / "tests"
        if not tests_dir.exists():
            return []

        candidates = sorted(
            path.relative_to(self.repo_root).as_posix()
            for path in tests_dir.glob("test_*.py")
            if path.is_file()
        )

        selected: set[str] = set()
        module_set = sorted(set(modules))
        for module in sorted(set(modules)):
            module_tokens = {
                module.replace(".", "_"),
                module.rsplit(".", maxsplit=1)[-1],
            }
            for test_path in candidates:
                stem = Path(test_path).stem
                if any(token in stem for token in module_tokens):
                    selected.add(test_path)

        selected.update(self._tests_importing_modules(candidates, module_set))

        return sorted(selected)

    def _tests_importing_modules(self, candidates: list[str], modules: Iterable[str]) -> list[str]:
        selected: set[str] = set()
        module_patterns = sorted(
            {
                pattern
                for module in modules
                for pattern in (module, f"src.qec.{module}")
            }
        )
        if not module_patterns:
            return []

        for test_path in candidates:
            full_path = self.repo_root / test_path
            try:
                content = full_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if any(pattern in content for pattern in module_patterns):
                selected.add(test_path)
        return sorted(selected)

    def changed_test_files(self, changed_files: list[str]) -> list[str]:
        """Return directly changed tests so test-only edits remain selectable."""
        selected = [
            path.replace("\\", "/")
            for path in changed_files
            if path.replace("\\", "/").startswith("tests/test_") and path.endswith(".py")
        ]
        return sorted(set(selected))

    def select_tests(self, changed_files: list[str] | None = None) -> list[str]:
        """Select tests affected by changed files.

        Returns an empty list when no mapping is found, allowing callers
        to fall back to collecting the full test suite.
        """
        files = changed_files if changed_files is not None else self.detect_changed_files()
        modules = self.changed_modules(files)
        module_tests = self.tests_for_modules(modules)
        direct_tests = self.changed_test_files(files)
        return sorted(set(module_tests).union(direct_tests))


def module_name_from_path(path: str) -> str | None:
    """Backward-compatible wrapper for module path mapping."""
    return SpectralTestSelector.module_from_path(path)


def select_tests_for_changed_files(
    changed_files: list[str],
    repo_root: str | Path | None = None,
) -> list[str]:
    """Backward-compatible wrapper for deterministic test selection."""
    return SpectralTestSelector(repo_root=repo_root).select_tests(changed_files)
