from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from src.qec.dev.dependency_graph import DependencyGraph


@dataclass(frozen=True)
class SelectionResult:
    changed_modules: tuple[str, ...]
    affected_modules: tuple[str, ...]
    selected_tests: tuple[str, ...]


def detect_changed_files(repo_root: Path) -> list[str]:
    commands = [
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    changed: set[str] = set()

    for cmd in commands:
        output = subprocess.check_output(cmd, cwd=repo_root, text=True)
        for line in output.splitlines():
            line = line.strip()
            if line:
                changed.add(line)

    return sorted(changed)


def select_tests_for_changed_files(changed_files: list[str], repo_root: Path) -> SelectionResult:
    changed_modules = sorted(
        {
            module
            for changed_file in changed_files
            for module in [module_name_from_path(changed_file)]
            if module is not None
        }
    )

    if not changed_modules:
        return SelectionResult((), (), ())

    graph = DependencyGraph(repo_root / "src" / "qec")
    graph.build()
    affected_modules = graph.affected_modules(set(changed_modules))

    selected_tests = sorted(
        {
            test_path
            for module in affected_modules
            for test_path in module_to_tests(module, repo_root)
        }
    )

    return SelectionResult(
        tuple(changed_modules),
        tuple(sorted(affected_modules)),
        tuple(selected_tests),
    )


def module_name_from_path(path_str: str) -> str | None:
    path = Path(path_str)
    if path.suffix != ".py":
        return None
    parts = path.parts
    if len(parts) < 3 or parts[0] != "src" or parts[1] != "qec":
        return None

    module_parts = list(Path(*parts[2:]).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    return ".".join(["src", "qec", *module_parts]) if module_parts else "src.qec"


def module_to_tests(module_name: str, repo_root: Path) -> list[str]:
    leaf = module_name.rsplit(".", 1)[-1]
    candidate = Path("tests") / f"test_{leaf}.py"
    if (repo_root / candidate).exists():
        return [candidate.as_posix()]
    return []
