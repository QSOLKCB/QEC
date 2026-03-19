from __future__ import annotations

from pathlib import Path

from qec.dev.dependency_graph import DependencyGraph
from qec.dev.test_selection import module_name_from_path, select_tests_for_changed_files


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_import_detection_and_reverse_dependency(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "qec"
    _write(package_root / "a.py", "")
    _write(package_root / "b.py", "import qec.a\n")

    graph = DependencyGraph(package_root)
    graph.build()

    assert graph.dependencies["src.qec.b"] == {"src.qec.a"}
    assert graph.reverse_dependencies["src.qec.a"] == {"src.qec.b"}


def test_dependency_closure_bfs_includes_indirect_dependents(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "qec"
    _write(package_root / "a.py", "")
    _write(package_root / "b.py", "from qec import a\n")
    _write(package_root / "c.py", "from qec import b\n")

    graph = DependencyGraph(package_root)
    graph.build()

    assert graph.affected_modules({"src.qec.a"}) == ["src.qec.a", "src.qec.b", "src.qec.c"]


def test_graph_build_is_deterministic(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "qec"
    _write(package_root / "a.py", "")
    _write(package_root / "b.py", "import qec.a\n")

    first = DependencyGraph(package_root)
    first.build()
    second = DependencyGraph(package_root)
    second.build()

    assert first.dependencies == second.dependencies
    assert first.reverse_dependencies == second.reverse_dependencies


def test_selector_includes_tests_for_dependent_modules(tmp_path: Path) -> None:
    repo_root = tmp_path
    package_root = repo_root / "src" / "qec"

    _write(package_root / "analysis" / "spectral_frustration.py", "")
    _write(
        package_root / "discovery" / "nonbacktracking_flow_mutation.py",
        "from qec.analysis import spectral_frustration\n",
    )
    _write(repo_root / "tests" / "test_spectral_frustration.py", "")
    _write(repo_root / "tests" / "test_nonbacktracking_flow_mutation.py", "")

    result = select_tests_for_changed_files(
        ["src/qec/analysis/spectral_frustration.py"],
        repo_root,
    )

    assert "tests/test_spectral_frustration.py" in result


def test_module_name_mapping() -> None:
    assert module_name_from_path("src/qec/analysis/spectral_frustration.py") == "analysis.spectral_frustration"
    assert module_name_from_path("scripts/select_tests.py") is None
