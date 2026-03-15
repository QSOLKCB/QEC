from __future__ import annotations

from pathlib import Path

from qec.dev.test_selection import SpectralTestSelector


def test_module_from_path_mapping_and_filters() -> None:
    assert (
        SpectralTestSelector.module_from_path(
            "src/qec/analysis/spectral_frustration.py"
        )
        == "analysis.spectral_frustration"
    )
    assert SpectralTestSelector.module_from_path("tests/test_x.py") is None
    assert SpectralTestSelector.module_from_path("docs/index.md") is None
    assert SpectralTestSelector.module_from_path("src/qec/analysis/readme.txt") is None
    assert SpectralTestSelector.module_from_path("src/qec/analysis/__init__.py") is None
    assert (
        SpectralTestSelector.module_from_path(
            "src\\qec\\analysis\\spectral_entropy.py"
        )
        == "analysis.spectral_entropy"
    )


def test_module_to_tests_mapping_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_nb_flow_mutation.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_spectral_frustration.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_unrelated.py").write_text("", encoding="utf-8")

    selector = SpectralTestSelector(repo_root=tmp_path)
    modules = ["analysis.spectral_frustration", "discovery.nb_flow_mutation"]

    first = selector.tests_for_modules(modules)
    second = selector.tests_for_modules(list(reversed(modules)))

    assert first == second
    assert first == [
        "tests/test_nb_flow_mutation.py",
        "tests/test_spectral_frustration.py",
    ]


def test_select_tests_fallback_when_no_tests_match(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_other_module.py").write_text("", encoding="utf-8")

    selector = SpectralTestSelector(repo_root=tmp_path)
    selected = selector.select_tests(["src/qec/analysis/spectral_frustration.py"])

    assert selected == []


def test_select_tests_is_deterministic_across_runs(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_a_spectral_frustration.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_b_spectral_frustration.py").write_text("", encoding="utf-8")

    selector = SpectralTestSelector(repo_root=tmp_path)
    changed_files = ["src/qec/analysis/spectral_frustration.py"]

    first = selector.select_tests(changed_files)
    second = selector.select_tests(changed_files)

    assert first == second
    assert first == sorted(first)


def test_dependency_string_match_selects_test_file(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_import_path_only.py").write_text(
        "from src.qec.analysis.spectral_entropy import compute\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_unrelated.py").write_text("", encoding="utf-8")

    selector = SpectralTestSelector(repo_root=tmp_path)
    selected = selector.select_tests(["src/qec/analysis/spectral_entropy.py"])

    assert selected == ["tests/test_import_path_only.py"]
