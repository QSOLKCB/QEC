from __future__ import annotations

import pytest

from scripts.qec_pytest_helper import (
    determine_pytest_command,
    normalize_paths,
    requires_full_suite,
)


def test_escalation_trigger_segments() -> None:
    assert requires_full_suite(["src/qec/x/identity/guard.py"])
    assert requires_full_suite(["pkg/hash/utils.py"])
    assert requires_full_suite(["a/b/canonicalize/step.py"])


def test_escalation_trigger_core_prefixes() -> None:
    assert requires_full_suite(["src/qec/analysis/module.py"])
    assert requires_full_suite(["src/qec/decoder/core.py"])


def test_safe_partial_single_test() -> None:
    command = determine_pytest_command(["tests/test_alpha.py"])
    assert command == "pytest -q -- tests/test_alpha.py"


def test_safe_partial_multiple_tests_sorted() -> None:
    command = determine_pytest_command(
        ["tests/test_zeta.py", "tests/test_alpha.py", "docs/notes.md", "README.md"]
    )
    assert command == "pytest -q -- tests/test_alpha.py tests/test_zeta.py"


def test_requires_full_suite_accepts_empty_normalized_paths() -> None:
    assert requires_full_suite([]) is False


def test_safe_partial_test_paths_are_shell_quoted() -> None:
    command = determine_pytest_command(["tests/test alpha.py", "tests/-test_beta.py"])
    assert command == "pytest -q -- tests/-test_beta.py 'tests/test alpha.py'"


def test_mixed_paths_force_full_suite() -> None:
    command = determine_pytest_command(["tests/test_alpha.py", "src/qec/analysis/x.py"])
    assert command == "pytest -q"


def test_non_safe_paths_force_full_suite_fallback() -> None:
    command = determine_pytest_command(["scripts/tool.py"])
    assert command == "pytest -q"


def test_invalid_inputs_raise() -> None:
    invalid_values = [None, "tests/test_a.py", [], [""], [1], ["a", "a"]]
    for value in invalid_values:
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            determine_pytest_command(value)  # type: ignore[arg-type]


def test_duplicate_paths_rejected_after_slash_normalization() -> None:
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        normalize_paths([r"tests\\test_alpha.py", "tests/test_alpha.py"])


def test_determinism_100_runs() -> None:
    changed = ["tests/test_b.py", "tests/test_a.py", "docs/readme_notes.md"]
    expected = "pytest -q -- tests/test_a.py tests/test_b.py"
    for _ in range(100):
        assert determine_pytest_command(changed) == expected


def test_order_invariance() -> None:
    forward = ["tests/test_b.py", "tests/test_a.py", "docs/doc.md"]
    reverse = list(reversed(forward))
    assert determine_pytest_command(forward) == determine_pytest_command(reverse)
