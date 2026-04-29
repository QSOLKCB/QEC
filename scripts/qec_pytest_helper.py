"""Deterministic pytest command selector based on changed paths."""

from __future__ import annotations

from typing import Iterable

_INVALID_INPUT = "INVALID_INPUT"

_ESCALATION_SEGMENTS = {
    "identity",
    "hash",
    "canonical",
    "receipt",
    "protocol",
    "convergence",
    "conflict",
    "governance",
    "analysis",
    "ordering",
    "dedup",
    "canonicalize",
}

_ESCALATION_PREFIXES = (
    "src/qec/analysis/",
    "src/qec/decoder/",
)


def _raise_invalid_input() -> None:
    raise ValueError(_INVALID_INPUT)


def normalize_paths(changed_paths: Iterable[str]) -> list[str]:
    """Normalize and validate changed paths deterministically."""
    if isinstance(changed_paths, (str, bytes)):
        _raise_invalid_input()

    try:
        paths = list(changed_paths)
    except TypeError as exc:
        raise ValueError(_INVALID_INPUT) from exc

    if not paths:
        _raise_invalid_input()

    normalized: list[str] = []
    seen: set[str] = set()

    for raw_path in paths:
        if not isinstance(raw_path, str):
            _raise_invalid_input()
        if raw_path == "":
            _raise_invalid_input()

        path = raw_path.replace("\\", "/")
        while "//" in path:
            path = path.replace("//", "/")
        if path == "":
            _raise_invalid_input()

        if path in seen:
            _raise_invalid_input()

        seen.add(path)
        normalized.append(path)

    return normalized


def requires_full_suite(normalized_paths: list[str]) -> bool:
    """Return True when any changed path requires full-suite escalation."""
    paths = normalize_paths(normalized_paths)
    for path in paths:
        if any(path.startswith(prefix) for prefix in _ESCALATION_PREFIXES):
            return True
        segments = [segment for segment in path.split("/") if segment]
        if any(segment in _ESCALATION_SEGMENTS for segment in segments):
            return True
    return False


def _is_safe_partial_path(path: str) -> bool:
    return (
        path.startswith("tests/")
        or path.startswith("docs/")
        or path == "README"
        or path.startswith("README.")
    )


def determine_pytest_command(changed_paths: Iterable[str]) -> str:
    """Determine deterministic pytest command from changed paths."""
    paths = normalize_paths(changed_paths)

    if requires_full_suite(paths):
        return "pytest -q"

    if not all(_is_safe_partial_path(path) for path in paths):
        return "pytest -q"

    test_files = sorted(path for path in paths if path.startswith("tests/"))

    if not test_files:
        return "pytest -q"

    return "pytest -q " + " ".join(test_files)
