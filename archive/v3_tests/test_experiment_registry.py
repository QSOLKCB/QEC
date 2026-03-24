from __future__ import annotations

import pytest

from qec.experiments.registry import ExperimentRegistry, DEFAULT_REGISTRY, discover_experiments


def test_registry_register_and_get() -> None:
    registry = ExperimentRegistry()

    def _run(_: dict[str, object]) -> dict[str, object]:
        return {"ok": True}

    registry.register("alpha", _run)

    assert registry.get("alpha") is _run


def test_registry_duplicate_raises() -> None:
    registry = ExperimentRegistry()

    def _run(_: dict[str, object]) -> dict[str, object]:
        return {"ok": True}

    registry.register("alpha", _run)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("alpha", _run)


def test_registry_list_is_sorted() -> None:
    registry = ExperimentRegistry()

    registry.register("zeta", lambda _: {"ok": True})
    registry.register("alpha", lambda _: {"ok": True})

    assert registry.list() == ["alpha", "zeta"]


def test_cli_discovery_registers_expected_experiments() -> None:
    discover_experiments()
    names = DEFAULT_REGISTRY.list()

    assert "bp-threshold" in names
    assert "spectral-heatmap" in names
    assert "ldpc-search" in names
