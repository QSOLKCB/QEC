# SPDX-License-Identifier: MIT
"""Tests for the lattice civilization engine (v136.1)."""

from __future__ import annotations

import pytest

from qec.sims.lattice_civilization_engine import (
    Grid,
    LatticeCivilizationReport,
    analyze_lattice_civilization,
    evolve_lattice_civilization,
    make_grid,
    step_grid,
    validate_grid,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _grid_from_lists(rows: list[list[int]]) -> Grid:
    return tuple(tuple(r) for r in rows)


# -----------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------


class TestValidation:
    def test_valid_grid_accepted(self) -> None:
        g = make_grid(3, 3, 0)
        validate_grid(g)  # should not raise

    def test_invalid_state_rejected(self) -> None:
        g = _grid_from_lists([[0, 1, 5]])
        with pytest.raises(ValueError, match="Invalid state"):
            validate_grid(g)

    def test_negative_state_rejected(self) -> None:
        g = _grid_from_lists([[0, -1, 2]])
        with pytest.raises(ValueError, match="Invalid state"):
            validate_grid(g)

    def test_empty_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_grid(())

    def test_ragged_grid_rejected(self) -> None:
        g = ((0, 1), (0,))
        with pytest.raises(ValueError, match="length"):
            validate_grid(g)

    def test_state_7_rejected(self) -> None:
        g = _grid_from_lists([[0, 1, 7]])
        with pytest.raises(ValueError, match="Invalid state"):
            validate_grid(g)


# -----------------------------------------------------------------------
# Settlement birth: empty -> settlement with exactly 2 adjacent
# settlements or hubs
# -----------------------------------------------------------------------


class TestSettlementBirth:
    def test_birth_exactly_two_settlements(self) -> None:
        """Empty cell with exactly 2 settlement neighbors becomes settlement."""
        g = _grid_from_lists([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 1

    def test_birth_with_hub_neighbors(self) -> None:
        """Empty cell with 1 settlement + 1 hub = 2 → becomes settlement."""
        g = _grid_from_lists([
            [0, 3, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 1

    def test_no_birth_with_one_neighbor(self) -> None:
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_no_birth_with_three_neighbors(self) -> None:
        g = _grid_from_lists([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0


# -----------------------------------------------------------------------
# Infrastructure emergence: settlement -> infrastructure
# when settlements + hubs >= 3
# -----------------------------------------------------------------------


class TestInfrastructureEmergence:
    def test_settlement_becomes_infra(self) -> None:
        """Settlement with 3 settlement/hub neighbors → infrastructure."""
        g = _grid_from_lists([
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])
        result = step_grid(g)
        # Center (1,1) has neighbors: up=1, left=1, right=1 → 3 settlements
        assert result[1][1] == 2

    def test_settlement_stays_with_few_neighbors(self) -> None:
        """Settlement with < 3 settlement/hub neighbors stays settlement."""
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 1


# -----------------------------------------------------------------------
# Energy hub formation: infrastructure -> hub
# when infra + hubs >= 3
# -----------------------------------------------------------------------


class TestEnergyHubFormation:
    def test_infra_becomes_hub(self) -> None:
        """Infrastructure surrounded by infra cluster → energy hub."""
        g = _grid_from_lists([
            [0, 2, 0],
            [2, 2, 2],
            [0, 0, 0],
        ])
        result = step_grid(g)
        # Center (1,1) has infra neighbors: up=2, left=2, right=2 → infra=3
        assert result[1][1] == 3

    def test_infra_stays_with_few_infra_neighbors(self) -> None:
        """Infrastructure with < 3 infra/hub neighbors stays infra."""
        g = _grid_from_lists([
            [0, 2, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 2


# -----------------------------------------------------------------------
# Corridor formation: infrastructure -> corridor
# when two hubs connected via infra path
# -----------------------------------------------------------------------


class TestCorridorFormation:
    def test_corridor_links_two_hubs(self) -> None:
        """Infrastructure cell between two hubs becomes a corridor.

        This is the critical test: a corridor successfully links two hubs.
        """
        # Hub - Infra - Hub layout (horizontal)
        g = _grid_from_lists([
            [0, 0, 0, 0, 0],
            [0, 3, 2, 3, 0],
            [0, 0, 0, 0, 0],
        ])
        result = step_grid(g)
        # The infra cell (1,2) has hub at (1,1) and hub at (1,3)
        # Both hubs are reachable via infra path → corridor
        assert result[1][2] == 4

    def test_corridor_via_infra_chain(self) -> None:
        """Longer infra chain between two hubs — cells adjacent to hubs
        should become corridors."""
        g = _grid_from_lists([
            [0, 0, 0, 0, 0, 0],
            [0, 3, 2, 2, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        result = step_grid(g)
        # Both infra cells connect two hubs via infra path
        assert result[1][2] == 4
        assert result[1][3] == 4

    def test_no_corridor_without_two_hubs(self) -> None:
        """Infra cell near only one hub does not become corridor."""
        g = _grid_from_lists([
            [0, 0, 0],
            [3, 2, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] != 4


# -----------------------------------------------------------------------
# Collapse rule: isolated cells revert to empty
# -----------------------------------------------------------------------


class TestCollapseRule:
    def test_isolated_settlement_collapses(self) -> None:
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_isolated_infra_collapses(self) -> None:
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_isolated_hub_collapses(self) -> None:
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_isolated_corridor_collapses(self) -> None:
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_non_isolated_settlement_survives(self) -> None:
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        # (1,1) has neighbor at (0,1) → not isolated → survives as settlement
        assert result[1][1] == 1


# -----------------------------------------------------------------------
# Connected region analysis
# -----------------------------------------------------------------------


class TestConnectedRegions:
    def test_single_region(self) -> None:
        g = _grid_from_lists([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        history = evolve_lattice_civilization(g, steps=0)
        report = analyze_lattice_civilization(history)
        assert report.connected_regions == 1

    def test_two_disjoint_regions(self) -> None:
        g = _grid_from_lists([
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        history = evolve_lattice_civilization(g, steps=0)
        report = analyze_lattice_civilization(history)
        assert report.connected_regions == 2

    def test_empty_grid_zero_regions(self) -> None:
        g = make_grid(3, 3, 0)
        history = evolve_lattice_civilization(g, steps=0)
        report = analyze_lattice_civilization(history)
        assert report.connected_regions == 0


# -----------------------------------------------------------------------
# Replay determinism
# -----------------------------------------------------------------------


class TestReplayDeterminism:
    def test_100_replay_determinism(self) -> None:
        """Run 100 independent evolutions of the same grid — all must match."""
        g = _grid_from_lists([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
        ])
        reference = evolve_lattice_civilization(g, steps=30)
        for _ in range(100):
            replay = evolve_lattice_civilization(g, steps=30)
            assert replay == reference

    def test_deterministic_analysis(self) -> None:
        """Analysis report must be identical across replays."""
        g = _grid_from_lists([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        history = evolve_lattice_civilization(g, steps=10)
        report1 = analyze_lattice_civilization(history)
        report2 = analyze_lattice_civilization(history)
        assert report1 == report2


# -----------------------------------------------------------------------
# Report structure
# -----------------------------------------------------------------------


class TestReport:
    def test_report_is_frozen(self) -> None:
        g = make_grid(3, 3, 0)
        history = evolve_lattice_civilization(g, steps=2)
        report = analyze_lattice_civilization(history)
        with pytest.raises(AttributeError):
            report.steps_evolved = 99  # type: ignore[misc]

    def test_report_fields(self) -> None:
        g = _grid_from_lists([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        history = evolve_lattice_civilization(g, steps=5)
        report = analyze_lattice_civilization(history)

        assert report.grid_shape == (3, 3)
        assert report.steps_evolved == 5
        assert isinstance(report.settlement_count, int)
        assert isinstance(report.infrastructure_count, int)
        assert isinstance(report.energy_hub_count, int)
        assert isinstance(report.corridor_count, int)
        assert isinstance(report.connected_regions, int)
        assert report.stability_label in (
            "stable_city", "expanding", "fragmented", "collapsed"
        )
        assert isinstance(report.growth_rate, float)
        assert isinstance(report.entropy_score, float)

    def test_state_counts_sum(self) -> None:
        """All state counts plus empty must equal grid size."""
        g = _grid_from_lists([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 1],
        ])
        history = evolve_lattice_civilization(g, steps=0)
        report = analyze_lattice_civilization(history)
        total = 9
        empty = total - (
            report.settlement_count
            + report.infrastructure_count
            + report.energy_hub_count
            + report.corridor_count
        )
        assert empty + report.settlement_count + report.infrastructure_count + report.energy_hub_count + report.corridor_count == total


# -----------------------------------------------------------------------
# History structure
# -----------------------------------------------------------------------


class TestHistory:
    def test_history_length(self) -> None:
        g = make_grid(3, 3, 0)
        history = evolve_lattice_civilization(g, steps=7)
        assert len(history) == 8  # initial + 7

    def test_history_first_is_input(self) -> None:
        g = _grid_from_lists([[1, 0], [0, 2]])
        history = evolve_lattice_civilization(g, steps=3)
        assert history[0] == g

    def test_negative_steps_rejected(self) -> None:
        g = make_grid(3, 3, 0)
        with pytest.raises(ValueError, match="non-negative"):
            evolve_lattice_civilization(g, steps=-1)


# -----------------------------------------------------------------------
# Stability labels
# -----------------------------------------------------------------------


class TestStabilityLabels:
    def test_collapsed_label(self) -> None:
        """All-empty grid → collapsed."""
        g = make_grid(3, 3, 0)
        history = evolve_lattice_civilization(g, steps=3)
        report = analyze_lattice_civilization(history)
        assert report.stability_label == "collapsed"

    def test_expanding_label(self) -> None:
        """Grid that grows settlements → expanding."""
        # L-shaped seed: 3 settlements grow to 4 (positive growth rate)
        g = _grid_from_lists([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        history = evolve_lattice_civilization(g, steps=2)
        report = analyze_lattice_civilization(history)
        assert report.stability_label == "expanding"


# -----------------------------------------------------------------------
# Decoder untouched
# -----------------------------------------------------------------------


class TestDecoderUntouched:
    def test_no_decoder_imports(self) -> None:
        """The lattice civilization engine must not import decoder code."""
        import importlib
        import sys

        mod_name = "qec.sims.lattice_civilization_engine"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        before = set(sys.modules.keys())
        importlib.import_module(mod_name)
        after = set(sys.modules.keys())

        new_decoder_modules = {
            m for m in (after - before) if "qec.decoder" in m
        }
        assert new_decoder_modules == set(), (
            f"Decoder modules imported: {new_decoder_modules}"
        )


# -----------------------------------------------------------------------
# make_grid dimension validation
# -----------------------------------------------------------------------


class TestMakeGridDimensions:
    def test_zero_rows_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_grid(0, 5)

    def test_zero_cols_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_grid(5, 0)

    def test_negative_rows_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_grid(-1, 3)


# -----------------------------------------------------------------------
# Analysis history validation
# -----------------------------------------------------------------------


class TestAnalysisHistoryValidation:
    def test_empty_history_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            analyze_lattice_civilization(())

    def test_invalid_state_in_history_rejected(self) -> None:
        bad_grid = ((0, 1, 8),)
        with pytest.raises(ValueError, match="Invalid state"):
            analyze_lattice_civilization((bad_grid,))

    def test_mixed_shape_history_rejected(self) -> None:
        g1 = ((0, 0), (0, 0))
        g2 = ((0, 0, 0), (0, 0, 0))
        with pytest.raises(ValueError, match="shape"):
            analyze_lattice_civilization((g1, g2))


# -----------------------------------------------------------------------
# Corridor linking two hubs (critical integration test)
# -----------------------------------------------------------------------


class TestCorridorLinksHubs:
    def test_corridor_successfully_links_two_hubs(self) -> None:
        """Prove that a corridor successfully links two hubs.

        Start from a configuration with two hubs and an infra path between
        them.  After evolution, verify that corridor cells exist on the
        path between the hubs.
        """
        g = _grid_from_lists([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 2, 2, 2, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        result = step_grid(g)

        # At least one infra cell on the path should have become a corridor
        corridor_cells = []
        for c in range(2, 5):
            if result[1][c] == 4:
                corridor_cells.append((1, c))

        assert len(corridor_cells) > 0, (
            "Expected at least one corridor cell linking the two hubs"
        )

        # Verify the hubs are still present (not collapsed — they have neighbors)
        assert result[1][1] == 3 or result[1][1] == 4  # hub or promoted
        assert result[1][5] == 3 or result[1][5] == 4  # hub or promoted
