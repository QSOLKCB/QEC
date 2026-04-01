# SPDX-License-Identifier: MIT
"""Tests for the qutrit life automata engine (v136)."""

from __future__ import annotations

import pytest

from qec.sims.qutrit_life_automata import (
    Grid,
    QutritLifeReport,
    analyze_qutrit_life,
    evolve_qutrit_life,
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
        g = _grid_from_lists([[0, 1, 3]])
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


# -----------------------------------------------------------------------
# Birth rule: vacuum -> active with exactly 3 alive neighbors
# -----------------------------------------------------------------------


class TestBirthRule:
    def test_birth_exactly_three_active_neighbors(self) -> None:
        """Center cell (vacuum) with exactly 3 active neighbors becomes active."""
        g = _grid_from_lists([
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 1

    def test_no_birth_with_two_neighbors(self) -> None:
        g = _grid_from_lists([
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_no_birth_with_four_neighbors(self) -> None:
        g = _grid_from_lists([
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0


# -----------------------------------------------------------------------
# Resonance rule: active -> resonant when energy >= 4
# -----------------------------------------------------------------------


class TestResonanceRule:
    def test_active_becomes_resonant_high_energy(self) -> None:
        """Active cell with energy = active + 2*resonant >= 4."""
        # Center is active=1, neighbors: 2 resonant neighbors → energy = 0 + 2*2 = 4
        # Also need alive_neighbors in [2,3] for survival
        g = _grid_from_lists([
            [2, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
        ])
        result = step_grid(g)
        # alive_neighbors = 2, energy = 0 + 2*2 = 4 → resonant
        assert result[1][1] == 2

    def test_active_stays_active_low_energy(self) -> None:
        """Active cell with energy < 4 stays active (if surviving)."""
        # 2 active neighbors, 0 resonant → energy = 2 < 4
        g = _grid_from_lists([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ])
        result = step_grid(g)
        # alive_neighbors = 2, energy = 2 < 4 → stays active
        assert result[1][1] == 1


# -----------------------------------------------------------------------
# Decay rules
# -----------------------------------------------------------------------


class TestDecayRules:
    def test_resonant_overload_decay(self) -> None:
        """Resonant cell with >= 4 resonant neighbors decays to vacuum."""
        g = _grid_from_lists([
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ])
        result = step_grid(g)
        # Center has 8 resonant neighbors → overload → 0
        assert result[1][1] == 0

    def test_resonant_deexcites_to_active(self) -> None:
        """Resonant cell with < 4 resonant neighbors becomes active."""
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        # 0 resonant neighbors → de-excite to 1
        assert result[1][1] == 1

    def test_active_dies_isolation(self) -> None:
        """Active cell with < 2 alive neighbors dies."""
        g = _grid_from_lists([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        assert result[1][1] == 0

    def test_active_dies_overcrowding(self) -> None:
        """Active cell with > 3 alive neighbors dies."""
        g = _grid_from_lists([
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ])
        result = step_grid(g)
        # Center has 4 alive neighbors → dies
        assert result[1][1] == 0


# -----------------------------------------------------------------------
# Fixed point detection
# -----------------------------------------------------------------------


class TestFixedPoint:
    def test_all_vacuum_is_fixed_point(self) -> None:
        g = make_grid(5, 5, 0)
        history = evolve_qutrit_life(g, steps=5)
        report = analyze_qutrit_life(history)
        assert report.stability_label == "fixed"
        assert report.period_detected == 1
        for h in history:
            assert h == g


# -----------------------------------------------------------------------
# Oscillator detection
# -----------------------------------------------------------------------


class TestOscillator:
    def test_stable_oscillator(self) -> None:
        """Construct a configuration that oscillates with a short period.

        Strategy: use a small pattern in a larger grid so boundary effects
        don't interfere, and evolve enough steps to detect the period.
        """
        # Start with a line of 3 active cells in a 5x5 grid (blinker analog)
        rows = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        g = _grid_from_lists(rows)
        history = evolve_qutrit_life(g, steps=20)

        # The pattern must not be a fixed point (it should evolve)
        assert history[1] != history[0], "Pattern should evolve"

        report = analyze_qutrit_life(history)
        # Must detect a period
        assert report.period_detected >= 2, (
            f"Expected oscillator period >= 2, got {report.period_detected}"
        )
        assert report.stability_label == "oscillatory"

    def test_blinker_period(self) -> None:
        """The 3-cell horizontal line should produce a short period oscillator."""
        rows = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        g = _grid_from_lists(rows)
        history = evolve_qutrit_life(g, steps=40)
        report = analyze_qutrit_life(history)

        # Must detect actual oscillatory behavior
        assert report.period_detected >= 2, (
            f"Expected oscillator period >= 2, got {report.period_detected}"
        )
        assert report.stability_label == "oscillatory"


# -----------------------------------------------------------------------
# Replay determinism
# -----------------------------------------------------------------------


class TestReplayDeterminism:
    def test_100_replay_determinism(self) -> None:
        """Run 100 independent evolutions of the same grid — all must match."""
        g = _grid_from_lists([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        reference = evolve_qutrit_life(g, steps=30)
        for _ in range(100):
            replay = evolve_qutrit_life(g, steps=30)
            assert replay == reference

    def test_deterministic_analysis(self) -> None:
        """Analysis report must be identical across replays."""
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        history = evolve_qutrit_life(g, steps=20)
        report1 = analyze_qutrit_life(history)
        report2 = analyze_qutrit_life(history)
        assert report1 == report2


# -----------------------------------------------------------------------
# Report structure
# -----------------------------------------------------------------------


class TestReport:
    def test_report_is_frozen(self) -> None:
        g = make_grid(3, 3, 0)
        history = evolve_qutrit_life(g, steps=2)
        report = analyze_qutrit_life(history)
        with pytest.raises(AttributeError):
            report.steps_evolved = 99  # type: ignore[misc]

    def test_report_fields(self) -> None:
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        history = evolve_qutrit_life(g, steps=10)
        report = analyze_qutrit_life(history)

        assert report.grid_shape == (3, 3)
        assert report.steps_evolved == 10
        assert len(report.state_counts) == 11  # initial + 10 steps
        assert isinstance(report.resonant_clusters, int)
        assert report.stability_label in (
            "fixed", "oscillatory", "emergent", "chaotic"
        )
        assert isinstance(report.period_detected, int)
        assert isinstance(report.entropy_score, float)

    def test_state_counts_sum(self) -> None:
        """State counts at each step must sum to grid size."""
        g = _grid_from_lists([
            [1, 0, 2],
            [0, 1, 0],
            [2, 0, 1],
        ])
        history = evolve_qutrit_life(g, steps=5)
        report = analyze_qutrit_life(history)
        total = 9
        for c0, c1, c2 in report.state_counts:
            assert c0 + c1 + c2 == total


# -----------------------------------------------------------------------
# History structure
# -----------------------------------------------------------------------


class TestHistory:
    def test_history_length(self) -> None:
        g = make_grid(3, 3, 0)
        history = evolve_qutrit_life(g, steps=7)
        assert len(history) == 8  # initial + 7

    def test_history_first_is_input(self) -> None:
        g = _grid_from_lists([[1, 0], [0, 2]])
        history = evolve_qutrit_life(g, steps=3)
        assert history[0] == g


# -----------------------------------------------------------------------
# Decoder untouched
# -----------------------------------------------------------------------


class TestDecoderUntouched:
    def test_no_decoder_imports(self) -> None:
        """The qutrit life automata module must not import decoder code."""
        import importlib
        import sys

        # Snapshot before import to isolate new modules
        mod_name = "qec.sims.qutrit_life_automata"
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

    def test_negative_cols_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_grid(3, -2)


# -----------------------------------------------------------------------
# History validation in analyze_qutrit_life
# -----------------------------------------------------------------------


class TestAnalysisHistoryValidation:
    def test_empty_history_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            analyze_qutrit_life(())

    def test_invalid_state_in_history_rejected(self) -> None:
        bad_grid = ((0, 1, 5),)
        with pytest.raises(ValueError, match="Invalid state"):
            analyze_qutrit_life((bad_grid,))

    def test_ragged_grid_in_history_rejected(self) -> None:
        ragged = ((0, 1), (0,))
        with pytest.raises(ValueError, match="length"):
            analyze_qutrit_life((ragged,))

    def test_mixed_shape_history_rejected(self) -> None:
        g1 = ((0, 0), (0, 0))
        g2 = ((0, 0, 0), (0, 0, 0))
        with pytest.raises(ValueError, match="shape"):
            analyze_qutrit_life((g1, g2))


# -----------------------------------------------------------------------
# Replay determinism after unchecked-step refactor
# -----------------------------------------------------------------------


class TestUncheckedStepReplay:
    def test_step_grid_matches_evolve(self) -> None:
        """step_grid (validated) and evolve (unchecked inner) must agree."""
        g = _grid_from_lists([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        # Single step via public API
        one_step = step_grid(g)
        # Same step via evolve history
        history = evolve_qutrit_life(g, steps=1)
        assert history[1] == one_step

    def test_evolve_replay_after_refactor(self) -> None:
        """Ensure evolve still produces deterministic results."""
        g = _grid_from_lists([
            [1, 0, 2, 0],
            [0, 1, 0, 1],
            [2, 0, 1, 0],
            [0, 1, 0, 2],
        ])
        ref = evolve_qutrit_life(g, steps=15)
        for _ in range(50):
            assert evolve_qutrit_life(g, steps=15) == ref
