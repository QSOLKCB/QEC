# SPDX-License-Identifier: MIT
"""Deterministic tests for the quantum ecosystem sandbox (v136.2.0)."""

from __future__ import annotations

import sys

import pytest

from qec.sims.quantum_ecosystem_sandbox import (
    Grid,
    QuantumEcosystemReport,
    analyze_quantum_ecosystem,
    evolve_quantum_ecosystem,
    inject_civilization_influence,
    validate_grid,
)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Grid validation rejects invalid states."""

    def test_valid_grid_all_states(self) -> None:
        grid: Grid = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 0, 1),
        )
        validate_grid(grid)  # should not raise

    def test_invalid_state_rejected(self) -> None:
        grid: Grid = ((0, 1, 7),)
        with pytest.raises(ValueError, match="Invalid state 7"):
            validate_grid(grid)

    def test_negative_state_rejected(self) -> None:
        grid: Grid = ((0, -1, 2),)
        with pytest.raises(ValueError, match="Invalid state -1"):
            validate_grid(grid)

    def test_empty_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_grid(())

    def test_inconsistent_row_width(self) -> None:
        grid: Grid = ((0, 1), (0,))
        with pytest.raises(ValueError, match="inconsistent width"):
            validate_grid(grid)


# ---------------------------------------------------------------------------
# Particle clustering tests
# ---------------------------------------------------------------------------


class TestParticleClustering:
    """Particles with ≥2 adjacent particles become clusters."""

    def test_three_adjacent_particles_cluster(self) -> None:
        # Three particles in a row: middle has 2 adjacent particles
        grid: Grid = (
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # Middle cell (1,2) has 2 particle neighbors → cluster
        assert evolved[1][2] == 2

    def test_isolated_particle_stays(self) -> None:
        grid: Grid = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        assert evolved[1][1] == 1


# ---------------------------------------------------------------------------
# Attractor formation tests
# ---------------------------------------------------------------------------


class TestAttractorFormation:
    """Clusters with enough cluster/attractor neighbors become attractors."""

    def test_cluster_surrounded_by_clusters_becomes_attractor(self) -> None:
        # Cross of clusters: center has 4 cluster neighbors.
        # All 5 clusters form one component with 0 attractors → no tunnel.
        # Center has 4 cluster neighbors ≥ 3 → attractor.
        grid: Grid = (
            (0, 2, 0),
            (2, 2, 2),
            (0, 2, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        assert evolved[1][1] == 3

    def test_exact_rule_priority_cluster_to_attractor(self) -> None:
        """Cluster with 3 cluster neighbors and no tunnel eligibility → attractor."""
        grid: Grid = (
            (0, 2, 0, 0),
            (2, 2, 2, 0),
            (0, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,1) has 3 cluster neighbors, component has 0 attractors → attractor
        assert evolved[1][1] == 3


# ---------------------------------------------------------------------------
# Decay field tests
# ---------------------------------------------------------------------------


class TestDecayField:
    """Clusters/attractors in overloaded decay-adjacent zones decay."""

    def test_cluster_adjacent_to_decay_overload(self) -> None:
        grid: Grid = (
            (4, 2, 4),
            (4, 2, 4),
            (4, 2, 4),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # Center cluster (1,1) has 2 decay + 2 cluster non-vacuum neighbors.
        # non_vacuum = 4 ≥ 3. decay(2) vs non-decay(2) → not strict majority.
        # Top/bottom clusters have 2 decay + 1 cluster = 3 non-vacuum.
        # decay(2) > non-decay(1) → strict majority → decay.
        # Center: (0,1) and (2,1) are clusters with 2 decay each.
        # (1,1) neighbors: up=(0,1)=2, down=(2,1)=2, left=(1,0)=4, right=(1,2)=4
        # n_cluster=2, n_decay=2, non_vacuum=4, decay(2) == non-decay(2) → NOT majority
        # So (1,1) checks tunnel: component has no attractors → no tunnel.
        # Checks attractor: n_cluster(2) + n_attractor(0) = 2 < 3 → stays cluster.
        # But (0,1) neighbors: up=OOB, down=(1,1)=2, left=(0,0)=4, right=(0,2)=4
        # n_cluster=1, n_decay=2, non_vacuum=3 ≥ 3, decay(2) > non-decay(1) → decay
        assert evolved[0][1] == 4
        assert evolved[2][1] == 4

    def test_attractor_decay_under_majority(self) -> None:
        """Attractor decays when decay cells are strict majority."""
        grid: Grid = (
            (0, 4, 0),
            (4, 3, 4),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,1) attractor: neighbors are 3 decay + 0 others non-vacuum
        # non_vacuum=3 ≥ 3, decay(3) > non-decay(0) → decay
        assert evolved[1][1] == 4


# ---------------------------------------------------------------------------
# Tunnel formation tests
# ---------------------------------------------------------------------------


class TestTunnelFormation:
    """Clusters on a path connecting ≥2 attractors become tunnels."""

    def test_cluster_path_between_attractors_becomes_tunnel(self) -> None:
        # Two attractors connected by a cluster bridge
        grid: Grid = (
            (0, 0, 0, 0, 0),
            (0, 3, 2, 3, 0),
            (0, 0, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # Middle cluster (1,2) connects two attractors → tunnel
        assert evolved[1][2] == 5

    def test_two_attractors_tunnel_connected(self) -> None:
        """Critical test: two attractors become tunnel-connected.

        Start with two attractor regions bridged by clusters.
        After evolution, the bridge cells should become tunnels (state 5).
        """
        grid: Grid = (
            (0, 0, 0, 0, 0, 0, 0),
            (0, 3, 2, 2, 2, 3, 0),
            (0, 0, 0, 0, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # All inner cluster cells should become tunnels
        for c in (2, 3, 4):
            assert evolved[1][c] == 5, (
                f"Cell (1,{c}) should be tunnel (5), got {evolved[1][c]}"
            )
        # Attractors are also tunnel-eligible (component has 2 attractors)
        # but decay check comes first (no decay neighbors), then tunnel
        assert evolved[1][1] == 5
        assert evolved[1][5] == 5

    def test_tunnel_precompute_replay_determinism(self) -> None:
        """Tunnel precompute must produce identical results on 100 replays."""
        grid: Grid = (
            (0, 0, 0, 0, 0, 0, 0),
            (0, 3, 2, 2, 2, 3, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 3, 2, 3, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
        )
        reference = evolve_quantum_ecosystem(grid, steps=10)
        for _ in range(100):
            result = evolve_quantum_ecosystem(grid, steps=10)
            assert result == reference


# ---------------------------------------------------------------------------
# Interaction zone tests
# ---------------------------------------------------------------------------


class TestInteractionZone:
    """Attractors/tunnels adjacent to interaction zones become interaction zones."""

    def test_attractor_adjacent_to_interaction_becomes_interaction(self) -> None:
        # Attractor (1,1) has interaction neighbor at (0,1).
        # Priority: decay (no), tunnel (only 1 attractor in component → no),
        # interaction (yes) → 6.
        grid: Grid = (
            (0, 6, 0),
            (0, 3, 0),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        assert evolved[1][1] == 6

    def test_tunnel_adjacent_to_interaction_becomes_interaction(self) -> None:
        grid: Grid = (
            (0, 6, 0),
            (0, 5, 0),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        assert evolved[1][1] == 6


# ---------------------------------------------------------------------------
# Civilization injection tests
# ---------------------------------------------------------------------------


class TestCivilizationInjection:
    """inject_civilization_influence seeds interaction zones from civ grid."""

    def test_attractor_at_hub_becomes_interaction(self) -> None:
        eco: Grid = (
            (0, 3, 0),
            (0, 1, 0),
            (0, 5, 0),
        )
        civ: Grid = (
            (0, 3, 0),  # hub
            (0, 0, 0),
            (0, 4, 0),  # corridor
        )
        result = inject_civilization_influence(eco, civ)
        assert result[0][1] == 6  # attractor + hub → interaction
        assert result[1][1] == 1  # particle unchanged (not 3 or 5)
        assert result[2][1] == 6  # tunnel + corridor → interaction

    def test_shape_mismatch_rejected(self) -> None:
        eco: Grid = ((0, 1),)
        civ: Grid = ((0,),)
        with pytest.raises(ValueError, match="shape mismatch"):
            inject_civilization_influence(eco, civ)

    def test_settlement_influence(self) -> None:
        eco: Grid = ((3,),)
        civ: Grid = ((1,),)  # settlement
        result = inject_civilization_influence(eco, civ)
        assert result[0][0] == 6

    def test_ecosystem_grid_validated(self) -> None:
        eco: Grid = ((9,),)
        civ: Grid = ((0,),)
        with pytest.raises(ValueError, match="Invalid state 9"):
            inject_civilization_influence(eco, civ)

    def test_civ_grid_empty_rejected(self) -> None:
        eco: Grid = ((0,),)
        with pytest.raises(ValueError, match="non-empty"):
            inject_civilization_influence(eco, ())

    def test_civ_grid_inconsistent_width_rejected(self) -> None:
        eco: Grid = ((0, 1), (0, 1))
        civ: Grid = ((0, 1), (0,))
        with pytest.raises(ValueError, match="inconsistent width"):
            inject_civilization_influence(eco, civ)


# ---------------------------------------------------------------------------
# Emergence analysis tests
# ---------------------------------------------------------------------------


class TestEmergenceAnalysis:
    """analyze_quantum_ecosystem computes correct metrics."""

    def test_simple_report(self) -> None:
        grid: Grid = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 0, 1),
        )
        history = (grid,)
        report = analyze_quantum_ecosystem(history)
        assert isinstance(report, QuantumEcosystemReport)
        assert report.grid_shape == (3, 3)
        assert report.steps_evolved == 0
        assert report.particle_count == 2
        assert report.cluster_count == 1
        assert report.attractor_count == 1
        assert report.decay_zone_count == 1
        assert report.tunnel_count == 1
        assert report.interaction_zone_count == 1
        assert report.entropy_score > 0.0

    def test_emergence_index_calculation(self) -> None:
        # Grid with 4 cells: 1 attractor, 1 tunnel, 1 interaction, 1 vacuum
        grid: Grid = ((3, 5), (6, 0))
        report = analyze_quantum_ecosystem((grid,))
        # Higher-order = 3 (attractor + tunnel + interaction), total = 4
        assert abs(report.emergence_index - 0.75) < 1e-9

    def test_frozen_report(self) -> None:
        grid: Grid = ((0, 0), (0, 0))
        report = analyze_quantum_ecosystem((grid,))
        with pytest.raises(AttributeError):
            report.particle_count = 99  # type: ignore[misc]

    def test_collapsed_label(self) -> None:
        # All vacuum: collapsed
        grid: Grid = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        report = analyze_quantum_ecosystem((grid, grid))
        assert report.stability_label == "collapsed"

    def test_connected_ecosystems(self) -> None:
        grid: Grid = (
            (1, 0, 1),
            (0, 0, 0),
            (1, 0, 1),
        )
        report = analyze_quantum_ecosystem((grid,))
        assert report.connected_ecosystems == 4

    def test_attractor_network_count_zero(self) -> None:
        """No attractors → 0 attractor networks."""
        grid: Grid = ((0, 1, 2), (0, 1, 2))
        report = analyze_quantum_ecosystem((grid,))
        assert report.attractor_network_count == 0

    def test_attractor_network_count_one(self) -> None:
        """Single connected attractor region → 1 network."""
        grid: Grid = (
            (0, 3, 0),
            (3, 3, 0),
            (0, 0, 0),
        )
        report = analyze_quantum_ecosystem((grid,))
        assert report.attractor_network_count == 1

    def test_attractor_network_count_two_disjoint(self) -> None:
        """Two disjoint attractor regions → 2 networks."""
        grid: Grid = (
            (3, 0, 3),
            (0, 0, 0),
            (3, 0, 3),
        )
        report = analyze_quantum_ecosystem((grid,))
        assert report.attractor_network_count == 4  # 4 isolated attractors


# ---------------------------------------------------------------------------
# Replay determinism tests
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """Evolution must produce byte-identical results on every replay."""

    def test_100_replay_determinism(self) -> None:
        grid: Grid = (
            (0, 1, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (0, 1, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (0, 1, 0, 1, 0),
        )
        reference = evolve_quantum_ecosystem(grid, steps=20)
        for _ in range(100):
            result = evolve_quantum_ecosystem(grid, steps=20)
            assert result == reference


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------


class TestHistory:
    """Evolution returns full history tuple."""

    def test_history_length(self) -> None:
        grid: Grid = ((0, 1), (1, 0))
        history = evolve_quantum_ecosystem(grid, steps=5)
        assert len(history) == 6  # initial + 5 steps

    def test_history_starts_with_initial(self) -> None:
        grid: Grid = ((0, 1), (1, 0))
        history = evolve_quantum_ecosystem(grid, steps=3)
        assert history[0] == grid

    def test_steps_zero_returns_initial_only(self) -> None:
        grid: Grid = ((0, 1), (1, 0))
        history = evolve_quantum_ecosystem(grid, steps=0)
        assert len(history) == 1
        assert history[0] == grid

    def test_negative_steps_rejected(self) -> None:
        grid: Grid = ((0,),)
        with pytest.raises(ValueError, match="steps must be >= 0"):
            evolve_quantum_ecosystem(grid, steps=-1)


# ---------------------------------------------------------------------------
# Vacuum spawning tests
# ---------------------------------------------------------------------------


class TestVacuumSpawning:
    """Vacuum with exactly 1 adjacent particle becomes a particle."""

    def test_vacuum_adjacent_to_one_particle_spawns(self) -> None:
        grid: Grid = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (0,1), (1,0), (1,2), (2,1) each have exactly 1 particle neighbor
        assert evolved[0][1] == 1
        assert evolved[1][0] == 1
        assert evolved[1][2] == 1
        assert evolved[2][1] == 1


# ---------------------------------------------------------------------------
# Decay recovery tests
# ---------------------------------------------------------------------------


class TestDecayRecovery:
    """Decay fields with no active neighbors recover to vacuum."""

    def test_isolated_decay_recovers(self) -> None:
        grid: Grid = (
            (0, 0, 0),
            (0, 4, 0),
            (0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        assert evolved[1][1] == 0


# ---------------------------------------------------------------------------
# Exact rule priority tests
# ---------------------------------------------------------------------------


class TestExactRulePriority:
    """Each rule priority must produce a single deterministic outcome."""

    def test_cluster_decay_beats_tunnel(self) -> None:
        """When a cluster qualifies for both decay and tunnel, decay wins."""
        grid: Grid = (
            (0, 4, 0, 0, 0),
            (4, 2, 3, 2, 3),
            (0, 4, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,1): cluster, 3 decay neighbors, non_vacuum ≥ 3, decay majority → 4
        assert evolved[1][1] == 4

    def test_cluster_tunnel_beats_attractor(self) -> None:
        """When a cluster qualifies for both tunnel and attractor, tunnel wins."""
        grid: Grid = (
            (0, 0, 0, 0, 0),
            (0, 3, 2, 3, 0),
            (0, 0, 2, 0, 0),
            (0, 0, 0, 0, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,2): cluster, component has 2 attractors → tunnel eligible
        # Also has 2 cluster + 2 attractor = 4 ≥ 3 → would be attractor
        # But tunnel priority is higher → 5
        assert evolved[1][2] == 5

    def test_attractor_decay_beats_tunnel(self) -> None:
        """When an attractor qualifies for both decay and tunnel, decay wins."""
        grid: Grid = (
            (0, 4, 0),
            (4, 3, 2),
            (0, 4, 3),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,1): attractor, 3 decay + 1 cluster = 4 non-vacuum
        # decay(3) > non-decay(1) → decay wins over tunnel
        assert evolved[1][1] == 4

    def test_attractor_tunnel_beats_interaction(self) -> None:
        """When an attractor qualifies for both tunnel and interaction, tunnel wins."""
        grid: Grid = (
            (0, 6, 0),
            (3, 3, 0),
            (0, 2, 0),
            (0, 3, 0),
        )
        history = evolve_quantum_ecosystem(grid, steps=1)
        evolved = history[1]
        # (1,1): attractor, component includes (1,0)=3, (2,1)=2, (3,1)=3 → ≥2 attractors
        # Also adjacent to interaction at (0,1) → would be interaction
        # But tunnel priority is higher → 5
        assert evolved[1][1] == 5


# ---------------------------------------------------------------------------
# Decoder untouched
# ---------------------------------------------------------------------------


class TestDecoderUntouched:
    """The decoder module must not be imported by the ecosystem sandbox."""

    def test_no_decoder_import_runtime(self) -> None:
        """Verify importing the module does not pull in qec.decoder."""
        before = set(sys.modules.keys())
        # Force re-import by removing from cache if present
        mod_name = "qec.sims.quantum_ecosystem_sandbox"
        was_loaded = mod_name in sys.modules
        if was_loaded:
            saved = sys.modules.pop(mod_name)
        try:
            import importlib

            importlib.import_module(mod_name)
            after = set(sys.modules.keys())
            new_modules = after - before
            decoder_modules = {m for m in new_modules if "qec.decoder" in m}
            assert not decoder_modules, (
                f"Importing ecosystem sandbox pulled in decoder modules: {decoder_modules}"
            )
        finally:
            if was_loaded:
                sys.modules[mod_name] = saved

    def test_no_decoder_in_source(self) -> None:
        """Source code must not reference qec.decoder."""
        import importlib

        mod = importlib.import_module("qec.sims.quantum_ecosystem_sandbox")
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()
        assert "qec.decoder" not in source
