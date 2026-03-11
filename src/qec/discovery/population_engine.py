"""
v11.0.0 — Population Discovery Engine.

Deterministic evolutionary search for high-performance LDPC/QLDPC
parity-check matrices.  Uses tournament selection, elitism, and
guided mutations with spectral fitness evaluation.

v11 extension: optional decoder-aware mode incorporates trapping-set
analysis, BP stability probes, and Jacobian spectral radius into
fitness evaluation.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from src.qec.fitness.fitness_engine import FitnessEngine
from src.qec.discovery.guided_mutations import apply_guided_mutation
from src.qec.discovery.repair_operators import (
    repair_tanner_graph,
    validate_tanner_graph,
)
from src.qec.generation.tanner_graph_generator import generate_tanner_graph_candidates
from src.qec.generation.peg_generator import generate_peg_population


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _matrix_id(H: np.ndarray) -> str:
    """Compute a deterministic content-addressable ID for a matrix."""
    data = np.asarray(H, dtype=np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


class DiscoveryEngine:
    """Deterministic evolutionary search for LDPC/QLDPC codes.

    Parameters
    ----------
    population_size : int
        Number of candidates per generation.
    generations : int
        Number of evolutionary generations to run.
    seed : int
        Base seed for all deterministic derivation.
    archive_path : str
        Path for persistent archive storage.
    decoder_aware : bool
        Enable decoder-aware fitness evaluation (default False).
    bp_trials : int
        Number of BP probe trials when decoder-aware (default 50).
    bp_iterations : int
        Max BP iterations per probe trial (default 10).
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 500,
        seed: int = 42,
        archive_path: str = "./discovery_archive",
        decoder_aware: bool = False,
        bp_trials: int = 50,
        bp_iterations: int = 10,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        self.archive_path = archive_path
        self.decoder_aware = decoder_aware
        self._fitness_engine = FitnessEngine(
            decoder_aware=decoder_aware,
            bp_trials=bp_trials,
            bp_iterations=bp_iterations,
            seed=seed,
        )
        self._population: list[dict[str, Any]] = []
        self._archive: list[dict[str, Any]] = []
        self._generation: int = 0
        self._elite_history: list[dict[str, Any]] = []
        self._generation_summaries: list[dict[str, Any]] = []

    def run(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Run the full discovery loop.

        Parameters
        ----------
        spec : dict[str, Any]
            Generation specification with keys: num_variables, num_checks,
            variable_degree, check_degree.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``best`` : dict — best candidate found
            - ``best_H`` : np.ndarray — best parity-check matrix
            - ``elite_history`` : list — best fitness per generation
            - ``archive`` : list — final elite archive
            - ``generation_summaries`` : list — per-generation stats
        """
        self._spec = spec
        self._fitness_engine.clear_cache()

        # Step 1: Initialize population
        self.initialize_population(spec)

        # Step 2: Main loop
        for gen in range(1, self.generations + 1):
            self._generation = gen
            gen_seed = _derive_seed(self.seed, f"gen_{gen}")

            # Evaluate population
            self.evaluate_population()

            # Update archive
            self.update_archive()

            # Record generation summary
            self._record_summary(gen)

            # Select parents
            parents = self.select_parents(gen_seed)

            # Apply mutations
            children = self.apply_mutations(parents, gen_seed)

            # Preserve elite
            self.preserve_elite(children)

        # Final evaluation
        self.evaluate_population()
        self.update_archive()
        self._record_summary(self._generation)

        # Assemble result
        best = self._population[0] if self._population else None

        return {
            "best": _serialize_entry(best) if best else None,
            "best_H": best["H"] if best else None,
            "elite_history": self._elite_history,
            "archive": [_serialize_entry(e) for e in self._archive],
            "generation_summaries": self._generation_summaries,
        }

    def initialize_population(self, spec: dict[str, Any]) -> None:
        """Create the initial population from the specification.

        Parameters
        ----------
        spec : dict[str, Any]
            Graph generation specification.  If ``init_strategy`` is
            ``"peg"``, uses PEG-seeded generation.  Otherwise uses the
            default random generator.
        """
        init_seed = _derive_seed(self.seed, "init_pop")

        if spec.get("init_strategy") == "peg":
            raw = generate_peg_population(
                spec, self.population_size, base_seed=init_seed,
            )
        else:
            raw = generate_tanner_graph_candidates(
                spec, self.population_size, base_seed=init_seed,
            )

        self._population = []
        for i, rc in enumerate(raw):
            entry = {
                "code_id": _matrix_id(rc["H"]),
                "H": np.asarray(rc["H"], dtype=np.float64),
                "fitness": None,
                "metrics": {},
                "parent_id": None,
                "operator": None,
                "generation": 0,
            }
            self._population.append(entry)

    def evaluate_population(self) -> None:
        """Evaluate fitness for all candidates in the population."""
        for entry in self._population:
            if entry["fitness"] is None:
                result = self._fitness_engine.evaluate(entry["H"])
                entry["fitness"] = result["composite"]
                entry["metrics"] = result["metrics"]

        # Sort by fitness descending (higher = better)
        self._population.sort(
            key=lambda e: (-e.get("fitness", float("-inf")), e.get("code_id", "")),
        )

    def update_archive(self) -> None:
        """Update the elite archive with the current population."""
        archive_size = max(5, self.population_size // 5)

        # Combine archive and top of population
        combined = list(self._archive) + self._population[:archive_size]

        # Deduplicate by code_id, keeping highest fitness
        seen: dict[str, dict[str, Any]] = {}
        for entry in combined:
            cid = entry.get("code_id", "")
            existing = seen.get(cid)
            if existing is None or (entry.get("fitness") or 0) > (existing.get("fitness") or 0):
                seen[cid] = entry

        self._archive = sorted(
            seen.values(),
            key=lambda e: (-e.get("fitness", float("-inf")), e.get("code_id", "")),
        )[:archive_size]

    def select_parents(self, gen_seed: int) -> list[dict[str, Any]]:
        """Select parents via deterministic tournament selection.

        Parameters
        ----------
        gen_seed : int
            Seed for this generation.

        Returns
        -------
        list[dict[str, Any]]
            Selected parent candidates.
        """
        rng = np.random.RandomState(_derive_seed(gen_seed, "select"))
        n_parents = max(2, self.population_size // 2)
        tournament_size = min(3, len(self._population))
        parents = []

        for i in range(n_parents):
            # Deterministic tournament: pick tournament_size candidates
            indices = rng.choice(
                len(self._population), size=tournament_size, replace=False,
            )
            candidates = [self._population[idx] for idx in indices]
            # Best fitness wins
            winner = max(
                candidates,
                key=lambda e: (e.get("fitness") or float("-inf"), e.get("code_id", "")),
            )
            parents.append(winner)

        return parents

    def apply_mutations(
        self,
        parents: list[dict[str, Any]],
        gen_seed: int,
    ) -> list[dict[str, Any]]:
        """Apply guided mutations to parent candidates.

        Parameters
        ----------
        parents : list[dict[str, Any]]
            Parent candidates to mutate.
        gen_seed : int
            Seed for this generation.

        Returns
        -------
        list[dict[str, Any]]
            Mutated child candidates.
        """
        children = []

        for i, parent in enumerate(parents):
            mut_seed = _derive_seed(gen_seed, f"mutate_{i}")
            operators = [
                "spectral_edge_pressure",
                "cycle_pressure",
                "ace_repair",
                "girth_preserving_rewire",
                "expansion_driven_rewire",
                "ipr_trapping_pressure",
                "trapping_set_pressure",
                "residual_guided",
            ]
            operator_idx = (self._generation + i) % len(operators)
            operator = operators[operator_idx]

            H_mutated = apply_guided_mutation(
                parent["H"],
                operator=operator,
                seed=mut_seed,
            )

            # Repair if needed
            target_vd = self._spec.get("variable_degree")
            target_cd = self._spec.get("check_degree")
            H_repaired, validation = repair_tanner_graph(
                H_mutated,
                target_variable_degree=target_vd,
                target_check_degree=target_cd,
            )

            if not validation["is_valid"]:
                continue

            child = {
                "code_id": _matrix_id(H_repaired),
                "H": H_repaired,
                "fitness": None,
                "metrics": {},
                "parent_id": parent.get("code_id"),
                "operator": operator,
                "generation": self._generation,
            }
            children.append(child)

        return children

    def preserve_elite(self, children: list[dict[str, Any]]) -> None:
        """Merge children into population, preserving top 10% elite.

        Parameters
        ----------
        children : list[dict[str, Any]]
            New child candidates.
        """
        # Top 10% survive unchanged
        n_elite = max(1, self.population_size // 10)
        elites = self._population[:n_elite]

        # Evaluate children
        for child in children:
            if child["fitness"] is None:
                result = self._fitness_engine.evaluate(child["H"])
                child["fitness"] = result["composite"]
                child["metrics"] = result["metrics"]

        # Combine: elites + remaining population + children
        combined = elites + self._population[n_elite:] + children

        # Deduplicate by code_id
        seen: dict[str, dict[str, Any]] = {}
        for entry in combined:
            cid = entry.get("code_id", "")
            existing = seen.get(cid)
            if existing is None or (entry.get("fitness") or 0) > (existing.get("fitness") or 0):
                seen[cid] = entry

        # Sort and truncate
        self._population = sorted(
            seen.values(),
            key=lambda e: (-e.get("fitness", float("-inf")), e.get("code_id", "")),
        )[:self.population_size]

    def _record_summary(self, generation: int) -> None:
        """Record a generation summary."""
        if not self._population:
            return

        best = self._population[0]
        fitnesses = [e.get("fitness", 0.0) or 0.0 for e in self._population]

        summary = {
            "generation": generation,
            "best_fitness": best.get("fitness", 0.0),
            "best_code_id": best.get("code_id", ""),
            "mean_fitness": round(float(np.mean(fitnesses)), _ROUND),
            "std_fitness": round(float(np.std(fitnesses)), _ROUND),
            "population_size": len(self._population),
            "archive_size": len(self._archive),
        }
        self._generation_summaries.append(summary)

        self._elite_history.append({
            "generation": generation,
            "code_id": best.get("code_id", ""),
            "fitness": best.get("fitness", 0.0),
        })


def _serialize_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a population entry to a JSON-safe dict."""
    if entry is None:
        return None
    return {
        "code_id": entry.get("code_id", ""),
        "fitness": entry.get("fitness"),
        "metrics": entry.get("metrics", {}),
        "parent_id": entry.get("parent_id"),
        "operator": entry.get("operator"),
        "generation": entry.get("generation", 0),
    }
