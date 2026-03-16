"""
v9.2.0 — Discovery Engine.

Main loop for deterministic QLDPC structure discovery.  Combines
mutation, repair, spectral evaluation, multi-objective ranking,
novelty tracking, cycle-pressure guidance, spectral bad-edge
detection, ACE-gated mutation filtering, and incremental metric
updates for local structural metrics.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.qec.discovery.search_state import make_search_state
from src.qec.discovery.objectives import compute_discovery_objectives
from src.qec.discovery.mutation_operators import (
    mutate_tanner_graph,
    get_operator_for_generation,
)
from src.qec.discovery.repair_operators import (
    repair_tanner_graph,
    validate_tanner_graph,
)
from src.qec.discovery.novelty import (
    extract_feature_vector,
    compute_novelty_score,
)
from src.qec.discovery.archive import (
    create_archive,
    update_discovery_archive,
    get_archive_features,
    get_archive_summary,
)
from src.qec.discovery.cycle_pressure import compute_cycle_pressure
from src.qec.discovery.spectral_bad_edge import detect_bad_edges
from src.qec.discovery.ace_filter import ace_gate_mutation, compute_local_ace_score
from src.qec.discovery.incremental_metrics import update_metrics_incrementally
from src.qec.discovery.diversity import (
    compute_structure_signature,
    compute_diversity_penalty,
)
from src.qec.discovery.basin_switch_detector import BasinSwitchDetector
from src.qec.discovery.mutation_trust_region import SpectralTrustRegion
from src.qec.generation.tanner_graph_generator import generate_tanner_graph_candidates
from src.qec.analysis.spectral_gradient import estimate_spectral_gradient
from src.qec.analysis.spectral_trajectory import SpectralTrajectoryRecorder
from src.qec.analysis.non_backtracking_matrix import build_non_backtracking_matrix
from src.qec.analysis.non_backtracking_spectrum import leading_nb_eigenmode
from src.qec.discovery.nb_eigenmode_mutation import score_edges_by_eigenmode
from src.qec.discovery.spectral_gradient_mutation import propose_gradient_step
from src.qec.discovery.cooperative_region_planner import plan_agent_targets
from src.qec.discovery.agent_coordination import AgentCoordinationState
from src.qec.discovery.agent_messages import AgentMessage, FRONTIER_EXPLORED, REGION_EXPLORED
from src.qec.analysis.agent_spacing import enforce_agent_spacing
from src.qec.analysis.cooperative_metrics import (
    agent_region_overlap,
    agent_spacing_distance,
    cooperative_coverage,
    frontier_exploration_rate,
)
from src.qec.analysis.spectral_frontiers import detect_spectral_frontiers


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _rank_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank candidates by feasibility → dominance → composite → novelty.

    Parameters
    ----------
    candidates : list[dict[str, Any]]
        Search states with objectives and novelty.

    Returns
    -------
    list[dict[str, Any]]
        Candidates sorted best-first.
    """
    def sort_key(c: dict[str, Any]) -> tuple:
        obj = c.get("objectives", {})
        return (
            0 if c.get("is_feasible", True) else 1,
            c.get("dominance_rank", 0),
            obj.get("composite_score", float("inf")),
            -c.get("novelty", 0.0),
            c.get("candidate_id", ""),
        )

    return sorted(candidates, key=sort_key)


def run_structure_discovery(
    spec: dict[str, Any],
    num_generations: int = 10,
    population_size: int = 8,
    *,
    base_seed: int = 0,
    archive_top_k: int = 5,
    target_variable_degree: int | None = None,
    target_check_degree: int | None = None,
    use_nb_eigenmode_mutation: bool = False,
    enable_nb_eigenmode_mutation: bool = False,
    enable_spectral_trust_region: bool = False,
    trust_region_radius: float = 0.25,
    enable_basin_switch_detection: bool = False,
    basin_switch_threshold: float = 0.5,
    enable_spectral_trajectory: bool = False,
    trajectory_recorder: SpectralTrajectoryRecorder | None = None,
    enable_spectral_gradient: bool = False,
    gradient_step_size: float = 0.1,
    enable_cooperative_agents: bool = False,
    cooperative_min_distance: float = 0.3,
    enable_frontier_guidance: bool = False,
    frontier_distance_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the deterministic structure discovery engine.

    Loop:
    1. Initialize deterministic population.
    2. Evaluate spectral metrics.
    3. Compute objectives.
    4. Rank candidates.
    5. Update archive.
    6. Detect spectral bad edges.
    7. Compute cycle pressure.
    8. Mutate elites.
    9. ACE gate.
    10. Repair graphs.
    11. Evaluate children.
    12. Novelty filtering.
    13. Repeat.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification with keys: num_variables, num_checks,
        variable_degree, check_degree.
    num_generations : int
        Number of discovery generations.
    population_size : int
        Population size per generation.
    base_seed : int
        Base seed for all deterministic derivation.
    archive_top_k : int
        Number of elites per archive category.
    target_variable_degree : int or None
        Target variable degree for repair.
    target_check_degree : int or None
        Target check degree for repair.
    use_nb_eigenmode_mutation : bool
        Enable opt-in NB eigenmode mutation operator. Default False.
    enable_nb_eigenmode_mutation : bool
        Enable opt-in NB eigenmode edge scoring for mutation bias. Default False.
    enable_spectral_trust_region : bool
        Enable opt-in spectral trust-region mutation rejection. Default False.
    trust_region_radius : float
        Trust-region spectral radius when enabled.
    enable_basin_switch_detection : bool
        Enable opt-in basin switch detection in discovery logs. Default False.
    basin_switch_threshold : float
        Spectral jump threshold for basin switch detection.
    enable_spectral_trajectory : bool
        Enable opt-in spectral trajectory recording. Default False.
    trajectory_recorder : SpectralTrajectoryRecorder or None
        Optional externally managed recorder for trajectory capture.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``best_candidate`` : dict — best search state found
        - ``elite_history`` : list — best per generation
        - ``archive_summary`` : dict — final archive summary
        - ``generation_summaries`` : list — per-generation statistics
    """
    if target_variable_degree is None:
        target_variable_degree = spec.get("variable_degree")
    if target_check_degree is None:
        target_check_degree = spec.get("check_degree")

    trust_region = (
        SpectralTrustRegion(radius=trust_region_radius)
        if enable_spectral_trust_region
        else None
    )
    basin_detector = (
        BasinSwitchDetector(threshold=basin_switch_threshold)
        if enable_basin_switch_detection
        else None
    )
    if enable_spectral_trajectory and trajectory_recorder is None:
        trajectory_recorder = SpectralTrajectoryRecorder()

    coordination_state = AgentCoordinationState() if enable_cooperative_agents else None
    agent_assignment_history: list[dict[str, list[float] | None]] = []
    agent_spacing_history: list[float] = []
    cooperative_coverage_history: list[float] = []
    frontier_exploration_rate_history: list[float] = []
    agent_region_overlap_history: list[float] = []
    agent_messages: list[dict[str, Any]] = []

    # ── Step 1: Initialize population ──────────────────────────────
    init_seed = _derive_seed(base_seed, "init")
    raw_candidates = generate_tanner_graph_candidates(
        spec, population_size, base_seed=init_seed,
    )

    archive = create_archive(top_k=archive_top_k)
    elite_history: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    signature_archive: list[tuple[float, ...]] = []

    # Build initial population as search states
    population: list[dict[str, Any]] = []
    for i, rc in enumerate(raw_candidates):
        obj_seed = _derive_seed(base_seed, f"obj_init_{i}")
        objectives = compute_discovery_objectives(rc["H"], seed=obj_seed)
        validation = validate_tanner_graph(rc["H"])

        state = make_search_state(
            candidate_id=rc["candidate_id"],
            generation=0,
            parent_id=None,
            operator=None,
            H=rc["H"],
            metrics=objectives,
            objectives=objectives,
            novelty=0.0,
            dominance_rank=0,
            is_feasible=validation["is_valid"],
        )
        population.append(state)

    # Seed signature archive with initial population
    for state in population:
        sig = compute_structure_signature(state["H"])
        signature_archive.append(sig)

    # Initial ranking
    population = _rank_candidates(population)

    # Update archive with initial population
    archive = update_discovery_archive(archive, population)

    # Record generation 0
    best = population[0] if population else None
    if best:
        elite_history.append({
            "generation": 0,
            "candidate_id": best["candidate_id"],
            "composite_score": best["objectives"].get("composite_score", 0.0),
            "instability_score": best["objectives"].get("instability_score", 0.0),
        })
        generation_summaries.append(_make_generation_summary(0, population, archive))

    # ── Main loop ──────────────────────────────────────────────────
    for gen in range(1, num_generations + 1):
        gen_seed = _derive_seed(base_seed, f"gen_{gen}")

        # Select elites for mutation (top half)
        n_elites = max(1, len(population) // 2)
        elites = population[:n_elites]

        cooperative_assignments: dict[str, np.ndarray | None] = {}
        frontier_assignments: dict[str, np.ndarray | None] = {}
        frontier_regions: list[np.ndarray] = []
        if enable_cooperative_agents:
            agents = [
                SimpleNamespace(agent_id=str(elite.get("candidate_id", f"agent_{idx}")))
                for idx, elite in enumerate(elites)
            ]
            candidate_regions = [
                _objective_spectrum(elite.get("objectives", {}))
                for elite in elites
            ]
            memory = {
                "region_centers": candidate_regions,
                "candidate_regions": candidate_regions,
            }
            frontier_regions = detect_spectral_frontiers(memory, frontier_distance_threshold)
            if enable_frontier_guidance and frontier_regions:
                cooperative_assignments = plan_agent_targets(
                    agents,
                    {"region_centers": frontier_regions},
                )
                frontier_assignments = dict(cooperative_assignments)
            else:
                cooperative_assignments = plan_agent_targets(agents, memory)
            cooperative_assignments = enforce_agent_spacing(
                cooperative_assignments,
                min_distance=cooperative_min_distance,
            )
            ordered_assignment = {
                aid: (
                    None
                    if cooperative_assignments[aid] is None
                    else np.asarray(
                        cooperative_assignments[aid], dtype=np.float64,
                    ).tolist()
                )
                for aid in sorted(cooperative_assignments.keys())
            }
            explored_regions = [
                cooperative_assignments[aid]
                for aid in sorted(cooperative_assignments.keys())
                if cooperative_assignments[aid] is not None
            ]
            agent_assignment_history.append(ordered_assignment)
            agent_spacing_history.append(float(agent_spacing_distance(cooperative_assignments)))
            cooperative_coverage_history.append(
                float(cooperative_coverage(explored_regions, candidate_regions)),
            )
            frontier_exploration_rate_history.append(
                float(frontier_exploration_rate(explored_regions, frontier_regions)),
            )
            agent_region_overlap_history.append(float(agent_region_overlap(cooperative_assignments)))
            if coordination_state is not None:
                if frontier_assignments:
                    coordination_state.record_frontier_assignments(gen, frontier_assignments)
                for aid in sorted(cooperative_assignments.keys()):
                    region = cooperative_assignments[aid]
                    coordination_state.update_target(aid, region)
                    if region is None:
                        continue
                    coordination_state.record_history(aid, region)
                    message_type = REGION_EXPLORED
                    if aid in frontier_assignments and frontier_assignments.get(aid) is not None:
                        message_type = FRONTIER_EXPLORED
                    message = AgentMessage(
                        agent_id=aid,
                        message_type=message_type,
                        payload=np.asarray(region, dtype=np.float64).tolist(),
                        generation=gen,
                    ).to_dict()
                    agent_messages.append(message)
                    coordination_state.record_message(gen, message)

        # Detect spectral bad edges and cycle pressure on best
        best_H = elites[0]["H"]
        bad_edge_result = detect_bad_edges(best_H)
        cycle_result = compute_cycle_pressure(best_H)

        # Combine guidance: union of top bad edges and high-pressure edges
        n_guide = max(3, len(bad_edge_result["bad_edges"]) // 4)
        guide_edges = list(
            dict.fromkeys(
                bad_edge_result["bad_edges"][:n_guide]
                + cycle_result["ranked_edges"][:n_guide]
            )
        )

        # Optional NB-eigenmode guidance from dominant non-backtracking mode.
        nb_guide_edges: list[tuple[int, int]] = []
        if enable_nb_eigenmode_mutation:
            nb_guide_edges = _rank_tanner_edges_by_nb_mode(best_H)

        # Mutate elites
        children: list[dict[str, Any]] = []
        operator_name = (
            "nb_eigenmode_mutation"
            if use_nb_eigenmode_mutation
            else get_operator_for_generation(gen)
        )

        for ei, elite in enumerate(elites):
            mut_seed = _derive_seed(gen_seed, f"mutate_{ei}")

            gradient_target_spectrum = None
            agent_id = str(elite.get("candidate_id", f"agent_{ei}"))
            if enable_cooperative_agents and agent_id in cooperative_assignments:
                assigned = cooperative_assignments[agent_id]
                if assigned is not None:
                    gradient_target_spectrum = np.asarray(assigned, dtype=np.float64)
            if enable_spectral_gradient and trajectory_recorder is not None:
                current_spectrum = _objective_spectrum(elite.get("objectives", {}))
                gradient = estimate_spectral_gradient(trajectory_recorder.as_array())
                if gradient.shape == current_spectrum.shape:
                    gradient_target_spectrum = propose_gradient_step(
                        current_spectrum,
                        gradient,
                        step=gradient_step_size,
                    )

            H_mutated, op_used = mutate_tanner_graph(
                elite["H"],
                operator=operator_name,
                generation=gen,
                seed=mut_seed,
                target_edges=guide_edges,
                target_spectrum=gradient_target_spectrum,
            )

            # Find mutated edges for ACE evaluation and incremental updates
            diff = np.abs(H_mutated - elite["H"])
            mutated_edges = [
                (int(ci), int(vi))
                for ci in range(diff.shape[0])
                for vi in range(diff.shape[1])
                if diff[ci, vi] > 0.5
            ]

            # Attempt incremental metric update for quick ACE pre-check
            removed = [e for e in mutated_edges if elite["H"][e[0], e[1]] > 0.5]
            added = [e for e in mutated_edges if H_mutated[e[0], e[1]] > 0.5]
            mutation_info = {"removed_edges": removed, "added_edges": added}

            try:
                incremental = update_metrics_incrementally(
                    elite.get("objectives", {}), mutation_info,
                )
            except Exception:
                incremental = None

            # Full objective computation (spectral metrics require full recompute)
            # ACE gate
            parent_composite = elite["objectives"].get("composite_score", float("inf"))
            child_obj_seed = _derive_seed(gen_seed, f"child_obj_{ei}")
            child_objectives = compute_discovery_objectives(
                H_mutated, seed=child_obj_seed,
            )
            if trust_region is not None:
                old_spectrum = _objective_spectrum(elite.get("objectives", {}))
                new_spectrum = _objective_spectrum(child_objectives)
                if not trust_region.allow(old_spectrum, new_spectrum):
                    continue
            child_composite = child_objectives.get("composite_score", float("inf"))

            ace_result = ace_gate_mutation(
                elite["H"],
                H_mutated,
                composite_before=parent_composite,
                composite_after=child_composite,
                mutated_edges=mutated_edges if mutated_edges else None,
            )

            if not ace_result["accept"]:
                continue

            # Repair
            H_repaired, validation = repair_tanner_graph(
                H_mutated,
                target_variable_degree=target_variable_degree,
                target_check_degree=target_check_degree,
            )

            if not validation["is_valid"]:
                continue

            # Re-evaluate after repair
            repair_obj_seed = _derive_seed(gen_seed, f"repair_obj_{ei}")
            repaired_objectives = compute_discovery_objectives(
                H_repaired, seed=repair_obj_seed,
            )

            basin_switch = False
            if basin_detector is not None:
                prev_spectrum = _objective_spectrum(elite.get("objectives", {}))
                repaired_spectrum = _objective_spectrum(repaired_objectives)
                basin_switch = basin_detector.detect(prev_spectrum, repaired_spectrum)

            if enable_spectral_trajectory and trajectory_recorder is not None:
                current_spectrum = _objective_spectrum(repaired_objectives)
                nb_eigenvalue = _compute_nb_mode_magnitude(repaired_objectives, H_repaired)
                trajectory_recorder.record(np.append(current_spectrum, nb_eigenvalue))

            # Diversity penalty
            child_sig = compute_structure_signature(H_repaired)
            diversity_penalty = compute_diversity_penalty(
                child_sig, signature_archive,
            )
            repaired_objectives["composite_score"] = (
                repaired_objectives.get("composite_score", 0.0)
                + diversity_penalty
            )
            signature_archive.append(child_sig)

            # Novelty
            archive_features = get_archive_features(archive)
            fv = extract_feature_vector(repaired_objectives)
            novelty = compute_novelty_score(fv, archive_features)

            child_id = f"gen{gen:04d}_child{ei:04d}"
            child_state = make_search_state(
                candidate_id=child_id,
                generation=gen,
                parent_id=elite["candidate_id"],
                operator=op_used,
                H=H_repaired,
                metrics=repaired_objectives,
                objectives=repaired_objectives,
                novelty=novelty,
                dominance_rank=0,
                is_feasible=True,
            )
            if basin_detector is not None:
                child_state["basin_switch"] = basin_switch
            children.append(child_state)

        # Combine population: elites + children, re-rank, truncate
        combined = population + children
        combined = _rank_candidates(combined)
        population = combined[:population_size]

        # Update archive
        archive = update_discovery_archive(archive, children)

        # Record generation
        best = population[0] if population else None
        if best:
            elite_history.append({
                "generation": gen,
                "candidate_id": best["candidate_id"],
                "composite_score": best["objectives"].get("composite_score", 0.0),
                "instability_score": best["objectives"].get("instability_score", 0.0),
            })

        generation_summaries.append(
            _make_generation_summary(
                gen,
                population,
                archive,
                include_basin_switches=enable_basin_switch_detection,
            )
        )

    # ── Assemble result ────────────────────────────────────────────
    best_candidate = population[0] if population else None
    archive_summary = get_archive_summary(archive)

    result = {
        "best_candidate": _serialize_candidate(best_candidate) if best_candidate else None,
        "best_H": best_candidate["H"] if best_candidate else None,
        "elite_history": elite_history,
        "archive_summary": archive_summary,
        "generation_summaries": generation_summaries,
    }
    if enable_spectral_trajectory and trajectory_recorder is not None:
        result["spectral_trajectory"] = trajectory_recorder.as_array().tolist()
    if enable_cooperative_agents:
        result["agent_assignments"] = agent_assignment_history
        result["agent_spacing"] = [float(x) for x in agent_spacing_history]
        result["cooperative_coverage"] = [float(x) for x in cooperative_coverage_history]
        result["frontier_exploration_rate"] = [
            float(x) for x in frontier_exploration_rate_history
        ]
        result["agent_messages"] = list(agent_messages)
        result["coordination_state"] = (
            coordination_state.snapshot() if coordination_state is not None else {}
        )
        result["agent_region_overlap"] = [float(x) for x in agent_region_overlap_history]
    return result


def _make_generation_summary(
    generation: int,
    population: list[dict[str, Any]],
    archive: dict[str, Any],
    *,
    include_basin_switches: bool = False,
) -> dict[str, Any]:
    """Produce a summary for one generation."""
    feasible = [c for c in population if c.get("is_feasible", True)]
    novel = [c for c in population if c.get("novelty", 0.0) > 0.1]

    best = population[0] if population else None
    summary = get_archive_summary(archive)

    result = {
        "generation": generation,
        "best_candidate_id": best["candidate_id"] if best else "",
        "best_composite_score": (
            best["objectives"].get("composite_score", 0.0) if best else 0.0
        ),
        "best_instability_score": (
            best["objectives"].get("instability_score", 0.0) if best else 0.0
        ),
        "best_spectral_radius": (
            best["objectives"].get("spectral_radius", 0.0) if best else 0.0
        ),
        "best_bethe_margin": (
            best["objectives"].get("bethe_margin", 0.0) if best else 0.0
        ),
        "archive_size": summary.get("total_unique", 0),
        "num_feasible": len(feasible),
        "num_novel": len(novel),
    }
    if include_basin_switches:
        result["basin_switches"] = int(
            sum(1 for c in population if c.get("basin_switch", False))
        )
    return result



def _build_tanner_adjacency(H: np.ndarray) -> np.ndarray:
    """Build deterministic bipartite adjacency from parity-check matrix."""
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    zero_mm = np.zeros((m, m), dtype=np.float64)
    zero_nn = np.zeros((n, n), dtype=np.float64)
    top = np.hstack((zero_mm, H_arr))
    bottom = np.hstack((H_arr.T, zero_nn))
    return np.vstack((top, bottom)).astype(np.float64, copy=False)


def _rank_tanner_edges_by_nb_mode(H: np.ndarray) -> list[tuple[int, int]]:
    """Rank Tanner edges by dominant NB eigenmode magnitude (stable order)."""
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    if m == 0 or n == 0:
        return []

    adj = _build_tanner_adjacency(H_arr)
    B, directed_edges = build_non_backtracking_matrix(adj)
    if B.size == 0:
        return []

    _, eigvec = leading_nb_eigenmode(B)
    edge_scores = score_edges_by_eigenmode(directed_edges, eigvec)

    undirected_scores: dict[tuple[int, int], float] = {}
    for (u, v), score in edge_scores.items():
        if u < m and v >= m:
            key = (int(u), int(v - m))
        elif v < m and u >= m:
            key = (int(v), int(u - m))
        else:
            continue
        prev = undirected_scores.get(key, 0.0)
        if score > prev:
            undirected_scores[key] = float(score)

    return [
        edge
        for edge, _ in sorted(
            undirected_scores.items(),
            key=lambda item: (-float(item[1]), item[0][0], item[0][1]),
        )
    ]


def _compute_nb_mode_magnitude(objectives: dict[str, Any], H: np.ndarray) -> float:
    """Compute deterministic NB dominant eigenvalue magnitude for trajectory."""
    del objectives
    adj = _build_tanner_adjacency(np.asarray(H, dtype=np.float64))
    B, _ = build_non_backtracking_matrix(adj)
    if B.size == 0:
        return 0.0
    eigval, _ = leading_nb_eigenmode(B)
    return float(np.abs(eigval))

def _objective_spectrum(objectives: dict[str, Any]) -> np.ndarray:
    """Build deterministic spectral feature vector for trust/basin checks."""
    return np.asarray(
        [
            float(objectives.get("spectral_radius", 0.0)),
            float(objectives.get("bethe_margin", 0.0)),
            float(objectives.get("ipr_localization", 0.0)),
            float(objectives.get("entropy", 0.0)),
        ],
        dtype=np.float64,
    )


def _serialize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Convert a candidate to a JSON-safe dict (without H matrix)."""
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "generation": candidate.get("generation", 0),
        "parent_id": candidate.get("parent_id"),
        "operator": candidate.get("operator"),
        "objectives": candidate.get("objectives", {}),
        "novelty": candidate.get("novelty", 0.0),
        "dominance_rank": candidate.get("dominance_rank", 0),
        "is_feasible": candidate.get("is_feasible", True),
    }
