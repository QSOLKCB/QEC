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
from src.qec.analysis.basin_stagnation import detect_basin_stagnation
from src.qec.analysis.basin_escape_direction import estimate_escape_direction
from src.qec.analysis.spectral_trajectory import SpectralTrajectoryRecorder
from src.qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from src.qec.analysis.landscape_metrics import novelty_score as landscape_novelty_score, landscape_coverage
from src.qec.analysis.spectral_basins import identify_spectral_basins
from src.qec.analysis.basin_transitions import detect_basin_transitions
from src.qec.analysis.basin_statistics import basin_sizes
from src.qec.analysis.basin_map_export import export_basin_map
from src.qec.analysis.non_backtracking_matrix import build_non_backtracking_matrix
from src.qec.analysis.non_backtracking_spectrum import leading_nb_eigenmode
from src.qec.discovery.nb_eigenmode_mutation import score_edges_by_eigenmode
from src.qec.discovery.spectral_gradient_mutation import propose_gradient_step
from src.qec.discovery.discovery_agent import DiscoveryAgent
from src.qec.discovery.multi_agent_coordinator import MultiAgentCoordinator
from src.qec.discovery.autonomous_scheduler import schedule_next_experiment
from src.qec.discovery.experiment_queue import ExperimentQueue
from src.qec.analysis.exploration_state import analyze_exploration_state
from src.qec.analysis.exploration_metrics import (
    basin_switch_rate,
    exploration_entropy,
    mean_basin_duration,
)
from src.qec.discovery.exploration_policy import (
    apply_escape_feedback_bias,
    choose_exploration_strategy,
)
from src.qec.discovery.basin_escape_mutation import propose_escape_step


_ROUND = 12
_ESCAPE_OPERATORS = [
    "edge_swap",
    "cycle_break",
    "seeded_reconstruction",
    "spectral_pressure_guided_mutation",
]


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
    enable_multi_agent_discovery: bool = False,
    num_agents: int = 4,
    landscape_regions: list[list[float]] | None = None,
    enable_autonomous_scheduler: bool = False,
    scheduler_gap_radius: float = 0.3,
    scheduler_max_gaps: int = 16,
    scheduler_queue: ExperimentQueue | None = None,
    enable_landscape_learning: bool = False,
    landscape_memory: SpectralLandscapeMemory | None = None,
    landscape_cluster_threshold: float = 0.25,
    use_landscape_novelty_bias: bool = False,
    landscape_novelty_weight: float = 0.0,
    novelty_weight: float | None = None,
    enable_adaptive_exploration: bool = False,
    exploration_window: int = 10,
    escape_base_step: float = 0.01,
    early_exploration_rate_threshold: float = 0.5,
    low_escape_success_threshold: float = 0.1,
    enable_basin_escape: bool = False,
    basin_escape_window: int = 10,
    basin_escape_step: float = 0.3,
    enable_basin_topology_mapping: bool = False,
    basin_distance_threshold: float = 0.25,
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
    enable_multi_agent_discovery : bool
        Enable opt-in multi-agent discovery coordination. Default False.
    num_agents : int
        Number of discovery agents when multi-agent mode is enabled.
    landscape_regions : list[list[float]] or None
        Optional spectral regions assigned to agents by index order.
    enable_basin_topology_mapping : bool
        Enable opt-in basin topology identification from trajectory history.
    basin_distance_threshold : float
        Distance threshold used for deterministic basin assignment.

    enable_landscape_learning : bool
        Enable opt-in persistent spectral landscape memory updates. Default False.
    landscape_memory : SpectralLandscapeMemory or None
        Optional externally managed spectral landscape memory.
    landscape_cluster_threshold : float
        Distance threshold for deterministic region clustering.
    use_landscape_novelty_bias : bool
        Backward-compatible alias that enables novelty bias when True.
    landscape_novelty_weight : float
        Legacy novelty weight alias. Default 0.0.
    novelty_weight : float or None
        Primary novelty weight used in ranking when landscape learning is enabled.

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

    agents: list[DiscoveryAgent] = []
    agent_artifacts: list[dict[str, Any]] = []
    if enable_multi_agent_discovery:
        coordinator = MultiAgentCoordinator()
        for agent_idx in range(max(0, int(num_agents))):
            coordinator.register_agent(DiscoveryAgent(agent_id=agent_idx))
        coordinator.assign_agents_to_regions(landscape_regions)
        agents = coordinator.list_agents()
    scheduled_target_spectrum = None
    scheduler_strategy = None
    detected_landscape_gap_count = 0
    scheduler_queue_obj = scheduler_queue
    if enable_autonomous_scheduler and scheduler_queue_obj is None:
        scheduler_queue_obj = ExperimentQueue(max_length=scheduler_max_gaps)
    if enable_landscape_learning and landscape_memory is None:
        landscape_memory = SpectralLandscapeMemory()

    effective_novelty_weight = (
        float(np.float64(landscape_novelty_weight))
        if novelty_weight is None
        else float(np.float64(novelty_weight))
    )
    if enable_basin_topology_mapping and not enable_spectral_trajectory and trajectory_recorder is None:
        trajectory_recorder = SpectralTrajectoryRecorder()

    # ── Step 1: Initialize population ──────────────────────────────
    init_seed = _derive_seed(base_seed, "init")
    raw_candidates = generate_tanner_graph_candidates(
        spec, population_size, base_seed=init_seed,
    )

    archive = create_archive(top_k=archive_top_k)
    elite_history: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    signature_archive: list[tuple[float, ...]] = []
    basin_assignments: list[int] = []
    strategy_history: list[str] = []
    escape_attempts = 0
    escape_successes = 0
    basin_centers: dict[int, np.ndarray] = {}
    basin_counts: dict[int, int] = {}
    basin_assignments: list[int] = []
    basin_escape_events: list[dict[str, Any]] = []

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
        basin_assignments.append(_candidate_basin_id(best))

    # ── Main loop ──────────────────────────────────────────────────
    for gen in range(1, num_generations + 1):
        gen_seed = _derive_seed(base_seed, f"gen_{gen}")

        # Select elites for mutation (top half)
        n_elites = max(1, len(population) // 2)
        elites = population[:n_elites]

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

        gradient_target_spectrum = None
        if enable_spectral_gradient and trajectory_recorder is not None:
            current_spectrum = _objective_spectrum(elites[0].get("objectives", {}))
            gradient = estimate_spectral_gradient(trajectory_recorder.as_array())
            if gradient.shape == current_spectrum.shape:
                gradient_target_spectrum = propose_gradient_step(
                    current_spectrum,
                    gradient,
                    step=gradient_step_size,
                )

        exploration_state = "LOCAL_OPTIMIZATION"
        exploration_strategy = "GRADIENT"
        switch_rate_value = 0.0
        entropy_value = 0.0
        mean_duration_value = 0.0
        escape_success_rate_value = 0.0
        if enable_adaptive_exploration:
            trajectory_data = (
                trajectory_recorder.as_array()
                if trajectory_recorder is not None
                else np.zeros((0, 0), dtype=np.float64)
            )
            exploration_state = analyze_exploration_state(
                trajectory_data,
                basin_assignments,
                window=exploration_window,
            )
            switch_rate_value = basin_switch_rate(basin_assignments, window=exploration_window)
            entropy_value = exploration_entropy(basin_assignments)
            mean_duration_value = mean_basin_duration(basin_assignments)
            exploration_strategy = choose_exploration_strategy(exploration_state)

            escape_success_rate_value = _safe_escape_success_rate(
                escape_successes,
                escape_attempts,
            )
            if switch_rate_value > float(early_exploration_rate_threshold):
                if exploration_strategy == "ESCAPE":
                    exploration_strategy = "GRADIENT"
            if escape_attempts >= int(max(1, exploration_window)):
                exploration_strategy = apply_escape_feedback_bias(
                    exploration_strategy,
                    escape_success_rate=escape_success_rate_value,
                    low_success_threshold=low_escape_success_threshold,
                )

        escape_target = None
        if gradient_target_spectrum is not None:
            adaptive_escape_step = float(escape_base_step)
            if enable_adaptive_exploration:
                scale = 1.0 + float(mean_duration_value) / float(max(1, exploration_window))
                scale = min(scale, 3.0)
                adaptive_escape_step = float(escape_base_step) * float(scale)
            escape_target = np.asarray(
                gradient_target_spectrum,
                dtype=np.float64,
            ) + np.array(
                [
                    adaptive_escape_step,
                    -adaptive_escape_step,
                    adaptive_escape_step,
                    adaptive_escape_step,
                ],
                dtype=np.float64,
            )

        strategy_history.append(str(exploration_strategy))
        if len(strategy_history) > int(max(1, exploration_window)):
            strategy_history.pop(0)
        if strategy_history.count("ESCAPE") > int(max(1, exploration_window)) // 2:
            exploration_strategy = "ESCAPE"
        current_spectrum_best = _objective_spectrum(elites[0].get("objectives", {}))
        current_basin = _assign_to_basin(current_spectrum_best, basin_centers)
        basin_assignments.append(current_basin)
        _update_basin_center(current_basin, current_spectrum_best, basin_centers, basin_counts)

        escape_triggered = False
        escape_direction = np.zeros_like(current_spectrum_best, dtype=np.float64)
        escape_target = None
        escape_event: dict[str, Any] | None = None
        if enable_basin_escape:
            escape_triggered = detect_basin_stagnation(
                basin_assignments,
                window=basin_escape_window,
            )
            if escape_triggered:
                basin_center = basin_centers.get(current_basin, current_spectrum_best)
                other_centers = [
                    center
                    for basin_id, center in basin_centers.items()
                    if int(basin_id) != int(current_basin)
                ]
                escape_direction = estimate_escape_direction(
                    current_spectrum_best,
                    basin_center,
                    other_centers=other_centers,
                )
                escape_target = propose_escape_step(
                    current_spectrum_best,
                    escape_direction,
                    step=basin_escape_step,
                )
                escape_event = {
                    "step": int(gen),
                    "basin_id": int(current_basin),
                    "escape_direction": escape_direction.tolist(),
                    "escape_target": escape_target.tolist(),
                    "escape_norm": float(np.linalg.norm(escape_direction)),
                    "escape_success": False,
                }
                basin_escape_events.append(escape_event)

        # Mutate elites
        children: list[dict[str, Any]] = []
        if use_nb_eigenmode_mutation:
            operator_name = "nb_eigenmode_mutation"
        elif escape_triggered:
            operator_name = _ESCAPE_OPERATORS[gen % len(_ESCAPE_OPERATORS)]
        else:
            operator_name = get_operator_for_generation(gen)

        scheduled_generation_target = None
        if enable_autonomous_scheduler:
            landscape_memory = _PopulationLandscapeMemory(population)
            scheduled_experiment = schedule_next_experiment(
                landscape_memory,
                gap_radius=scheduler_gap_radius,
                max_gaps=scheduler_max_gaps,
            )
            scheduled_generation_target = scheduled_experiment.get("target_spectrum")
            if scheduled_generation_target is not None and scheduler_queue_obj is not None:
                scheduler_queue_obj.push(scheduled_generation_target)
                scheduled_generation_target = scheduler_queue_obj.pop()
            scheduled_target_spectrum = scheduled_generation_target
            scheduler_strategy = scheduled_experiment.get("strategy")
            detected_landscape_gap_count = int(scheduled_experiment.get("gap_count", 0))

        for ei, elite in enumerate(elites):
            mut_seed = _derive_seed(gen_seed, f"mutate_{ei}")

            routing = _route_exploration_targets(
                exploration_strategy,
                gradient_target=gradient_target_spectrum,
                nb_target_edges=nb_guide_edges,
                escape_target=escape_target,
                default_target_edges=guide_edges,
            )
            target_edges = routing["target_edges"]
            target_spectrum = routing["target_spectrum"]

            if escape_target is not None:
                gradient_target_spectrum = escape_target

            if scheduled_generation_target is not None:
                gradient_target_spectrum = np.asarray(
                    scheduled_generation_target, dtype=np.float64,
                )

            H_mutated, op_used = mutate_tanner_graph(
                elite["H"],
                operator=operator_name,
                generation=gen,
                seed=mut_seed,
                target_edges=target_edges,
                target_spectrum=target_spectrum,
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

            if (enable_spectral_trajectory or enable_basin_topology_mapping) and trajectory_recorder is not None:
                current_spectrum = _objective_spectrum(repaired_objectives)
                nb_eigenvalue = _compute_nb_mode_magnitude(repaired_objectives, H_repaired)
                trajectory_recorder.record(np.append(current_spectrum, nb_eigenvalue))

            if enable_multi_agent_discovery and agents:
                agent = agents[ei % len(agents)]
                current_spectrum = _objective_spectrum(repaired_objectives)
                agent.record(current_spectrum, region_id=agent.assigned_region)

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
            if enable_landscape_learning and landscape_memory is not None:
                repaired_spectrum = _objective_spectrum(repaired_objectives)
                if use_landscape_novelty_bias or effective_novelty_weight != 0.0:
                    novelty_bias = landscape_novelty_score(repaired_spectrum, landscape_memory)
                    novelty += effective_novelty_weight * novelty_bias
                landscape_memory.add(
                    repaired_spectrum, threshold=landscape_cluster_threshold,
                )

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
            if enable_adaptive_exploration:
                child_state["exploration_state"] = exploration_state
                child_state["exploration_strategy"] = exploration_strategy
                if exploration_strategy == "ESCAPE":
                    escape_attempts += 1
                    parent_score = float(parent_composite)
                    child_score = float(repaired_objectives.get("composite_score", parent_score))
                    if child_score < parent_score:
                        escape_successes += 1
            children.append(child_state)

        if escape_event is not None:
            parent_score = float(elites[0]["objectives"].get("composite_score", float("inf")))
            best_child_score = min(
                (float(c["objectives"].get("composite_score", float("inf"))) for c in children),
                default=float("inf"),
            )
            escape_event["escape_success"] = bool(best_child_score < parent_score)

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

        if best:
            new_basin_id = _candidate_basin_id(best)
            basin_assignments.append(new_basin_id)

        generation_summaries.append(
            _make_generation_summary(
                gen,
                population,
                archive,
                include_basin_switches=enable_basin_switch_detection,
                exploration_state=(exploration_state if enable_adaptive_exploration else None),
                exploration_strategy=(exploration_strategy if enable_adaptive_exploration else None),
                basin_switch_rate_value=(switch_rate_value if enable_adaptive_exploration else None),
                exploration_entropy_value=(entropy_value if enable_adaptive_exploration else None),
                escape_success_rate_value=(
                    _safe_escape_success_rate(escape_successes, escape_attempts)
                    if enable_adaptive_exploration
                    else None
                ),
            )
        )

    # ── Assemble result ────────────────────────────────────────────
    best_candidate = population[0] if population else None
    archive_summary = get_archive_summary(archive)

    if enable_multi_agent_discovery and agents:
        for agent in agents:
            for spectrum in agent.trajectory:
                archive_summary.setdefault("shared_landscape_memory", []).append(
                    np.asarray(spectrum, dtype=np.float64).tolist()
                )
            agent_artifacts.append(agent.to_artifact())

    result = {
        "best_candidate": _serialize_candidate(best_candidate) if best_candidate else None,
        "best_H": best_candidate["H"] if best_candidate else None,
        "elite_history": elite_history,
        "archive_summary": archive_summary,
        "generation_summaries": generation_summaries,
    }
    if enable_spectral_trajectory and trajectory_recorder is not None:
        result["spectral_trajectory"] = trajectory_recorder.as_array().tolist()
    if enable_multi_agent_discovery:
        result["agent_artifacts"] = agent_artifacts
        result["num_agents"] = len(agents)
    if enable_autonomous_scheduler:
        result["scheduled_target_spectrum"] = (
            None
            if scheduled_target_spectrum is None
            else np.asarray(scheduled_target_spectrum, dtype=np.float64).tolist()
        )
        result["landscape_gap_count"] = int(detected_landscape_gap_count)
        result["scheduler_strategy"] = (
            scheduler_strategy if scheduler_strategy is not None else "landscape_exploration"
        )
    if enable_landscape_learning and landscape_memory is not None:
        result["spectral_landscape_regions"] = landscape_memory.centers().tolist()
        result["landscape_coverage"] = landscape_coverage(landscape_memory)
    if enable_basin_escape:
        result["basin_escape_events"] = basin_escape_events

    if enable_basin_topology_mapping and trajectory_recorder is not None:
        trajectory = trajectory_recorder.as_array()
        if trajectory.shape[0] > 0:
            assignments, centers = identify_spectral_basins(
                trajectory,
                threshold=basin_distance_threshold,
            )
            transitions = detect_basin_transitions(assignments)
            result["spectral_basin_topology"] = export_basin_map(
                centers,
                assignments,
                transitions,
                include_phase_space_projections=(trajectory.shape[1] >= 3),
            )
            result["spectral_basin_sizes"] = basin_sizes(assignments)
        else:
            empty_assignments = np.zeros((0,), dtype=np.int64)
            empty_centers = np.zeros((0, 0), dtype=np.float64)
            result["spectral_basin_topology"] = export_basin_map(
                empty_centers,
                empty_assignments,
                [],
                include_phase_space_projections=False,
            )
            result["spectral_basin_sizes"] = {}
    return result


def _assign_to_basin(
    spectrum: np.ndarray,
    basin_centers: dict[int, np.ndarray],
) -> int:
    """Assign spectrum to nearest center (or create basin 0 when empty)."""
    if not basin_centers:
        return 0
    best_id = min(
        basin_centers.keys(),
        key=lambda b: (float(np.linalg.norm(spectrum - basin_centers[b])), int(b)),
    )
    distance = float(np.linalg.norm(spectrum - basin_centers[best_id]))
    if distance > 0.25:
        return int(max(basin_centers.keys()) + 1)
    return int(best_id)


def _update_basin_center(
    basin_id: int,
    spectrum: np.ndarray,
    basin_centers: dict[int, np.ndarray],
    basin_counts: dict[int, int],
) -> None:
    """Deterministically update running basin center mean."""
    if basin_id not in basin_centers:
        basin_centers[basin_id] = np.asarray(spectrum, dtype=np.float64)
        basin_counts[basin_id] = 1
        return

    count = int(basin_counts.get(basin_id, 1))
    center = np.asarray(basin_centers[basin_id], dtype=np.float64)
    updated = (center * count + np.asarray(spectrum, dtype=np.float64)) / float(count + 1)
    basin_centers[basin_id] = updated.astype(np.float64, copy=False)
    basin_counts[basin_id] = count + 1


def _make_generation_summary(
    generation: int,
    population: list[dict[str, Any]],
    archive: dict[str, Any],
    *,
    include_basin_switches: bool = False,
    exploration_state: str | None = None,
    exploration_strategy: str | None = None,
    basin_switch_rate_value: float | None = None,
    exploration_entropy_value: float | None = None,
    escape_success_rate_value: float | None = None,
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
    if exploration_state is not None:
        result["exploration_state"] = str(exploration_state)
    if exploration_strategy is not None:
        result["exploration_strategy"] = str(exploration_strategy)
    if basin_switch_rate_value is not None:
        result["basin_switch_rate"] = float(basin_switch_rate_value)
    if exploration_entropy_value is not None:
        result["exploration_entropy"] = float(exploration_entropy_value)
    if escape_success_rate_value is not None:
        result["escape_success_rate"] = float(escape_success_rate_value)
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


class _PopulationLandscapeMemory:
    """Minimal deterministic landscape memory view over current population."""

    def __init__(self, population: list[dict[str, Any]]) -> None:
        self._population = population

    def centers(self) -> np.ndarray:
        if not self._population:
            return np.zeros((0, 0), dtype=np.float64)
        centers = [
            _objective_spectrum(state.get("objectives", {}))
            for state in self._population
        ]
        return np.asarray(centers, dtype=np.float64)

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



def _candidate_basin_id(candidate: dict[str, Any]) -> int:
    """Compute deterministic basin assignment from rounded objective signature."""
    obj = candidate.get("objectives", {})
    signature = (
        round(float(obj.get("spectral_radius", 0.0)), 6),
        round(float(obj.get("bethe_margin", 0.0)), 6),
    )
    data = repr(signature).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int(int.from_bytes(digest[:8], "big") % (2**63 - 1))


def _route_exploration_targets(
    strategy: str,
    *,
    gradient_target: np.ndarray | None,
    nb_target_edges: list[tuple[int, int]],
    escape_target: np.ndarray | None,
    default_target_edges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Route strategy to deterministic mutation guidance targets."""
    if strategy == "GRADIENT":
        return {"target_edges": default_target_edges, "target_spectrum": gradient_target}
    if strategy == "NB_EIGENMODE":
        return {"target_edges": nb_target_edges if nb_target_edges else default_target_edges, "target_spectrum": None}
    if strategy == "ESCAPE":
        return {"target_edges": default_target_edges, "target_spectrum": escape_target}
    return {"target_edges": None, "target_spectrum": None}


def _safe_escape_success_rate(escape_successes: int, escape_attempts: int) -> float:
    """Compute escape success-rate with deterministic safe division and clamp."""
    rate = (
        float(escape_successes) / float(escape_attempts)
        if int(escape_attempts) > 0
        else 0.0
    )
    return float(min(max(float(rate), 0.0), 1.0))

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
