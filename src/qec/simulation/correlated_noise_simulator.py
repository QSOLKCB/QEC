"""v138.1.0 — Correlated Noise Simulator.

Deterministic realistic simulation kernel for temporal, spatial, and
spatiotemporal correlated fault generation. This module is additive-only and
decoder-untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

CORRELATED_NOISE_SIMULATOR_VERSION = "v138.1.0"

SUPPORTED_MODELS: Tuple[str, ...] = (
    "independent_baseline",
    "temporal_markov",
    "nearest_neighbor_spatial",
    "spatiotemporal_cluster",
)
SUPPORTED_TOPOLOGIES: Tuple[str, ...] = ("line", "ring", "grid")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize_value(value[k]) for k in sorted(value.keys(), key=lambda x: str(x))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in canonicalized value")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def _dict_from_any(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, CorrelatedNoiseConfig):
        return raw.to_dict()
    if not isinstance(raw, Mapping):
        raise TypeError("config must be mapping or CorrelatedNoiseConfig")
    return {str(k): raw[k] for k in raw.keys()}


@dataclass(frozen=True)
class CorrelatedNoiseConfig:
    model: str
    topology: str
    num_sites: int
    time_steps: int
    event_rate: float
    seed: int
    temporal_alpha: float = 0.65
    spatial_beta: float = 0.45
    cluster_rate: float = 0.20
    cluster_max_size: int = 4
    version: str = CORRELATED_NOISE_SIMULATOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model,
            "topology": self.topology,
            "num_sites": self.num_sites,
            "time_steps": self.time_steps,
            "event_rate": float(self.event_rate),
            "seed": int(self.seed),
            "temporal_alpha": float(self.temporal_alpha),
            "spatial_beta": float(self.spatial_beta),
            "cluster_rate": float(self.cluster_rate),
            "cluster_max_size": int(self.cluster_max_size),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseEvent:
    event_id: str
    model: str
    topology: str
    time_step: int
    site_index: int
    probability: float
    trigger: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "model": self.model,
            "topology": self.topology,
            "time_step": int(self.time_step),
            "site_index": int(self.site_index),
            "probability": float(self.probability),
            "trigger": self.trigger,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseCluster:
    cluster_id: str
    model: str
    start_time_step: int
    end_time_step: int
    site_indices: Tuple[int, ...]
    event_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "model": self.model,
            "start_time_step": int(self.start_time_step),
            "end_time_step": int(self.end_time_step),
            "site_indices": [int(site) for site in self.site_indices],
            "event_ids": list(self.event_ids),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseRealization:
    version: str
    config_hash: str
    topology: str
    model: str
    adjacency: Tuple[Tuple[int, ...], ...]
    events: Tuple[CorrelatedNoiseEvent, ...]
    clusters: Tuple[CorrelatedNoiseCluster, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "topology": self.topology,
            "model": self.model,
            "adjacency": [[int(site) for site in neighbors] for neighbors in self.adjacency],
            "events": [event.to_dict() for event in self.events],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseReport:
    version: str
    realization_hash: str
    num_events: int
    num_clusters: int
    event_count_by_time_step: Tuple[Tuple[int, int], ...]
    event_count_by_site: Tuple[Tuple[int, int], ...]
    cluster_sizes: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "realization_hash": self.realization_hash,
            "num_events": int(self.num_events),
            "num_clusters": int(self.num_clusters),
            "event_count_by_time_step": [[int(k), int(v)] for k, v in self.event_count_by_time_step],
            "event_count_by_site": [[int(k), int(v)] for k, v in self.event_count_by_site],
            "cluster_sizes": [int(size) for size in self.cluster_sizes],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseReceipt:
    version: str
    config_hash: str
    realization_hash: str
    report_hash: str
    receipt_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "realization_hash": self.realization_hash,
            "report_hash": self.report_hash,
            "receipt_hash": self.receipt_hash,
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseSimulator:
    config: CorrelatedNoiseConfig
    decoder_untouched: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "decoder_untouched": self.decoder_untouched,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def validate_noise_config(config: Any) -> CorrelatedNoiseConfig:
    data = _dict_from_any(config)
    normalized = CorrelatedNoiseConfig(
        version=str(data.get("version", CORRELATED_NOISE_SIMULATOR_VERSION)).strip() or CORRELATED_NOISE_SIMULATOR_VERSION,
        model=str(data.get("model", "")).strip(),
        topology=str(data.get("topology", "")).strip(),
        num_sites=int(data.get("num_sites", 0)),
        time_steps=int(data.get("time_steps", 0)),
        event_rate=float(data.get("event_rate", -1.0)),
        seed=int(data["seed"]),
        temporal_alpha=float(data.get("temporal_alpha", 0.65)),
        spatial_beta=float(data.get("spatial_beta", 0.45)),
        cluster_rate=float(data.get("cluster_rate", 0.20)),
        cluster_max_size=int(data.get("cluster_max_size", 4)),
    )

    if normalized.model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {normalized.model}")
    if normalized.topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"unsupported topology: {normalized.topology}")
    if normalized.num_sites <= 0:
        raise ValueError("num_sites must be > 0")
    if normalized.time_steps <= 0:
        raise ValueError("time_steps must be > 0")
    if not (0.0 <= normalized.event_rate <= 1.0):
        raise ValueError("event_rate must be within [0.0, 1.0]")
    if not (0.0 <= normalized.temporal_alpha <= 1.0):
        raise ValueError("temporal_alpha must be within [0.0, 1.0]")
    if not (0.0 <= normalized.spatial_beta <= 1.0):
        raise ValueError("spatial_beta must be within [0.0, 1.0]")
    if not (0.0 <= normalized.cluster_rate <= 1.0):
        raise ValueError("cluster_rate must be within [0.0, 1.0]")
    if normalized.cluster_max_size <= 0:
        raise ValueError("cluster_max_size must be > 0")

    return normalized


def build_topology_adjacency(config: Any) -> Tuple[Tuple[int, ...], ...]:
    cfg = validate_noise_config(config)
    neighbors: list[Tuple[int, ...]] = []

    if cfg.topology == "line":
        for site in range(cfg.num_sites):
            linked = []
            if site > 0:
                linked.append(site - 1)
            if site + 1 < cfg.num_sites:
                linked.append(site + 1)
            neighbors.append(tuple(linked))
    elif cfg.topology == "ring":
        if cfg.num_sites == 1:
            neighbors = [tuple()]
        else:
            for site in range(cfg.num_sites):
                left = (site - 1) % cfg.num_sites
                right = (site + 1) % cfg.num_sites
                linked = (left,) if left == right else tuple(sorted((left, right)))
                neighbors.append(linked)
    elif cfg.topology == "grid":
        width = int(math.ceil(math.sqrt(cfg.num_sites)))
        for site in range(cfg.num_sites):
            row, col = divmod(site, width)
            linked: list[int] = []
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = row + d_row
                nc = col + d_col
                if nr < 0 or nc < 0:
                    continue
                neighbor = nr * width + nc
                if 0 <= neighbor < cfg.num_sites:
                    linked.append(neighbor)
            neighbors.append(tuple(sorted(set(linked))))

    return tuple(neighbors)


def _register_event(
    *,
    events: Dict[Tuple[int, int], CorrelatedNoiseEvent],
    config: CorrelatedNoiseConfig,
    time_step: int,
    site_index: int,
    probability: float,
    trigger: str,
) -> None:
    key = (time_step, site_index)
    if key in events:
        return
    event_id = f"event.t{time_step:04d}.s{site_index:04d}"
    events[key] = CorrelatedNoiseEvent(
        event_id=event_id,
        model=config.model,
        topology=config.topology,
        time_step=time_step,
        site_index=site_index,
        probability=_clamp01(probability),
        trigger=trigger,
    )


def generate_correlated_noise(config: Any) -> CorrelatedNoiseRealization:
    cfg = validate_noise_config(config)
    adjacency = build_topology_adjacency(cfg)
    rng = random.Random(cfg.seed)

    events: Dict[Tuple[int, int], CorrelatedNoiseEvent] = {}
    clusters_raw: list[Dict[str, Any]] = []

    previous_active = [False] * cfg.num_sites

    for time_step in range(cfg.time_steps):
        active_now = [False] * cfg.num_sites
        for site_index in range(cfg.num_sites):
            neighbor_indices = adjacency[site_index]
            previous_neighbor_active = 0
            if neighbor_indices:
                previous_neighbor_active = sum(1 for neighbor in neighbor_indices if previous_active[neighbor])
            neighbor_ratio = (
                float(previous_neighbor_active) / float(len(neighbor_indices))
                if neighbor_indices
                else 0.0
            )

            probability = cfg.event_rate
            trigger = "baseline"
            if cfg.model == "temporal_markov":
                probability = _clamp01(cfg.event_rate + cfg.temporal_alpha * (1.0 if previous_active[site_index] else -cfg.event_rate))
                trigger = "temporal"
            elif cfg.model == "nearest_neighbor_spatial":
                probability = _clamp01(cfg.event_rate + (cfg.spatial_beta * cfg.event_rate * neighbor_ratio))
                trigger = "spatial"
            elif cfg.model == "spatiotemporal_cluster":
                probability = _clamp01(
                    cfg.event_rate
                    + (0.5 * cfg.temporal_alpha * (1.0 if previous_active[site_index] else -cfg.event_rate))
                    + (0.5 * cfg.spatial_beta * cfg.event_rate * neighbor_ratio)
                )
                trigger = "spatiotemporal"

            if rng.random() < probability:
                _register_event(
                    events=events,
                    config=cfg,
                    time_step=time_step,
                    site_index=site_index,
                    probability=probability,
                    trigger=trigger,
                )
                active_now[site_index] = True

                if cfg.model == "spatiotemporal_cluster":
                    cluster_probability = _clamp01(cfg.cluster_rate * probability)
                    if rng.random() < cluster_probability:
                        target_size = 1 + rng.randrange(cfg.cluster_max_size)
                        cluster_members: list[Tuple[int, int, str]] = [(time_step, site_index, "cluster_seed")]
                        visited = {(time_step, site_index)}

                        frontier: list[Tuple[int, int, str]] = []
                        for neighbor in sorted(adjacency[site_index]):
                            frontier.append((time_step, neighbor, "cluster_spatial"))
                        if time_step + 1 < cfg.time_steps:
                            frontier.append((time_step + 1, site_index, "cluster_temporal"))

                        while frontier and len(cluster_members) < target_size:
                            next_time, next_site, mode = frontier.pop(0)
                            if (next_time, next_site) in visited:
                                continue
                            visited.add((next_time, next_site))
                            if rng.random() < _clamp01(probability + 0.25 * cfg.cluster_rate):
                                cluster_members.append((next_time, next_site, mode))
                                if next_time == time_step:
                                    active_now[next_site] = True
                                for neighbor in sorted(adjacency[next_site]):
                                    candidate = (next_time, neighbor, "cluster_spatial")
                                    if (candidate[0], candidate[1]) not in visited:
                                        frontier.append(candidate)
                                if next_time + 1 < cfg.time_steps:
                                    frontier.append((next_time + 1, next_site, "cluster_temporal"))

                        for member_time, member_site, mode in sorted(cluster_members, key=lambda x: (x[0], x[1], x[2])):
                            _register_event(
                                events=events,
                                config=cfg,
                                time_step=member_time,
                                site_index=member_site,
                                probability=probability,
                                trigger=mode,
                            )

                        clusters_raw.append(
                            {
                                "start_time_step": min(member[0] for member in cluster_members),
                                "end_time_step": max(member[0] for member in cluster_members),
                                "site_indices": tuple(sorted({member[1] for member in cluster_members})),
                                "event_keys": tuple(sorted((member[0], member[1]) for member in cluster_members)),
                            }
                        )

        previous_active = active_now

    ordered_events = tuple(events[key] for key in sorted(events.keys(), key=lambda key: (key[0], key[1])))

    clusters: list[CorrelatedNoiseCluster] = []
    for cluster_index, raw_cluster in enumerate(
        sorted(clusters_raw, key=lambda item: (item["start_time_step"], item["end_time_step"], item["site_indices"]))
    ):
        cluster_event_ids = tuple(
            events[event_key].event_id
            for event_key in raw_cluster["event_keys"]
            if event_key in events
        )
        clusters.append(
            CorrelatedNoiseCluster(
                cluster_id=f"cluster.{cluster_index:04d}",
                model=cfg.model,
                start_time_step=int(raw_cluster["start_time_step"]),
                end_time_step=int(raw_cluster["end_time_step"]),
                site_indices=tuple(int(site) for site in raw_cluster["site_indices"]),
                event_ids=cluster_event_ids,
            )
        )

    return CorrelatedNoiseRealization(
        version=cfg.version,
        config_hash=cfg.stable_hash(),
        topology=cfg.topology,
        model=cfg.model,
        adjacency=adjacency,
        events=ordered_events,
        clusters=tuple(clusters),
    )


def summarize_noise_realization(realization: CorrelatedNoiseRealization) -> CorrelatedNoiseReport:
    by_time: Dict[int, int] = {}
    by_site: Dict[int, int] = {}

    for event in realization.events:
        by_time[event.time_step] = by_time.get(event.time_step, 0) + 1
        by_site[event.site_index] = by_site.get(event.site_index, 0) + 1

    cluster_sizes = tuple(sorted((len(cluster.event_ids) for cluster in realization.clusters), reverse=True))

    return CorrelatedNoiseReport(
        version=realization.version,
        realization_hash=realization.stable_hash(),
        num_events=len(realization.events),
        num_clusters=len(realization.clusters),
        event_count_by_time_step=tuple((step, by_time[step]) for step in sorted(by_time.keys())),
        event_count_by_site=tuple((site, by_site[site]) for site in sorted(by_site.keys())),
        cluster_sizes=cluster_sizes,
    )


def build_noise_receipt(
    config: Any,
    realization: CorrelatedNoiseRealization,
    report: CorrelatedNoiseReport,
) -> CorrelatedNoiseReceipt:
    cfg = validate_noise_config(config)
    payload = {
        "version": cfg.version,
        "config_hash": cfg.stable_hash(),
        "realization_hash": realization.stable_hash(),
        "report_hash": report.stable_hash(),
    }
    replay_identity = _stable_hash(payload)
    receipt_hash = _stable_hash({"payload": payload, "replay_identity": replay_identity})
    return CorrelatedNoiseReceipt(
        version=cfg.version,
        config_hash=payload["config_hash"],
        realization_hash=payload["realization_hash"],
        report_hash=payload["report_hash"],
        replay_identity=replay_identity,
        receipt_hash=receipt_hash,
    )


__all__ = [
    "CORRELATED_NOISE_SIMULATOR_VERSION",
    "SUPPORTED_MODELS",
    "SUPPORTED_TOPOLOGIES",
    "CorrelatedNoiseConfig",
    "CorrelatedNoiseEvent",
    "CorrelatedNoiseCluster",
    "CorrelatedNoiseRealization",
    "CorrelatedNoiseReport",
    "CorrelatedNoiseReceipt",
    "CorrelatedNoiseSimulator",
    "validate_noise_config",
    "build_topology_adjacency",
    "generate_correlated_noise",
    "summarize_noise_realization",
    "build_noise_receipt",
]
