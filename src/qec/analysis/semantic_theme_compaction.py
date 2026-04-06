"""v137.7.1 — Semantic → Theme Compaction.

Deterministic Layer-4 semantic compaction from episodes into bounded symbolic themes.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import hashlib
import json
import math
import re
from typing import Any, Mapping

from qec.analysis.episodic_memory_lifting import EpisodeRecord, EpisodicMemoryArtifact

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

SEMANTIC_THEME_COMPACTION_LAW = "SEMANTIC_THEME_COMPACTION_LAW"
DETERMINISTIC_THEME_ASSIGNMENT_RULE = "DETERMINISTIC_THEME_ASSIGNMENT_RULE"
REPLAY_SAFE_THEME_CHAIN_INVARIANT = "REPLAY_SAFE_THEME_CHAIN_INVARIANT"
BOUNDED_THEME_AGGREGATION_RULE = "BOUNDED_THEME_AGGREGATION_RULE"

class _ThemeRule(str, enum.Enum):
    EXPLICIT_LABEL = "explicit_label"
    PARENT_CHAIN = "parent_chain"
    TASK_COMPLETION = "task_completion"
    BOUNDARY_SIGNATURE = "boundary_signature"
    RECORD_PREFIX = "record_prefix"
    EPISODE_HASH = "episode_hash"


_THEME_RULE_PRECEDENCE: tuple[_ThemeRule, ...] = (
    _ThemeRule.EXPLICIT_LABEL,
    _ThemeRule.PARENT_CHAIN,
    _ThemeRule.TASK_COMPLETION,
    _ThemeRule.BOUNDARY_SIGNATURE,
    _ThemeRule.RECORD_PREFIX,
    _ThemeRule.EPISODE_HASH,
)



def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")



def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )



def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")



def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()



def _validate_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped



def _extract_record_prefix(record_id: str) -> str | None:
    match = re.match(r"([A-Za-z]+)", record_id)
    if match is None:
        return None
    return match.group(1).lower()



def _extract_explicit_label(episode: EpisodeRecord) -> str | None:
    labels: list[str] = []
    for record_id in episode.record_ids:
        rid = record_id.lower()
        if rid.startswith("theme:"):
            labels.append(rid.split(":", 1)[1])
        elif rid.startswith("semantic:"):
            labels.append(rid.split(":", 1)[1])
    if len(labels) == 0:
        return None
    if any(label.strip() == "" for label in labels):
        raise ValueError("explicit semantic label must be non-empty")
    return min(labels)



def _theme_candidates_for_episode(episode: EpisodeRecord) -> dict[_ThemeRule, str]:
    candidates: dict[_ThemeRule, str] = {}
    explicit_label = _extract_explicit_label(episode)
    if explicit_label is not None:
        candidates[_ThemeRule.EXPLICIT_LABEL] = f"explicit_label:{explicit_label}"

    candidates[_ThemeRule.PARENT_CHAIN] = f"parent_chain:{episode.parent_episode_hash}"

    if "task_completion" in episode.boundary_reasons:
        candidates[_ThemeRule.TASK_COMPLETION] = "task_completion"

    if len(episode.boundary_reasons) > 0:
        candidates[_ThemeRule.BOUNDARY_SIGNATURE] = f"boundary_signature:{'|'.join(sorted(episode.boundary_reasons))}"

    prefixes = tuple(sorted({p for rid in episode.record_ids if (p := _extract_record_prefix(rid)) is not None}))
    if len(prefixes) > 0:
        candidates[_ThemeRule.RECORD_PREFIX] = f"record_prefix:{'|'.join(prefixes)}"

    candidates[_ThemeRule.EPISODE_HASH] = f"episode_hash:{episode.episode_hash}"
    return candidates


@dataclass(frozen=True)
class ThemeCompactionConfig:
    normalize_episode_order: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.normalize_episode_order, bool):
            raise ValueError("normalize_episode_order must be bool")


@dataclass(frozen=True)
class ThemeRecord:
    theme_id: str
    theme_index: int
    episode_ids: tuple[str, ...]
    episode_hashes: tuple[str, ...]
    theme_signature: str
    compaction_reasons: tuple[str, ...]
    parent_theme_hash: str
    theme_hash: str
    replay_identity_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "theme_id": self.theme_id,
            "theme_index": self.theme_index,
            "episode_ids": self.episode_ids,
            "episode_hashes": self.episode_hashes,
            "theme_signature": self.theme_signature,
            "compaction_reasons": self.compaction_reasons,
            "parent_theme_hash": self.parent_theme_hash,
            "theme_hash": self.theme_hash,
            "replay_identity_hash": self.replay_identity_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        """Payload for stable_hash; excludes self-referential hash fields."""
        return {
            "theme_id": self.theme_id,
            "theme_index": self.theme_index,
            "episode_ids": self.episode_ids,
            "episode_hashes": self.episode_hashes,
            "theme_signature": self.theme_signature,
            "compaction_reasons": self.compaction_reasons,
            "parent_theme_hash": self.parent_theme_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SemanticThemeArtifact:
    schema_version: int
    source_artifact_hash: str
    episode_count: int
    theme_count: int
    theme_ids: tuple[str, ...]
    themes: tuple[ThemeRecord, ...]
    episode_to_theme: tuple[tuple[str, str], ...]
    compaction_ratio: float
    law_invariants: tuple[str, ...]
    assignment_precedence: tuple[str, ...]
    artifact_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "episode_count": self.episode_count,
            "theme_count": self.theme_count,
            "theme_ids": self.theme_ids,
            "themes": tuple(theme.to_dict() for theme in self.themes),
            "episode_to_theme": self.episode_to_theme,
            "compaction_ratio": self.compaction_ratio,
            "law_invariants": self.law_invariants,
            "assignment_precedence": self.assignment_precedence,
            "artifact_hash": self.artifact_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        """Payload for stable_hash; excludes self-referential artifact_hash field."""
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "episode_count": self.episode_count,
            "theme_count": self.theme_count,
            "theme_ids": self.theme_ids,
            "themes": tuple(theme.to_dict() for theme in self.themes),
            "episode_to_theme": self.episode_to_theme,
            "compaction_ratio": self.compaction_ratio,
            "law_invariants": self.law_invariants,
            "assignment_precedence": self.assignment_precedence,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SemanticThemeReceipt:
    schema_version: int
    artifact_hash: str
    source_artifact_hash: str
    replay_chain_head: str
    theme_hashes: tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "source_artifact_hash": self.source_artifact_hash,
            "replay_chain_head": self.replay_chain_head,
            "theme_hashes": self.theme_hashes,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        """Payload for stable_hash; excludes self-referential receipt_hash field."""
        return {
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "source_artifact_hash": self.source_artifact_hash,
            "replay_chain_head": self.replay_chain_head,
            "theme_hashes": self.theme_hashes,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())



def _validate_episodic_artifact(artifact: EpisodicMemoryArtifact) -> tuple[EpisodeRecord, ...]:
    if not isinstance(artifact, EpisodicMemoryArtifact):
        raise ValueError("artifact must be an EpisodicMemoryArtifact")
    _validate_non_empty_str(artifact.source_sequence_hash, field_name="source_sequence_hash")
    if artifact.episode_count != len(artifact.episodes):
        raise ValueError("episode_count must match episodes length")
    if artifact.episode_count != len(artifact.episode_ids):
        raise ValueError("episode_count must match episode_ids length")

    seen_ids: set[str] = set()
    for ep in artifact.episodes:
        _validate_non_empty_str(ep.episode_id, field_name="episode_id")
        _validate_non_empty_str(ep.episode_hash, field_name="episode_hash")
        _validate_non_empty_str(ep.parent_episode_hash, field_name="parent_episode_hash")
        if not isinstance(ep.episode_index, int) or ep.episode_index < 0:
            raise ValueError("episode_index must be a non-negative int")
        if not isinstance(ep.boundary_reasons, tuple):
            raise ValueError("boundary_reasons must be a tuple")
        for reason in ep.boundary_reasons:
            if not isinstance(reason, str) or reason.strip() == "":
                raise ValueError("each boundary_reason must be a non-empty string")
        if ep.episode_id in seen_ids:
            raise ValueError("episode_id values must be unique")
        seen_ids.add(ep.episode_id)
        if len(ep.record_ids) == 0:
            raise ValueError("episode record_ids must be non-empty")
        for rid in ep.record_ids:
            _validate_non_empty_str(rid, field_name="record_id")

    if tuple(ep.episode_id for ep in artifact.episodes) != artifact.episode_ids:
        raise ValueError("episode_ids must align with episodes ordering")

    return artifact.episodes



def compact_episodic_memory_to_semantic_themes(
    artifact: EpisodicMemoryArtifact,
    *,
    config: ThemeCompactionConfig | None = None,
) -> SemanticThemeArtifact:
    config = config if config is not None else ThemeCompactionConfig()
    episodes = _validate_episodic_artifact(artifact)
    if config.normalize_episode_order:
        episodes = tuple(sorted(episodes, key=lambda item: (item.episode_index, item.episode_id)))

    episode_candidates: dict[str, dict[_ThemeRule, str]] = {}
    signature_counts: dict[_ThemeRule, dict[str, int]] = {rule: {} for rule in _THEME_RULE_PRECEDENCE}
    for ep in episodes:
        candidates = _theme_candidates_for_episode(ep)
        episode_candidates[ep.episode_id] = candidates
        for rule_name, signature in candidates.items():
            if rule_name not in _THEME_RULE_PRECEDENCE:
                raise ValueError("theme assignment rule must be in precedence table")
            _validate_non_empty_str(signature, field_name="theme_signature")
            signature_counts[rule_name][signature] = signature_counts[rule_name].get(signature, 0) + 1

    grouped: dict[tuple[_ThemeRule, str], list[EpisodeRecord]] = {}
    for ep in episodes:
        candidates = episode_candidates[ep.episode_id]
        selected_rule = _ThemeRule.EPISODE_HASH
        selected_signature = candidates[_ThemeRule.EPISODE_HASH]
        for rule_name in _THEME_RULE_PRECEDENCE:
            signature = candidates.get(rule_name)
            if signature is None:
                continue
            if rule_name is _ThemeRule.EPISODE_HASH or signature_counts[rule_name].get(signature, 0) >= 2:
                selected_rule = rule_name
                selected_signature = signature
                break
        grouped.setdefault((selected_rule, selected_signature), []).append(ep)

    # Second pass: downgrade any singleton non-fallback group to episode_hash.
    # A higher-precedence rule may have reduced the group size to 1 after
    # signature_counts were computed over all candidates.
    final_grouped: dict[tuple[_ThemeRule, str], list[EpisodeRecord]] = {}
    for key, eps_list in grouped.items():
        rule_name, _signature = key
        if rule_name is not _ThemeRule.EPISODE_HASH and len(eps_list) == 1:
            ep = eps_list[0]
            fallback_sig = episode_candidates[ep.episode_id][_ThemeRule.EPISODE_HASH]
            final_grouped.setdefault((_ThemeRule.EPISODE_HASH, fallback_sig), []).append(ep)
        else:
            final_grouped.setdefault(key, []).extend(eps_list)
    grouped = final_grouped

    group_order = sorted(
        grouped.keys(),
        key=lambda k: (
            min(ep.episode_index for ep in grouped[k]),
            _THEME_RULE_PRECEDENCE.index(k[0]),
            k[1],
        ),
    )

    themes: list[ThemeRecord] = []
    mapping: list[tuple[str, str]] = []
    parent_theme_hash = artifact.source_sequence_hash
    for idx, key in enumerate(group_order):
        rule_name, signature = key
        group_eps = tuple(sorted(grouped[key], key=lambda ep: (ep.episode_index, ep.episode_id)))
        theme_id = f"theme-{idx:06d}"
        theme_body = {
            "theme_id": theme_id,
            "theme_index": idx,
            "episode_ids": tuple(ep.episode_id for ep in group_eps),
            "episode_hashes": tuple(ep.episode_hash for ep in group_eps),
            "theme_signature": signature,
            "compaction_reasons": (rule_name,),
            "parent_theme_hash": parent_theme_hash,
        }
        theme_hash = _sha256_hex(theme_body)
        replay_identity_hash = _sha256_hex({"parent": parent_theme_hash, "theme_hash": theme_hash})
        theme = ThemeRecord(
            theme_id=theme_id,
            theme_index=idx,
            episode_ids=theme_body["episode_ids"],
            episode_hashes=theme_body["episode_hashes"],
            theme_signature=signature,
            compaction_reasons=theme_body["compaction_reasons"],
            parent_theme_hash=parent_theme_hash,
            theme_hash=theme_hash,
            replay_identity_hash=replay_identity_hash,
        )
        themes.append(theme)
        for episode_id in theme.episode_ids:
            mapping.append((episode_id, theme.theme_id))
        parent_theme_hash = replay_identity_hash

    mapping_sorted = tuple(sorted(mapping, key=lambda item: item[0]))
    theme_ids = tuple(theme.theme_id for theme in themes)
    if len(set(theme_ids)) != len(theme_ids):
        raise ValueError("duplicate theme ids are not allowed")

    if any(theme.theme_signature.strip() == "" for theme in themes):
        raise ValueError("empty theme signature is not allowed")

    episode_count = len(episodes)
    if episode_count == 0:
        raise ValueError("episodic artifacts must contain at least one episode")
    compaction_ratio = float((episode_count - len(themes)) / episode_count)
    if not (0.0 <= compaction_ratio <= 1.0):
        raise ValueError("compaction_ratio must be bounded in [0, 1]")

    law_invariants = (
        SEMANTIC_THEME_COMPACTION_LAW,
        DETERMINISTIC_THEME_ASSIGNMENT_RULE,
        REPLAY_SAFE_THEME_CHAIN_INVARIANT,
        BOUNDED_THEME_AGGREGATION_RULE,
    )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_artifact_hash": artifact.artifact_hash,
        "episode_count": episode_count,
        "theme_count": len(themes),
        "theme_ids": theme_ids,
        "themes": tuple(theme.to_dict() for theme in themes),
        "episode_to_theme": mapping_sorted,
        "compaction_ratio": compaction_ratio,
        "law_invariants": law_invariants,
        "assignment_precedence": _THEME_RULE_PRECEDENCE,
    }
    artifact_hash = _sha256_hex(payload)

    return SemanticThemeArtifact(
        schema_version=_SCHEMA_VERSION,
        source_artifact_hash=artifact.artifact_hash,
        episode_count=episode_count,
        theme_count=len(themes),
        theme_ids=theme_ids,
        themes=tuple(themes),
        episode_to_theme=mapping_sorted,
        compaction_ratio=compaction_ratio,
        law_invariants=law_invariants,
        assignment_precedence=_THEME_RULE_PRECEDENCE,
        artifact_hash=artifact_hash,
    )



def generate_semantic_theme_receipt(artifact: SemanticThemeArtifact) -> SemanticThemeReceipt:
    if not isinstance(artifact, SemanticThemeArtifact):
        raise ValueError("artifact must be a SemanticThemeArtifact")
    replay_chain_head = artifact.source_artifact_hash if artifact.theme_count == 0 else artifact.themes[-1].replay_identity_hash
    payload = {
        "schema_version": artifact.schema_version,
        "artifact_hash": artifact.artifact_hash,
        "source_artifact_hash": artifact.source_artifact_hash,
        "replay_chain_head": replay_chain_head,
        "theme_hashes": tuple(theme.theme_hash for theme in artifact.themes),
    }
    receipt_hash = _sha256_hex(payload)
    return SemanticThemeReceipt(
        schema_version=artifact.schema_version,
        artifact_hash=artifact.artifact_hash,
        source_artifact_hash=artifact.source_artifact_hash,
        replay_chain_head=replay_chain_head,
        theme_hashes=tuple(theme.theme_hash for theme in artifact.themes),
        receipt_hash=receipt_hash,
    )



def export_semantic_theme_bytes(artifact: SemanticThemeArtifact) -> bytes:
    if not isinstance(artifact, SemanticThemeArtifact):
        raise ValueError("artifact must be a SemanticThemeArtifact")
    return artifact.to_canonical_bytes()


__all__ = [
    "BOUNDED_THEME_AGGREGATION_RULE",
    "DETERMINISTIC_THEME_ASSIGNMENT_RULE",
    "REPLAY_SAFE_THEME_CHAIN_INVARIANT",
    "SEMANTIC_THEME_COMPACTION_LAW",
    "SemanticThemeArtifact",
    "SemanticThemeReceipt",
    "ThemeCompactionConfig",
    "ThemeRecord",
    "compact_episodic_memory_to_semantic_themes",
    "export_semantic_theme_bytes",
    "generate_semantic_theme_receipt",
]
