"""v137.3.0 — Deterministic NLP Governance Layer.

Canonical deterministic path:

    natural language
    -> intent lattice
    -> canonical semantic state schema
    -> governed response plan
    -> replay-safe output artifact

Layer: analysis (Layer 4), additive only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

DETERMINISTIC_NLP_GOVERNANCE_VERSION: str = "v137.3.0"

_INTENT_RULES: Mapping[str, Tuple[str, ...]] = {
    "escalate": ("urgent", "critical", "error", "incident", "fail", "failure", "outage"),
    "stabilize": ("stabilize", "steady", "recover", "restore", "safe", "safety", "guardrail"),
    "diagnose": ("inspect", "analyze", "diagnose", "trace", "debug", "report", "status"),
    "optimize": ("optimize", "improve", "efficiency", "throughput", "latency", "tune"),
}

_ACTION_BY_INTENT: Mapping[str, str] = {
    "escalate": "apply_fail_safe_latch",
    "stabilize": "apply_hysteresis_control",
    "diagnose": "emit_diagnostic_snapshot",
    "optimize": "apply_conservative_tuning",
}

_ALLOWED_MODES: Tuple[str, ...] = ("strict", "balanced")


@dataclass(frozen=True)
class IntentScore:
    """Deterministic score for a single intent in the lattice."""

    intent: str
    score: int
    matched_terms: Tuple[str, ...]


@dataclass(frozen=True)
class CanonicalSemanticState:
    """Canonical semantic state snapshot derived from normalized text."""

    normalized_text: str
    tokens: Tuple[str, ...]
    token_count: int
    unique_token_count: int


@dataclass(frozen=True)
class IntentActionEdge:
    """Directed deterministic graph edge from state/intent/action."""

    source: str
    target: str
    edge_type: str


@dataclass(frozen=True)
class GovernedResponsePlan:
    """Deterministic governed response planner output."""

    selected_intent: str
    selected_action: str
    rationale: str
    policy_mode: str


@dataclass(frozen=True)
class GovernedOutputArtifact:
    """Replay-safe governed output artifact and identity chain."""

    version: str
    state: CanonicalSemanticState
    lattice: Tuple[IntentScore, ...]
    graph: Tuple[IntentActionEdge, ...]
    plan: GovernedResponsePlan
    identity_chain: Tuple[str, ...]
    stable_hash: str


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_from_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must be non-empty")


def _validate_mode(policy_mode: str) -> None:
    if policy_mode not in _ALLOWED_MODES:
        raise ValueError(f"policy_mode must be one of {_ALLOWED_MODES}")


# ---------------------------------------------------------------------------
# Core deterministic transforms
# ---------------------------------------------------------------------------


def normalize_natural_language(text: str) -> CanonicalSemanticState:
    """Deterministically normalize free text into canonical semantic state."""
    _validate_text(text)
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    tokens = tuple(normalized_text.split()) if normalized_text else ()
    if not tokens:
        raise ValueError("text produced no canonical tokens")

    return CanonicalSemanticState(
        normalized_text=normalized_text,
        tokens=tokens,
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
    )


def compile_intent_lattice(state: CanonicalSemanticState) -> Tuple[IntentScore, ...]:
    """Compile deterministic intent lattice using lexical intent rules."""
    token_set = set(state.tokens)
    scored: list[IntentScore] = []
    for intent in sorted(_INTENT_RULES):
        terms = tuple(sorted(token_set.intersection(_INTENT_RULES[intent])))
        scored.append(IntentScore(intent=intent, score=len(terms), matched_terms=terms))

    scored.sort(key=lambda item: (-item.score, item.intent))
    return tuple(scored)


def _select_intent(lattice: Tuple[IntentScore, ...], policy_mode: str) -> IntentScore:
    if not lattice:
        raise ValueError("lattice must contain at least one intent")

    top_score = lattice[0].score
    top_candidates = [item for item in lattice if item.score == top_score]
    if policy_mode == "strict" and top_score == 0:
        return next(item for item in lattice if item.intent == "diagnose")

    return sorted(top_candidates, key=lambda item: item.intent)[0]


def build_state_intent_action_graph(
    state: CanonicalSemanticState,
    lattice: Tuple[IntentScore, ...],
    plan: GovernedResponsePlan,
) -> Tuple[IntentActionEdge, ...]:
    """Build deterministic state -> intent -> action graph."""
    edges: list[IntentActionEdge] = []
    state_node = f"state:{state.normalized_text}"

    for item in lattice:
        intent_node = f"intent:{item.intent}"
        edges.append(IntentActionEdge(source=state_node, target=intent_node, edge_type="state_to_intent"))
        edges.append(
            IntentActionEdge(
                source=intent_node,
                target=f"action:{_ACTION_BY_INTENT[item.intent]}",
                edge_type="intent_to_action",
            )
        )

    edges.append(
        IntentActionEdge(
            source=f"intent:{plan.selected_intent}",
            target=f"action:{plan.selected_action}",
            edge_type="selected_path",
        )
    )

    edges.sort(key=lambda edge: (edge.source, edge.target, edge.edge_type))
    return tuple(edges)


def plan_governed_response(
    state: CanonicalSemanticState,
    lattice: Tuple[IntentScore, ...],
    *,
    policy_mode: str = "strict",
) -> GovernedResponsePlan:
    """Deterministic governed response planner with stable tie-breaking."""
    _validate_mode(policy_mode)
    selected = _select_intent(lattice, policy_mode=policy_mode)
    action = _ACTION_BY_INTENT[selected.intent]
    rationale = (
        f"intent={selected.intent};score={selected.score};"
        f"token_count={state.token_count};policy_mode={policy_mode}"
    )
    return GovernedResponsePlan(
        selected_intent=selected.intent,
        selected_action=action,
        rationale=rationale,
        policy_mode=policy_mode,
    )


def _state_to_dict(state: CanonicalSemanticState) -> Dict[str, Any]:
    return {
        "normalized_text": state.normalized_text,
        "token_count": state.token_count,
        "tokens": list(state.tokens),
        "unique_token_count": state.unique_token_count,
    }


def _lattice_to_dict(lattice: Tuple[IntentScore, ...]) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "intent": item.intent,
            "matched_terms": list(item.matched_terms),
            "score": item.score,
        }
        for item in lattice
    )


def _graph_to_dict(graph: Tuple[IntentActionEdge, ...]) -> Tuple[Dict[str, str], ...]:
    return tuple(
        {
            "edge_type": edge.edge_type,
            "source": edge.source,
            "target": edge.target,
        }
        for edge in graph
    )


def _plan_to_dict(plan: GovernedResponsePlan) -> Dict[str, str]:
    return {
        "policy_mode": plan.policy_mode,
        "rationale": plan.rationale,
        "selected_action": plan.selected_action,
        "selected_intent": plan.selected_intent,
    }


def synthesize_deterministic_language_policy(
    text: str,
    *,
    policy_mode: str = "strict",
) -> GovernedOutputArtifact:
    """Full deterministic NLP governance synthesis pipeline."""
    _validate_mode(policy_mode)

    state = normalize_natural_language(text)
    lattice = compile_intent_lattice(state)
    plan = plan_governed_response(state, lattice, policy_mode=policy_mode)
    graph = build_state_intent_action_graph(state, lattice, plan)

    state_hash = _sha256_hex_from_payload(_state_to_dict(state))
    lattice_hash = _sha256_hex_from_payload(_lattice_to_dict(lattice))
    graph_hash = _sha256_hex_from_payload(_graph_to_dict(graph))
    plan_hash = _sha256_hex_from_payload(_plan_to_dict(plan))
    stable_hash = _sha256_hex_from_payload(
        {
            "graph_hash": graph_hash,
            "lattice_hash": lattice_hash,
            "plan_hash": plan_hash,
            "state_hash": state_hash,
            "version": DETERMINISTIC_NLP_GOVERNANCE_VERSION,
        }
    )
    identity_chain = (state_hash, lattice_hash, graph_hash, plan_hash, stable_hash)

    return GovernedOutputArtifact(
        version=DETERMINISTIC_NLP_GOVERNANCE_VERSION,
        state=state,
        lattice=lattice,
        graph=graph,
        plan=plan,
        identity_chain=identity_chain,
        stable_hash=stable_hash,
    )


def export_governed_output_bundle(artifact: GovernedOutputArtifact) -> Dict[str, Any]:
    """Canonical JSON-safe export for deterministic replay validation."""
    return {
        "graph": _graph_to_dict(artifact.graph),
        "identity_chain": list(artifact.identity_chain),
        "lattice": _lattice_to_dict(artifact.lattice),
        "plan": _plan_to_dict(artifact.plan),
        "stable_hash": artifact.stable_hash,
        "state": _state_to_dict(artifact.state),
        "version": artifact.version,
    }


def export_governed_output_bytes(artifact: GovernedOutputArtifact) -> bytes:
    """Canonical UTF-8 bytes export; same input always yields same bytes."""
    return _canonical_json(export_governed_output_bundle(artifact)).encode("utf-8")
