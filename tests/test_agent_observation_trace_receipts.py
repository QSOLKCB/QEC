from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import agent_observation_trace_receipts as aotr
from tests.test_kv_cache_policy_receipts import _receipt as _kv_receipt


def _mk_hash(payload, key):
    return aotr._hash_payload(aotr._base_payload(payload, key))


def _agent_identity(agent_type: str = "DECLARED_ASSISTANT_AGENT"):
    base = {"agent_name": "agent", "agent_version": "1.0", "agent_type": agent_type}
    return aotr.AgentIdentity(**base, agent_identity_hash=_mk_hash(base, "agent_identity_hash"))


def _pattern(pattern_type: str = "SEQUENTIAL_TOOL_PATTERN", reason: str = "declared"):
    base = {"pattern_type": pattern_type, "pattern_reason": reason}
    return aotr.AgentPatternDeclaration(**base, agent_pattern_declaration_hash=_mk_hash(base, "agent_pattern_declaration_hash"))


def _tool(mode: str = "TOOL_DECLARED_NOT_EXECUTED", reason: str = "declared"):
    base = {
        "tool_name": "search",
        "tool_mode": mode,
        "tool_input_hash": "a" * 64,
        "tool_output_hash": "b" * 64,
        "tool_reason": reason,
    }
    return aotr.ToolCallObservation(**base, tool_call_observation_hash=_mk_hash(base, "tool_call_observation_hash"))


def _decision(idx: int, mode: str = "DECLARED_PATTERN_DECISION", reason: str = "declared"):
    base = {"decision_index": idx, "decision_mode": mode, "decision_reason": reason}
    return aotr.IntermediateDecisionObservation(
        **base,
        intermediate_decision_observation_hash=_mk_hash(base, "intermediate_decision_observation_hash"),
    )


def _boundary(mode: str = "STRICT_ORDERED_SEQUENCE", count: int = 2, reason: str = "declared"):
    base = {"sequence_mode": mode, "declared_step_count": count, "sequence_reason": reason}
    return aotr.ObservationSequenceBoundary(
        **base,
        observation_sequence_boundary_hash=_mk_hash(base, "observation_sequence_boundary_hash"),
    )


def _receipt(sequence_mode: str = "STRICT_ORDERED_SEQUENCE", declared_count: int = 2, adapter_only: bool = True):
    kv, m, bw, c, t, b = _kv_receipt()
    ai = _agent_identity()
    pd = _pattern()
    tools = (_tool(),)
    decisions = (_decision(0),)
    ob = _boundary(sequence_mode, declared_count)
    r = aotr.build_agent_observation_trace_receipt(m, kv, ai, pd, tools, decisions, ob, adapter_only)
    return r, m, kv, dict(inference_memory_bandwidth_receipt=bw, parameter_golf_compression_receipt=c, tokenization_policy_receipt=t, byte_level_model_boundary_receipt=b)


def test_hash_canonical_replay_recompute_and_idempotent_rebuild():
    r1, m, kv, deps = _receipt()
    r2, *_ = _receipt()
    assert r1.agent_observation_trace_receipt_hash == r2.agent_observation_trace_receipt_hash
    assert aotr._canonical_json(r1.__dict__) == aotr._canonical_json(r2.__dict__)
    assert r1.replay_safe_observation_trace is True
    assert aotr.validate_agent_observation_trace_receipt(r1, m, kv, **deps) == r1
    with pytest.raises(ValueError):
        aotr.validate_agent_observation_trace_receipt(replace(r1, replay_safe_observation_trace=False), m, kv, **deps)


def test_replay_safe_false_for_contextual_custom_and_count_mismatch_and_strict_enforcement():
    r, m, kv, deps = _receipt(sequence_mode="DECLARED_CONTEXTUAL_SEQUENCE")
    assert r.replay_safe_observation_trace is False
    r2, _, _, _ = _receipt(sequence_mode="DECLARED_CUSTOM_SEQUENCE")
    assert r2.replay_safe_observation_trace is False
    bad = replace(r, observation_sequence_boundary=_boundary("STRICT_ORDERED_SEQUENCE", 99))
    with pytest.raises(ValueError):
        aotr.validate_agent_observation_trace_receipt(bad, m, kv, **deps)


@pytest.mark.parametrize(
    "ctor,arg",
    [
        (_agent_identity, "BAD"),
        (_pattern, "BAD"),
        (_tool, "BAD"),
        (lambda x: _decision(0, x), "BAD"),
        (lambda x: _boundary(x, 2), "BAD"),
    ],
)
def test_invalid_enums_rejected(ctor, arg):
    with pytest.raises(ValueError):
        ctor(arg)


def test_exact_type_bool_int_alias_immutable_and_malformed_hash_rejection():
    r, m, kv, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        aotr.validate_agent_observation_trace_receipt({}, m, kv, **deps)
    with pytest.raises(ValueError):
        aotr.validate_agent_observation_trace_receipt(replace(r, adapter_only=1), m, kv, **deps)
    bad_child = object.__new__(aotr.AgentIdentity)
    object.__setattr__(bad_child, "agent_name", "agent")
    object.__setattr__(bad_child, "agent_version", "1.0")
    object.__setattr__(bad_child, "agent_type", "DECLARED_ASSISTANT_AGENT")
    object.__setattr__(bad_child, "agent_identity_hash", "a" * 63)
    with pytest.raises(ValueError):
        aotr.validate_agent_observation_trace_receipt(replace(r, agent_identity=bad_child), m, kv, **deps)


def test_child_before_aggregate_duplicate_and_out_of_order_decisions_rejected():
    r, m, kv, deps = _receipt()
    bad_dup = replace(r, intermediate_decision_observations=(_decision(0), _decision(0)), agent_observation_trace_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        aotr.validate_agent_observation_trace_receipt(bad_dup, m, kv, **deps)
    bad_order = replace(r, intermediate_decision_observations=(_decision(1), _decision(0)), agent_observation_trace_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="out-of-order"):
        aotr.validate_agent_observation_trace_receipt(bad_order, m, kv, **deps)


@pytest.mark.parametrize(
    "text",
    [
        "tool execution succeeded",
        "runtime dispatch",
        "live crawler",
        "autonomous evaluation",
        "hidden replay equivalence",
        "hidden tool call",
        "agent output as evidence",
    ],
)
def test_hidden_semantics_and_forbidden_content_rejected(text):
    with pytest.raises(ValueError):
        _tool(reason=text)


def test_upstream_validations_called_and_hashseed_stability_and_no_forbidden_imports():
    r, m, kv, deps = _receipt()
    assert aotr.validate_agent_observation_trace_receipt(r, m, kv, **deps) == r
    assert _receipt()[0].agent_observation_trace_receipt_hash == _receipt()[0].agent_observation_trace_receipt_hash
    source = (Path(__file__).parent.parent / "src/qec/analysis/agent_observation_trace_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers",
        "torch",
        "tensorflow",
        "requests",
        "urllib",
        "selenium",
        "playwright",
        "subprocess",
        "asyncio",
        "multiprocessing",
        "os.system",
    ):
        assert token not in source
