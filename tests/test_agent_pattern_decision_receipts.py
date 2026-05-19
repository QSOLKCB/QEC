from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import agent_pattern_decision_receipts as apdr
from tests.test_crawler_boundary_receipts import _receipt as _crawler_receipt


def _mk_hash(payload, key):
    return apdr._hash_payload(apdr._base_payload(payload, key))


def _identity(pattern_type: str = "DECLARED_AUDIT_PATTERN"):
    base = {"pattern_name": "declared-pattern", "pattern_version": "1.0", "pattern_type": pattern_type}
    return apdr.AgentPatternIdentity(**base, agent_pattern_identity_hash=_mk_hash(base, "agent_pattern_identity_hash"))


def _selection(mode: str = "PATTERN_DECLARED_BEFORE_EXECUTION", reason: str = "deterministic declaration before execution"):
    base = {"selection_mode": mode, "selection_reason": reason}
    return apdr.PatternSelectionDeclaration(**base, pattern_selection_declaration_hash=_mk_hash(base, "pattern_selection_declaration_hash"))


def _decision(mode: str = "DECISION_BOUNDARY_DECLARED_ONLY", reason: str = "declared boundary"):
    base = {"decision_boundary_mode": mode, "decision_boundary_reason": reason}
    return apdr.PatternDecisionBoundary(**base, pattern_decision_boundary_hash=_mk_hash(base, "pattern_decision_boundary_hash"))


def _execution(mode: str = "PATTERN_NOT_EXECUTED", reason: str = "receipt only"):
    base = {"execution_boundary_mode": mode, "execution_boundary_reason": reason}
    return apdr.PatternExecutionBoundary(**base, pattern_execution_boundary_hash=_mk_hash(base, "pattern_execution_boundary_hash"))


def _audit(mode: str = "STRICT_PATTERN_AUDIT", reason: str = "strict replay-safe audit"):
    base = {"audit_boundary_mode": mode, "audit_boundary_reason": reason}
    return apdr.PatternAuditBoundary(**base, pattern_audit_boundary_hash=_mk_hash(base, "pattern_audit_boundary_hash"))


def _receipt(adapter_only: bool = True, replay_upstream: bool = True):
    crawler, dispatch, manifest, trace, deps = _crawler_receipt(adapter_only=True, replay_upstream=replay_upstream)
    rec = apdr.build_agent_pattern_decision_receipt(
        trace, manifest, dispatch, crawler, _identity(), _selection(), _decision(), _execution(), _audit(), adapter_only
    )
    return rec, trace, manifest, dispatch, crawler, deps


def test_hash_and_canonical_stability_hashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert a.agent_pattern_decision_receipt_hash == b.agent_pattern_decision_receipt_hash
    assert apdr._canonical_json(a.__dict__) == apdr._canonical_json(b.__dict__)


def test_replay_safe_recomputed_not_trusted():
    rec, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_pattern_decision is False
    apdr.validate_agent_pattern_decision_receipt(rec, trace, manifest, dispatch, crawler, **deps)
    forged = replace(rec, replay_safe_pattern_decision=True, agent_pattern_decision_receipt_hash="0" * 64)
    forged = replace(forged, agent_pattern_decision_receipt_hash=apdr._hash_payload(apdr._base_payload(forged.__dict__, "agent_pattern_decision_receipt_hash")))
    with pytest.raises(ValueError, match="recomputed"):
        apdr.validate_agent_pattern_decision_receipt(forged, trace, manifest, dispatch, crawler, **deps)


@pytest.mark.parametrize("value", ["abc", "Z" * 64, "a" * 63])
def test_malformed_hash_rejected(value):
    with pytest.raises(ValueError):
        apdr.AgentPatternIdentity("x", "1", "DECLARED_AUDIT_PATTERN", value)


@pytest.mark.parametrize(
    "fn,arg",
    [(_identity, "BAD"), (_selection, "BAD"), (_decision, "BAD"), (_execution, "BAD"), (_audit, "BAD")],
)
def test_invalid_modes_rejected(fn, arg):
    with pytest.raises(ValueError):
        fn(arg)


def test_exact_type_immutable_and_bool_int_alias_rejected():
    rec, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        apdr.validate_agent_pattern_decision_receipt({}, trace, manifest, dispatch, crawler, **deps)
    with pytest.raises(ValueError):
        apdr.validate_agent_pattern_decision_receipt(replace(rec, adapter_only=1), trace, manifest, dispatch, crawler, **deps)
    with pytest.raises(ValueError):
        apdr.validate_agent_pattern_decision_receipt(replace(rec, replay_safe_pattern_decision=1), trace, manifest, dispatch, crawler, **deps)


def test_child_before_aggregate_validation_enforced():
    rec, trace, manifest, dispatch, crawler, deps = _receipt()
    bad_child = object.__new__(apdr.AgentPatternIdentity)
    object.__setattr__(bad_child, "pattern_name", "x")
    object.__setattr__(bad_child, "pattern_version", "1")
    object.__setattr__(bad_child, "pattern_type", "BAD")
    object.__setattr__(bad_child, "agent_pattern_identity_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid pattern_type"):
        apdr.validate_agent_pattern_decision_receipt(
            replace(rec, pattern_identity=bad_child, agent_pattern_decision_receipt_hash="0" * 64),
            trace,
            manifest,
            dispatch,
            crawler,
            **deps,
        )


@pytest.mark.parametrize(
    "text",
    [
        "runtime dispatch",
        "tool execution succeeded",
        "live crawler",
        "autonomous planning",
        "autonomous evaluation",
        "semantic equivalence guaranteed",
        "agent output is evidence",
        "agent output as evidence",
        "hidden runtime execution",
        "hidden tool execution",
        "hidden tool call",
        "hidden crawler execution",
        "autonomous network crawling",
        "hidden network semantics",
        "hidden replay equivalence",
        "hidden mutable pattern",
    ],
)
def test_hidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _selection(reason=text)


def test_replay_safe_rejection_for_custom_modes_context_modes_and_non_replay_lineage():
    rec, trace, manifest, dispatch, crawler, deps = _receipt()
    for custom in (
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity("DECLARED_CUSTOM_PATTERN"), _selection(), _decision(), _execution(), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection("DECLARED_CUSTOM_SELECTION"), _decision(), _execution(), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision("DECLARED_CUSTOM_DECISION_BOUNDARY"), _execution(), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision(), _execution("DECLARED_CUSTOM_EXECUTION_BOUNDARY"), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision(), _execution(), _audit("DECLARED_CUSTOM_PATTERN_AUDIT"), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection("PATTERN_DECLARED_CONTEXT_ONLY"), _decision(), _execution(), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision("DECISION_BOUNDARY_CONTEXT_ONLY"), _execution(), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision(), _execution("PATTERN_CONTEXT_ONLY"), _audit(), True),
        apdr.build_agent_pattern_decision_receipt(trace, manifest, dispatch, crawler, _identity(), _selection(), _decision(), _execution(), _audit("CONTEXT_PATTERN_AUDIT"), True),
    ):
        assert custom.replay_safe_pattern_decision is False
        apdr.validate_agent_pattern_decision_receipt(custom, trace, manifest, dispatch, crawler, **deps)

    non, trace2, manifest2, dispatch2, crawler2, deps2 = _receipt(replay_upstream=False)
    assert non.replay_safe_pattern_decision is False
    apdr.validate_agent_pattern_decision_receipt(non, trace2, manifest2, dispatch2, crawler2, **deps2)


def test_replay_safe_requires_pattern_identity_match_with_trace_declaration():
    rec, trace, manifest, dispatch, crawler, deps = _receipt()
    mismatched = apdr.build_agent_pattern_decision_receipt(
        trace,
        manifest,
        dispatch,
        crawler,
        _identity("DECLARED_AUDIT_PATTERN"),
        _selection(),
        _decision(),
        _execution(),
        _audit(),
        True,
    )
    assert trace.pattern_declaration.pattern_type == "SEQUENTIAL_TOOL_PATTERN"
    assert mismatched.pattern_identity.pattern_type != trace.pattern_declaration.pattern_type
    assert rec.replay_safe_pattern_decision is False
    assert mismatched.replay_safe_pattern_decision is False
    apdr.validate_agent_pattern_decision_receipt(mismatched, trace, manifest, dispatch, crawler, **deps)


def test_pattern_declared_before_execution_boundary_and_replay_safe_audit_enforcement():
    rec, *_ = _receipt()
    assert rec.selection_declaration.selection_mode == "PATTERN_DECLARED_BEFORE_EXECUTION"
    assert rec.execution_boundary.execution_boundary_mode in {"PATTERN_NOT_EXECUTED", "PATTERN_REPLAY_ONLY", "PATTERN_AUDIT_ONLY"}
    assert rec.audit_boundary.audit_boundary_mode in {"STRICT_PATTERN_AUDIT", "REPLAY_PATTERN_AUDIT"}


def test_no_forbidden_imports_runtime_inference_network_or_subprocess():
    source = (Path(__file__).parent.parent / "src/qec/analysis/agent_pattern_decision_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers", "torch", "tensorflow", "requests", "urllib", "aiohttp", "selenium", "playwright", "scrapy", "bs4",
        "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec(",
    ):
        assert token not in source
