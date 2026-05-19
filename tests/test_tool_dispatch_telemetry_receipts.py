from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import tool_dispatch_telemetry_receipts as tdr
from tests.test_skill_library_manifest import _manifest as _skill_manifest


def _mk_hash(payload, key):
    return tdr._hash_payload(tdr._base_payload(payload, key))


def _identity(tool_type: str = "DECLARED_ANALYSIS_TOOL"):
    base = {"tool_name": "dispatch-tool", "tool_version": "1.0", "tool_type": tool_type}
    return tdr.ToolDispatchIdentity(**base, tool_dispatch_identity_hash=_mk_hash(base, "tool_dispatch_identity_hash"))


def _in(mode: str = "HASH_BOUND_INPUT", reason: str = "hash bound"):
    base = {"input_mode": mode, "input_hash": "1" * 64, "input_reason": reason}
    return tdr.ToolDispatchInputBoundary(**base, tool_dispatch_input_boundary_hash=_mk_hash(base, "tool_dispatch_input_boundary_hash"))


def _out(mode: str = "HASH_BOUND_OUTPUT", reason: str = "hash bound"):
    base = {"output_mode": mode, "output_hash": "2" * 64, "output_reason": reason}
    return tdr.ToolDispatchOutputBoundary(**base, tool_dispatch_output_boundary_hash=_mk_hash(base, "tool_dispatch_output_boundary_hash"))


def _exe(mode: str = "TOOL_DECLARED_REPLAY_ONLY", ms: int = 0, reason: str = "declared deterministic"):
    base = {"execution_mode": mode, "declared_execution_time_ms": ms, "execution_reason": reason}
    return tdr.ToolDispatchExecutionBoundary(**base, tool_dispatch_execution_boundary_hash=_mk_hash(base, "tool_dispatch_execution_boundary_hash"))


def _audit(mode: str = "STRICT_AUDIT_TRAIL", reason: str = "strict"):
    base = {"audit_mode": mode, "audit_reason": reason}
    return tdr.ToolDispatchAuditBoundary(**base, tool_dispatch_audit_boundary_hash=_mk_hash(base, "tool_dispatch_audit_boundary_hash"))


def _receipt(adapter_only: bool = True, replay_upstream: bool = True):
    manifest, trace, deps = _skill_manifest(adapter_only=True, replay_upstream=replay_upstream)
    rec = tdr.build_tool_dispatch_telemetry_receipt(manifest, trace, _identity(), _in(), _out(), _exe(), _audit(), adapter_only)
    return rec, manifest, trace, deps


def test_hash_canonical_stability_hashseed_and_idempotent_rebuild():
    a, _, _, _ = _receipt()
    b, _, _, _ = _receipt()
    assert a.tool_dispatch_telemetry_receipt_hash == b.tool_dispatch_telemetry_receipt_hash
    assert tdr._canonical_json(a.__dict__) == tdr._canonical_json(b.__dict__)


def test_replay_safe_recomputed_and_not_trusted():
    rec, manifest, trace, deps = _receipt()
    assert rec.replay_safe_dispatch is True
    tdr.validate_tool_dispatch_telemetry_receipt(rec, manifest, trace, **deps)
    forged = replace(rec, replay_safe_dispatch=False, tool_dispatch_telemetry_receipt_hash="0" * 64)
    forged = replace(forged, tool_dispatch_telemetry_receipt_hash=tdr._hash_payload(tdr._base_payload(forged.__dict__, "tool_dispatch_telemetry_receipt_hash")))
    with pytest.raises(ValueError, match="recomputed"):
        tdr.validate_tool_dispatch_telemetry_receipt(forged, manifest, trace, **deps)


@pytest.mark.parametrize("value", ["abc", "Z" * 64, "a" * 63])
def test_malformed_hash_rejected(value):
    with pytest.raises(ValueError):
        tdr.ToolDispatchIdentity("n", "v", "DECLARED_ANALYSIS_TOOL", value)


@pytest.mark.parametrize("ctor,arg", [
    (_identity, "BAD"),
    (lambda x: _in(x), "BAD"),
    (lambda x: _out(x), "BAD"),
    (lambda x: _exe(x), "BAD"),
    (lambda x: _audit(x), "BAD"),
])
def test_invalid_modes_rejected(ctor, arg):
    with pytest.raises(ValueError):
        ctor(arg)


def test_exact_type_bool_int_alias_and_immutable_rejections():
    rec, manifest, trace, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        tdr.validate_tool_dispatch_telemetry_receipt({}, manifest, trace, **deps)
    with pytest.raises(ValueError):
        tdr.validate_tool_dispatch_telemetry_receipt(replace(rec, adapter_only=1), manifest, trace, **deps)
    with pytest.raises(ValueError):
        tdr.validate_tool_dispatch_telemetry_receipt(replace(rec, replay_safe_dispatch=1), manifest, trace, **deps)
    with pytest.raises(ValueError):
        tdr.validate_tool_dispatch_telemetry_receipt(replace(rec, execution_boundary=replace(rec.execution_boundary, declared_execution_time_ms=True, tool_dispatch_execution_boundary_hash="0" * 64)), manifest, trace, **deps)


def test_child_before_aggregate_validation_and_negative_execution_time_rejected():
    rec, manifest, trace, deps = _receipt()
    bad_child = object.__new__(tdr.ToolDispatchIdentity)
    object.__setattr__(bad_child, "tool_name", "x")
    object.__setattr__(bad_child, "tool_version", "y")
    object.__setattr__(bad_child, "tool_type", "BAD")
    object.__setattr__(bad_child, "tool_dispatch_identity_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid tool_type"):
        tdr.validate_tool_dispatch_telemetry_receipt(replace(rec, dispatch_identity=bad_child, tool_dispatch_telemetry_receipt_hash="0" * 64), manifest, trace, **deps)
    with pytest.raises(ValueError):
        _exe(ms=-1)


@pytest.mark.parametrize("text", [
    "tool execution succeeded",
    "runtime dispatch",
    "live crawler",
    "autonomous evaluation",
    "semantic equivalence guaranteed",
    "hidden mutable dispatch",
    "agent output is evidence",
])
def test_hidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _in(reason=text)


def test_replay_safe_false_for_custom_modes_context_audit_and_nonreplay_lineage():
    rec, manifest, trace, deps = _receipt()
    c = tdr.build_tool_dispatch_telemetry_receipt(manifest, trace, _identity(), _in("DECLARED_CUSTOM_INPUT"), _out(), _exe(), _audit(), True)
    assert c.replay_safe_dispatch is False
    tdr.validate_tool_dispatch_telemetry_receipt(c, manifest, trace, **deps)

    a = tdr.build_tool_dispatch_telemetry_receipt(manifest, trace, _identity(), _in(), _out(), _exe(), _audit("DECLARED_CONTEXT_AUDIT"), True)
    assert a.replay_safe_dispatch is False

    non, nmanifest, ntrace, ndeps = _receipt(replay_upstream=False)
    assert non.replay_safe_dispatch is False
    tdr.validate_tool_dispatch_telemetry_receipt(non, nmanifest, ntrace, **ndeps)


def test_upstream_validation_and_forbidden_imports():
    rec, manifest, trace, deps = _receipt()
    assert tdr.validate_tool_dispatch_telemetry_receipt(rec, manifest, trace, **deps) == rec
    source = (Path(__file__).parent.parent / "src/qec/analysis/tool_dispatch_telemetry_receipts.py").read_text(encoding="utf-8")
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
        "eval(",
        "exec(",
    ):
        assert token not in source
