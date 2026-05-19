from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import skill_library_manifest as slm
from tests.test_agent_observation_trace_receipts import _receipt as _trace_receipt


def _mk_hash(payload, key):
    return slm._hash_payload(slm._base_payload(payload, key))


def _identity(library_type: str = "DECLARED_AGENT_SKILL_LIBRARY"):
    base = {"library_name": "skills", "library_version": "1.0", "library_type": library_type}
    return slm.SkillLibraryIdentity(**base, skill_library_identity_hash=_mk_hash(base, "skill_library_identity_hash"))


def _entry(i: int = 0, mode: str = "SKILL_DECLARED_NOT_EXECUTED", desc: str = "declared only"):
    base = {"skill_index": i, "skill_name": f"s{i}", "skill_mode": mode, "skill_description": desc}
    return slm.SkillDefinitionEntry(**base, skill_definition_entry_hash=_mk_hash(base, "skill_definition_entry_hash"))


def _version(mode: str = "FIXED_VERSION", reason: str = "pinned"):
    base = {"version_mode": mode, "version_reason": reason}
    return slm.SkillVersionDeclaration(**base, skill_version_declaration_hash=_mk_hash(base, "skill_version_declaration_hash"))


def _cap(mode: str = "CAPABILITY_CONTEXT_ONLY", reason: str = "declared"):
    base = {"capability_mode": mode, "capability_reason": reason}
    return slm.SkillCapabilityBoundary(**base, skill_capability_boundary_hash=_mk_hash(base, "skill_capability_boundary_hash"))


def _dep(mode: str = "NO_DEPENDENCIES", reason: str = "none"):
    base = {"dependency_mode": mode, "dependency_reason": reason}
    return slm.SkillDependencyBoundary(**base, skill_dependency_boundary_hash=_mk_hash(base, "skill_dependency_boundary_hash"))


def _manifest(adapter_only: bool = True, replay_upstream: bool = True):
    if replay_upstream:
        trace, m, kv, deps = _trace_receipt(sequence_mode="STRICT_ORDERED_SEQUENCE", declared_count=1, adapter_only=True)
        trace = replace(trace, tool_call_observations=(trace.tool_call_observations[0],), intermediate_decision_observations=tuple(), observation_sequence_boundary=replace(trace.observation_sequence_boundary, declared_step_count=1), replay_safe_observation_trace=True, agent_observation_trace_receipt_hash="0" * 64)
        trace = replace(trace, agent_observation_trace_receipt_hash=slm._hash_payload(slm._base_payload(trace.__dict__, "agent_observation_trace_receipt_hash")))
    else:
        trace, m, kv, deps = _trace_receipt(sequence_mode="STRICT_ORDERED_SEQUENCE", declared_count=2, adapter_only=True)
    entries = (_entry(0), _entry(1, "SKILL_DECLARED_CONTEXT_ONLY"))
    manifest = slm.build_skill_library_manifest(trace, _identity(), entries, _version(), _cap(), _dep(), adapter_only)
    return manifest, trace, dict(inference_backend_manifest=m, kv_cache_policy_receipt=kv, **deps)


def test_hash_stability_canonical_json_idempotent_and_hashseed_replay():
    a, _, _ = _manifest()
    b, _, _ = _manifest()
    assert a.skill_library_manifest_hash == b.skill_library_manifest_hash
    assert slm._canonical_json(a.__dict__) == slm._canonical_json(b.__dict__)


def test_skill_count_and_replay_safe_recomputed_not_trusted():
    m, trace, tkw = _manifest(adapter_only=True, replay_upstream=True)
    assert m.skill_count == len(m.skill_entries)
    assert m.replay_safe_skill_library is True
    slm.validate_skill_library_manifest(m, trace, **tkw)
    with pytest.raises(ValueError, match="skill_count mismatch"):
        slm.validate_skill_library_manifest(replace(m, skill_count=999), trace, **tkw)
    with pytest.raises(ValueError, match="recomputed"):
        slm.validate_skill_library_manifest(replace(m, replay_safe_skill_library=False), trace, **tkw)


@pytest.mark.parametrize("value", ["abc", "Z" * 64, "a" * 63])
def test_malformed_hash_rejected(value):
    with pytest.raises(ValueError):
        slm.SkillLibraryIdentity("n", "v", "DECLARED_AGENT_SKILL_LIBRARY", value)


@pytest.mark.parametrize("ctor,arg", [
    (_identity, "BAD"),
    (lambda x: _entry(0, x), "BAD"),
    (_version, "BAD"),
    (_cap, "BAD"),
    (_dep, "BAD"),
])
def test_invalid_modes_rejected(ctor, arg):
    with pytest.raises(ValueError):
        ctor(arg)


def test_exact_type_bool_int_alias_and_immutable_rejections():
    m, trace, tkw = _manifest()
    with pytest.raises(FrozenInstanceError):
        m.skill_count = 5
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        slm.validate_skill_library_manifest({}, trace, **tkw)
    with pytest.raises(ValueError):
        slm.validate_skill_library_manifest(replace(m, adapter_only=1), trace, **tkw)
    with pytest.raises(ValueError):
        slm.validate_skill_library_manifest(replace(m, skill_count=True), trace, **tkw)
    with pytest.raises(ValueError):
        slm.validate_skill_library_manifest(replace(m, skill_entries=list(m.skill_entries)), trace, **tkw)


def test_child_before_aggregate_and_index_ordering_enforced():
    m, trace, tkw = _manifest()
    bad_child = object.__new__(slm.SkillLibraryIdentity)
    object.__setattr__(bad_child, "library_name", "x")
    object.__setattr__(bad_child, "library_version", "y")
    object.__setattr__(bad_child, "library_type", "BAD")
    object.__setattr__(bad_child, "skill_library_identity_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid library_type"):
        slm.validate_skill_library_manifest(replace(m, skill_library_identity=bad_child, skill_library_manifest_hash="0" * 64), trace, **tkw)

    dup = replace(m, skill_entries=(_entry(0), _entry(0)), skill_library_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        slm.validate_skill_library_manifest(dup, trace, **tkw)
    oo = replace(m, skill_entries=(_entry(1), _entry(0)), skill_library_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="out-of-order"):
        slm.validate_skill_library_manifest(oo, trace, **tkw)
    nd = replace(m, skill_entries=(_entry(0), _entry(2)), skill_library_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="out-of-order"):
        slm.validate_skill_library_manifest(nd, trace, **tkw)


@pytest.mark.parametrize("text", [
    "skill executed",
    "tool execution succeeded",
    "runtime dispatch",
    "live crawler",
    "autonomous evaluation",
    "semantic equivalence guaranteed",
    "agent output is evidence",
])
def test_hidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _entry(0, desc=text)


def test_custom_modes_and_upstream_non_replay_safe_force_false():
    m, trace, tkw = _manifest()
    cver = replace(m, version_declaration=slm.build_skill_version_declaration("DECLARED_CUSTOM_VERSION", "declared"), replay_safe_skill_library=False, skill_library_manifest_hash="0" * 64)
    cver = replace(cver, skill_library_manifest_hash=slm._hash_payload(slm._base_payload(cver.__dict__, "skill_library_manifest_hash")))
    slm.validate_skill_library_manifest(cver, trace, **tkw)

    with pytest.raises(ValueError, match="recomputed"):
        wrong = replace(cver, replay_safe_skill_library=True, skill_library_manifest_hash="0" * 64)
        wrong = replace(wrong, skill_library_manifest_hash=slm._hash_payload(slm._base_payload(wrong.__dict__, "skill_library_manifest_hash")))
        slm.validate_skill_library_manifest(wrong, trace, **tkw)

    nonreplay, ntrace, ntkw = _manifest(replay_upstream=False)
    assert nonreplay.replay_safe_skill_library is False
    slm.validate_skill_library_manifest(nonreplay, ntrace, **ntkw)


def test_custom_skill_mode_forces_non_replay_safe():
    m, trace, tkw = _manifest()
    custom_entries = (_entry(0, "DECLARED_CUSTOM_SKILL_MODE"), _entry(1, "SKILL_DECLARED_CONTEXT_ONLY"))
    rebuilt = slm.build_skill_library_manifest(
        trace,
        m.skill_library_identity,
        custom_entries,
        m.version_declaration,
        m.capability_boundary,
        m.dependency_boundary,
        m.adapter_only,
    )
    assert rebuilt.replay_safe_skill_library is False
    slm.validate_skill_library_manifest(rebuilt, trace, **tkw)


def test_custom_library_type_forces_non_replay_safe():
    m, trace, tkw = _manifest()
    custom_identity = _identity("DECLARED_CUSTOM_SKILL_LIBRARY")
    rebuilt = slm.build_skill_library_manifest(
        trace,
        custom_identity,
        m.skill_entries,
        m.version_declaration,
        m.capability_boundary,
        m.dependency_boundary,
        m.adapter_only,
    )
    assert rebuilt.replay_safe_skill_library is False
    slm.validate_skill_library_manifest(rebuilt, trace, **tkw)


def test_invalid_skill_ordering_forces_non_replay_safe_in_builder():
    m, trace, tkw = _manifest()
    bad_entries = (_entry(0), _entry(0))
    rebuilt = slm.build_skill_library_manifest(
        trace,
        m.skill_library_identity,
        bad_entries,
        m.version_declaration,
        m.capability_boundary,
        m.dependency_boundary,
        m.adapter_only,
    )
    assert rebuilt.replay_safe_skill_library is False
    with pytest.raises(ValueError, match="duplicate"):
        slm.validate_skill_library_manifest(rebuilt, trace, **tkw)


def test_upstream_validation_and_forbidden_imports():
    m, trace, tkw = _manifest()
    assert slm.validate_skill_library_manifest(m, trace, **tkw) == m
    source = (Path(__file__).parent.parent / "src/qec/analysis/skill_library_manifest.py").read_text(encoding="utf-8")
    for token in ("transformers", "torch", "tensorflow", "requests", "urllib", "selenium", "playwright", "subprocess", "asyncio", "multiprocessing", "importlib", "os.system"):
        assert token not in source
