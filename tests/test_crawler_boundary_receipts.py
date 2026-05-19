from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import crawler_boundary_receipts as cbr
from tests.test_tool_dispatch_telemetry_receipts import _receipt as _dispatch_receipt


def _mk_hash(payload, key):
    return cbr._hash_payload(cbr._base_payload(payload, key))


def _crawler_identity(crawler_type: str = "DECLARED_AUDIT_CRAWLER"):
    base = {"crawler_name": "declared-crawler", "crawler_version": "1.0", "crawler_type": crawler_type}
    return cbr.CrawlerIdentity(**base, crawler_identity_hash=_mk_hash(base, "crawler_identity_hash"))


def _scope(mode: str = "NO_NETWORK_SCOPE", reason: str = "deterministic scope"):
    base = {"scope_mode": mode, "scope_reason": reason}
    return cbr.CrawlScopeBoundary(**base, crawl_scope_boundary_hash=_mk_hash(base, "crawl_scope_boundary_hash"))


def _perm(mode: str = "NETWORK_DISABLED", reason: str = "declared boundary"):
    base = {"permission_mode": mode, "permission_reason": reason}
    return cbr.CrawlPermissionBoundary(**base, crawl_permission_boundary_hash=_mk_hash(base, "crawl_permission_boundary_hash"))


def _replay(mode: str = "REPLAY_SAFE_CRAWL", reason: str = "replay constrained"):
    base = {"replay_mode": mode, "replay_reason": reason}
    return cbr.CrawlReplayBoundary(**base, crawl_replay_boundary_hash=_mk_hash(base, "crawl_replay_boundary_hash"))


def _audit(mode: str = "STRICT_AUDIT_BOUNDARY", reason: str = "strict replay audit"):
    base = {"audit_mode": mode, "audit_reason": reason}
    return cbr.CrawlAuditBoundary(**base, crawl_audit_boundary_hash=_mk_hash(base, "crawl_audit_boundary_hash"))


def _receipt(adapter_only: bool = True, replay_upstream: bool = True):
    dispatch, manifest, trace, deps = _dispatch_receipt(adapter_only=True, replay_upstream=replay_upstream)
    rec = cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay(), _audit(), adapter_only)
    return rec, dispatch, manifest, trace, deps


def test_hash_canonical_hashseed_and_idempotent_rebuild_stability():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert a.crawler_boundary_receipt_hash == b.crawler_boundary_receipt_hash
    assert cbr._canonical_json(a.__dict__) == cbr._canonical_json(b.__dict__)


def test_replay_safe_recomputed_not_trusted_and_upstream_validation_propagates():
    rec, dispatch, manifest, trace, deps = _receipt()
    assert rec.replay_safe_crawler is True
    cbr.validate_crawler_boundary_receipt(rec, dispatch, manifest, agent_observation_trace_receipt=trace, **deps)
    forged = replace(rec, replay_safe_crawler=False, crawler_boundary_receipt_hash="0" * 64)
    forged = replace(forged, crawler_boundary_receipt_hash=cbr._hash_payload(cbr._base_payload(forged.__dict__, "crawler_boundary_receipt_hash")))
    with pytest.raises(ValueError, match="recomputed"):
        cbr.validate_crawler_boundary_receipt(forged, dispatch, manifest, agent_observation_trace_receipt=trace, **deps)


@pytest.mark.parametrize("value", ["abc", "Z" * 64, "a" * 63])
def test_malformed_hash_rejected(value):
    with pytest.raises(ValueError):
        cbr.CrawlerIdentity("x", "1", "DECLARED_AUDIT_CRAWLER", value)


@pytest.mark.parametrize("fn,arg", [(_crawler_identity, "BAD"), (_scope, "BAD"), (_perm, "BAD"), (_replay, "BAD"), (_audit, "BAD")])
def test_invalid_modes_rejected(fn, arg):
    with pytest.raises(ValueError):
        fn(arg)


def test_exact_type_immutable_and_bool_int_alias_rejected():
    rec, dispatch, manifest, trace, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        cbr.validate_crawler_boundary_receipt({}, dispatch, manifest, agent_observation_trace_receipt=trace, **deps)
    with pytest.raises(ValueError):
        cbr.validate_crawler_boundary_receipt(replace(rec, adapter_only=1), dispatch, manifest, agent_observation_trace_receipt=trace, **deps)
    with pytest.raises(ValueError):
        cbr.validate_crawler_boundary_receipt(replace(rec, replay_safe_crawler=1), dispatch, manifest, agent_observation_trace_receipt=trace, **deps)


def test_child_before_aggregate_validation_enforced():
    rec, dispatch, manifest, trace, deps = _receipt()
    bad_child = object.__new__(cbr.CrawlerIdentity)
    object.__setattr__(bad_child, "crawler_name", "x")
    object.__setattr__(bad_child, "crawler_version", "1")
    object.__setattr__(bad_child, "crawler_type", "BAD")
    object.__setattr__(bad_child, "crawler_identity_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid crawler_type"):
        cbr.validate_crawler_boundary_receipt(
            replace(rec, crawler_identity=bad_child, crawler_boundary_receipt_hash="0" * 64),
            dispatch,
            manifest,
            agent_observation_trace_receipt=trace,
            **deps,
        )


@pytest.mark.parametrize("text", [
    "network request succeeded",
    "live crawler",
    "autonomous evaluation",
    "semantic equivalence guaranteed",
    "agent output is evidence",
    "hidden crawler execution",
    "hidden network semantics",
    "hidden replay equivalence",
    "hidden mutable crawler",
])
def test_hidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _scope(reason=text)


def test_replay_safe_rejection_for_custom_modes_and_contextual_modes_and_audit_enforcement():
    rec, dispatch, manifest, trace, deps = _receipt()
    for custom in (
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity("DECLARED_CUSTOM_CRAWLER"), _scope(), _perm(), _replay(), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope("DECLARED_CUSTOM_SCOPE"), _perm(), _replay(), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm("DECLARED_CUSTOM_PERMISSION"), _replay(), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay("DECLARED_CUSTOM_REPLAY"), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay(), _audit("DECLARED_CUSTOM_AUDIT"), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay("CONTEXTUAL_CRAWL"), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay("NON_REPLAY_SAFE_CRAWL"), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay("RESEARCH_CRAWL_ONLY"), _audit(), True),
        cbr.build_crawler_boundary_receipt(dispatch, manifest, _crawler_identity(), _scope(), _perm(), _replay(), _audit("CONTEXT_AUDIT_BOUNDARY"), True),
    ):
        assert custom.replay_safe_crawler is False
        cbr.validate_crawler_boundary_receipt(custom, dispatch, manifest, agent_observation_trace_receipt=trace, **deps)


def test_non_replay_safe_upstream_propagates_false_and_adapter_only_required():
    non, dispatch, manifest, trace, deps = _receipt(replay_upstream=False)
    assert non.replay_safe_crawler is False
    cbr.validate_crawler_boundary_receipt(non, dispatch, manifest, agent_observation_trace_receipt=trace, **deps)
    off, dispatch2, manifest2, trace2, deps2 = _receipt(adapter_only=False)
    assert off.replay_safe_crawler is False
    cbr.validate_crawler_boundary_receipt(off, dispatch2, manifest2, agent_observation_trace_receipt=trace2, **deps2)


def test_no_forbidden_imports_runtime_inference_network_or_subprocess():
    source = (Path(__file__).parent.parent / "src/qec/analysis/crawler_boundary_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers", "torch", "tensorflow", "requests", "urllib", "aiohttp", "selenium", "playwright", "scrapy", "bs4",
        "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec(",
    ):
        assert token not in source
