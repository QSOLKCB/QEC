from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import byte_level_model_boundary_receipts as bl
from qec.analysis import inference_backend_manifest as ibm
from qec.analysis import inference_memory_bandwidth_receipts as imbr
from qec.analysis import kv_cache_policy_receipts as kcp
from qec.analysis import parameter_golf_compression_receipts as pg
from qec.analysis import tokenization_policy_receipts as tpr


def _manifest():
    return ibm.build_inference_backend_manifest(
        ibm.build_inference_backend_identity("backend", "1.0", "BYTE_LEVEL_MODEL"),
        ibm.build_tokenization_policy_declaration("BYTE_LEVEL", "tok", "declared"),
        ibm.build_quantization_declaration("NO_QUANTIZATION", "none"),
        ibm.build_precision_format_declaration("FP32", "full"),
        ibm.build_hardware_kernel_boundary("ADAPTER_ONLY_HARDWARE", "boundary", True),
        ibm.build_benchmark_corpus_declaration("MEMORY_CONTEXT_ONLY", "none", "context"),
        ibm.build_kv_cache_declaration("NO_CACHE", "none"),
    )


def _compression_bundle(manifest):
    byte_receipt = bl.build_byte_level_model_boundary_receipt(
        manifest,
        bl.build_byte_level_model_identity("family", "PURE_BYTE_MODEL", "declared"),
        bl.build_patch_segmentation_declaration("FIXED_PATCH", 64, "declared"),
        bl.build_byte_window_declaration(128, True, "declared"),
        bl.build_entropy_threshold_declaration("FIXED_THRESHOLD", 0.0, "declared"),
        bl.build_byte_normalization_policy("NO_NORMALIZATION", "declared"),
        bl.build_tokenizer_bypass_declaration("TOKENIZER_REMOVED", "declared"),
        True,
    )
    token_receipt = tpr.build_tokenization_policy_receipt(
        manifest,
        byte_receipt,
        tpr.build_tokenizer_identity("tok", "1", "BYTE_LEVEL_TOKENIZER"),
        tpr.build_vocabulary_policy_declaration("BYTE_NATIVE_VOCABULARY", 256, "declared"),
        tpr.build_merge_rule_declaration("BYTE_NATIVE_SEGMENTATION", "declared"),
        tpr.build_token_boundary_declaration("BYTE_COMPATIBLE_BOUNDARY", "declared"),
        tpr.build_tokenizer_equivalence_declaration("NO_EQUIVALENCE_CLAIM", "declared"),
        tpr.build_tokenizer_drift_boundary("DRIFT_PROHIBITED", "declared"),
        True,
    )

    def with_hash(obj, key):
        return replace(obj, **{key: pg._hash_payload(pg._base_payload(obj.__dict__, key))})

    ci = with_hash(pg.CompressionIdentity("golf", "1", "PARAMETER_GOLF", ""), "compression_identity_hash")
    pr = with_hash(
        pg.ParameterReductionDeclaration("FIXED_PARAMETER_REDUCTION", 100, 50, "declared", ""),
        "parameter_reduction_declaration_hash",
    )
    qd = with_hash(
        pg.QuantizationCompressionDeclaration("FP16", "declared", ""),
        "quantization_compression_declaration_hash",
    )
    pd = with_hash(
        pg.PrecisionReductionDeclaration("FP32_TO_FP16", "declared", ""),
        "precision_reduction_declaration_hash",
    )
    eq = with_hash(
        pg.CompressionEquivalenceBoundary("NO_EQUIVALENCE_CLAIM", "declared", ""),
        "compression_equivalence_boundary_hash",
    )
    bm = with_hash(
        pg.CompressionBenchmarkBoundary("NO_BENCHMARK_DECLARED", "declared", ""),
        "compression_benchmark_boundary_hash",
    )
    compression = pg.build_parameter_golf_compression_receipt(manifest, token_receipt, byte_receipt, ci, pr, qd, pd, eq, bm, True)

    bw = imbr.InferenceMemoryBandwidthReceipt(
        schema_version="INFERENCE_MEMORY_BANDWIDTH_RECEIPT_V1",
        inference_backend_manifest_hash=manifest.inference_backend_manifest_hash,
        parameter_golf_compression_receipt_hash=compression.parameter_golf_compression_receipt_hash,
        memory_bandwidth_identity=imbr.MemoryBandwidthIdentity("profile", "1", ""),
        memory_transfer_declaration=imbr.MemoryTransferDeclaration("STATIC_MEMORY_LAYOUT", "declared", ""),
        cache_bandwidth_declaration=imbr.CacheBandwidthDeclaration("NO_CACHE", "declared", ""),
        precision_bandwidth_declaration=imbr.PrecisionBandwidthDeclaration("FP32_MEMORY", "declared", ""),
        hardware_bandwidth_boundary=imbr.HardwareBandwidthBoundary("ADAPTER_ONLY_HARDWARE", "declared", ""),
        benchmark_bandwidth_boundary=imbr.BenchmarkBandwidthBoundary(
            "MEMORY_CONTEXT_ONLY", "declared", 1.0, "TOKENS_PER_SECOND", ""
        ),
        reduced_precision_memory=True,
        replay_safe_bandwidth_layout=True,
        adapter_only=True,
        inference_memory_bandwidth_receipt_hash="",
    )

    def with_hash_bw(obj, key):
        return replace(obj, **{key: imbr._hash_payload(imbr._base_payload(obj.__dict__, key))})

    bw = replace(
        bw,
        memory_bandwidth_identity=with_hash_bw(bw.memory_bandwidth_identity, "memory_bandwidth_identity_hash"),
        memory_transfer_declaration=with_hash_bw(bw.memory_transfer_declaration, "memory_transfer_declaration_hash"),
        cache_bandwidth_declaration=with_hash_bw(bw.cache_bandwidth_declaration, "cache_bandwidth_declaration_hash"),
        precision_bandwidth_declaration=with_hash_bw(
            bw.precision_bandwidth_declaration,
            "precision_bandwidth_declaration_hash",
        ),
        hardware_bandwidth_boundary=with_hash_bw(bw.hardware_bandwidth_boundary, "hardware_bandwidth_boundary_hash"),
        benchmark_bandwidth_boundary=with_hash_bw(bw.benchmark_bandwidth_boundary, "benchmark_bandwidth_boundary_hash"),
    )
    bw = replace(
        bw,
        inference_memory_bandwidth_receipt_hash=imbr._hash_payload(
            imbr._base_payload(bw.__dict__, "inference_memory_bandwidth_receipt_hash")
        ),
    )
    assert imbr.validate_inference_memory_bandwidth_receipt(bw, manifest, compression, token_receipt, byte_receipt) == bw
    return compression, token_receipt, byte_receipt, bw


def _build_identity(reason: str = "declared"):
    return kcp.build_kv_cache_identity("kv", "1", reason)


def _build_storage(mode: str = "STATIC_CACHE", size: int = 4096, reason: str = "declared"):
    return kcp.build_cache_storage_declaration(mode, size, reason)


def _build_sharing(mode: str = "NO_CACHE_SHARING", reason: str = "declared"):
    return kcp.build_cache_sharing_declaration(mode, reason)


def _build_precision(mode: str = "FP32_CACHE", reason: str = "declared"):
    return kcp.build_cache_precision_declaration(mode, reason)


def _build_replay(mode: str = "REPLAY_SAFE_CACHE", reason: str = "declared"):
    return kcp.build_cache_replay_boundary(mode, reason)


def _build_eviction(mode: str = "NO_EVICTION", reason: str = "declared"):
    return kcp.build_cache_eviction_boundary(mode, reason)


def _receipt(**kwargs):
    m = _manifest()
    c, t, b, bw = _compression_bundle(m)
    r = kcp.build_kv_cache_policy_receipt(
        m,
        bw,
        _build_identity(kwargs.get("identity_reason", "declared")),
        _build_storage(kwargs.get("storage", "STATIC_CACHE_STORAGE"), kwargs.get("size", 4096), kwargs.get("storage_reason", "declared")),
        _build_sharing(kwargs.get("sharing", "NO_CACHE_SHARING"), kwargs.get("sharing_reason", "declared")),
        _build_precision(kwargs.get("precision", "FP32_CACHE"), kwargs.get("precision_reason", "declared")),
        _build_replay(kwargs.get("replay", "REPLAY_SAFE_CACHE"), kwargs.get("replay_reason", "declared")),
        _build_eviction(kwargs.get("eviction", "NO_EVICTION"), kwargs.get("eviction_reason", "declared")),
        loop_count=kwargs.get("loop_count", 1),
        adapter_only=kwargs.get("adapter_only", True),
    )
    return r, m, bw, c, t, b


def test_hash_and_canonical_json_stability_and_idempotent_rebuild():
    r1, m, bw, c, t, b = _receipt()
    r2, *_ = _receipt()
    assert r1.kv_cache_policy_receipt_hash == r2.kv_cache_policy_receipt_hash
    assert kcp._canonical_json(r1.__dict__) == kcp._canonical_json(r2.__dict__)
    assert kcp.validate_kv_cache_policy_receipt(r1, m, bw, c, t, b) == r1


def test_reduced_precision_recomputed_not_trusted():
    r, m, bw, c, t, b = _receipt(precision="DECLARED_CUSTOM_CACHE_PRECISION")
    assert r.reduced_precision_cache is True
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, reduced_precision_cache=False), m, bw, c, t, b)


def test_replay_safe_recomputed_not_trusted():
    r, m, bw, c, t, b = _receipt()
    assert r.replay_safe_cache is True
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, replay_safe_cache=False), m, bw, c, t, b)


@pytest.mark.parametrize("mode", ["SHARED_CACHE_STORAGE", "DYNAMIC_CACHE_STORAGE", "PAGED_CACHE_STORAGE", "DECLARED_CUSTOM_CACHE_STORAGE"])
def test_replay_safe_false_for_storage_boundaries(mode):
    r, m, bw, c, t, b = _receipt(storage=mode)
    assert r.replay_safe_cache is False
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, replay_safe_cache=True), m, bw, c, t, b)


@pytest.mark.parametrize("mode", ["DECLARED_CONTEXTUAL_CACHE", "NON_REPLAY_SAFE_CACHE", "RESEARCH_CACHE_ONLY", "DECLARED_CUSTOM_CACHE_REPLAY"])
def test_replay_safe_false_for_replay_boundaries(mode):
    r, m, bw, c, t, b = _receipt(replay=mode)
    assert r.replay_safe_cache is False
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, replay_safe_cache=True), m, bw, c, t, b)


@pytest.mark.parametrize("mode", ["FIFO_EVICTION", "LRU_EVICTION", "DECLARED_CONTEXTUAL_EVICTION", "DECLARED_CUSTOM_EVICTION"])
def test_replay_safe_false_for_eviction_boundaries(mode):
    r, m, bw, c, t, b = _receipt(eviction=mode)
    assert r.replay_safe_cache is False
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, replay_safe_cache=True), m, bw, c, t, b)


def test_bool_alias_and_adapter_only_enforcement():
    r, m, bw, c, t, b = _receipt()
    for bad in (1,):
        with pytest.raises(ValueError):
            kcp.validate_kv_cache_policy_receipt(replace(r, adapter_only=bad), m, bw, c, t, b)
    for bad in (0, 1):
        with pytest.raises(ValueError):
            kcp.validate_kv_cache_policy_receipt(replace(r, reduced_precision_cache=bad), m, bw, c, t, b)
        with pytest.raises(ValueError):
            kcp.validate_kv_cache_policy_receipt(replace(r, replay_safe_cache=bad), m, bw, c, t, b)
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, adapter_only=False), m, bw, c, t, b)


def test_hash_enum_and_exact_type_rejection():
    r, m, bw, c, t, b = _receipt()
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, kv_cache_policy_receipt_hash="x"), m, bw, c, t, b)
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, storage_declaration=_build_storage("BAD")), m, bw, c, t, b)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        kcp.validate_kv_cache_policy_receipt({}, m, bw, c, t, b)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        kcp.validate_kv_cache_policy_receipt(replace(r, storage_declaration={}), m, bw, c, t, b)


def test_child_before_aggregate_hash_validation_ordering():
    r, m, bw, c, t, b = _receipt()
    bad_child = object.__new__(kcp.CachePrecisionDeclaration)
    object.__setattr__(bad_child, "precision_mode", r.precision_declaration.precision_mode)
    object.__setattr__(bad_child, "precision_reason", r.precision_declaration.precision_reason)
    object.__setattr__(bad_child, "cache_precision_declaration_hash", "a" * 63)
    bad = replace(r, precision_declaration=bad_child, kv_cache_policy_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="cache_precision_declaration_hash"):
        kcp.validate_kv_cache_policy_receipt(bad, m, bw, c, t, b)


@pytest.mark.parametrize(
    "text",
    [
        "runtime inference",
        "autonomous evaluation",
        "hidden replay equivalence",
        "hidden semantic equivalence",
        "hidden cache mutation",
        "hidden cache-sharing superiority",
        "hidden_cache_mutation",
        "hidden-cache-sharing-superiority",
    ],
)
def test_forbidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _build_replay(reason=text)


def test_upstream_invalid_manifest_rejected():
    r, m, bw, c, t, b = _receipt()
    bad_manifest = replace(m, adapter_only=False)
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(r, bad_manifest, bw, c, t, b)


def test_upstream_invalid_bandwidth_receipt_rejected():
    r, m, bw, c, t, b = _receipt()
    bad_bw = replace(bw, replay_safe_bandwidth_layout=False)
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(r, m, bad_bw, c, t, b)


def test_max_cache_size_integer_bounds_and_bool_rejection():
    with pytest.raises(ValueError):
        _build_storage(size=True)
    with pytest.raises(ValueError):
        _build_storage(size=0)
    with pytest.raises(ValueError):
        _build_storage(size=kcp._MAX_CACHE_SIZE_BYTES + 1)


def test_fail_fast_child_validation_and_immutability_and_import_safety():
    with pytest.raises(ValueError):
        kcp.KVCacheIdentity("", "1", "declared", "0" * 64)
    r, *_ = _receipt()
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    source = (Path(__file__).parent.parent / "src/qec/analysis/kv_cache_policy_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers",
        "torch",
        "tensorflow",
        "cuda",
        "cupy",
        "triton",
        "mlx",
        "vllm",
        "llama_cpp",
        "requests",
        "urllib",
        "subprocess",
        "eval(",
        "exec(",
        "os.system",
    ):
        assert token not in source


def test_loop_count_required_and_validated():
    r, m, bw, c, t, b = _receipt(loop_count=2)
    assert r.loop_count == 2
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(replace(r, loop_count=0), m, bw, c, t, b)


def test_rejects_upstream_no_cache_contradictions_and_non_replay_safe_bandwidth():
    r, m, bw, c, t, b = _receipt()
    bad_manifest = replace(m, kv_cache_declaration=ibm.build_kv_cache_declaration("STATIC_CACHE", "declared"))
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(r, bad_manifest, bw, c, t, b)

    bad_bw = replace(
        bw,
        replay_safe_bandwidth_layout=False,
        inference_memory_bandwidth_receipt_hash=imbr._hash_payload(
            imbr._base_payload(replace(bw, replay_safe_bandwidth_layout=False).__dict__, "inference_memory_bandwidth_receipt_hash")
        ),
    )
    with pytest.raises(ValueError):
        kcp.validate_kv_cache_policy_receipt(r, m, bad_bw, c, t, b)
