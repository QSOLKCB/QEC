from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import inference_backend_manifest as ibm
from qec.analysis import parameter_golf_compression_receipts as pg
from qec.analysis import inference_memory_bandwidth_receipts as imbr
from qec.analysis import byte_level_model_boundary_receipts as bl
from qec.analysis import tokenization_policy_receipts as tpr


def _manifest():
    return ibm.build_inference_backend_manifest(
        ibm.build_inference_backend_identity("backend", "1.0", "BYTE_LEVEL_MODEL"),
        ibm.build_tokenization_policy_declaration("BYTE_LEVEL", "tok", "declared"),
        ibm.build_quantization_declaration("NO_QUANTIZATION", "none"),
        ibm.build_precision_format_declaration("FP32", "full"),
        ibm.build_hardware_kernel_boundary("ADAPTER_ONLY_HARDWARE", "boundary", True),
        ibm.build_benchmark_corpus_declaration("NO_BENCHMARK_DECLARED", "none", "context"),
        ibm.build_kv_cache_declaration("NO_CACHE", "none"),
    )


def _compression(manifest):
    def with_hash(obj, key):
        return replace(obj, **{key: pg._hash_payload(pg._base_payload(obj.__dict__, key))})

    ci = with_hash(pg.CompressionIdentity("golf", "1", "PARAMETER_GOLF", ""), "compression_identity_hash")
    pr = with_hash(pg.ParameterReductionDeclaration("FIXED_PARAMETER_REDUCTION", 100, 50, "declared", ""), "parameter_reduction_declaration_hash")
    qd = with_hash(pg.QuantizationCompressionDeclaration("FP16", "declared", ""), "quantization_compression_declaration_hash")
    pd = with_hash(pg.PrecisionReductionDeclaration("FP32_TO_FP16", "declared", ""), "precision_reduction_declaration_hash")
    eq = with_hash(pg.CompressionEquivalenceBoundary("NO_EQUIVALENCE_CLAIM", "declared", ""), "compression_equivalence_boundary_hash")
    bm = with_hash(pg.CompressionBenchmarkBoundary("NO_BENCHMARK_DECLARED", "declared", ""), "compression_benchmark_boundary_hash")
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
    return pg.build_parameter_golf_compression_receipt(manifest, token_receipt, byte_receipt, ci, pr, qd, pd, eq, bm, True), token_receipt, byte_receipt


def _receipt(**kwargs):
    m = _manifest()
    c, t, b = _compression(m)
    receipt = imbr.InferenceMemoryBandwidthReceipt(
        schema_version="INFERENCE_MEMORY_BANDWIDTH_RECEIPT_V1",
        inference_backend_manifest_hash=m.inference_backend_manifest_hash,
        parameter_golf_compression_receipt_hash=c.parameter_golf_compression_receipt_hash,
        memory_bandwidth_identity=imbr.MemoryBandwidthIdentity("profile", "1", ""),
        memory_transfer_declaration=imbr.MemoryTransferDeclaration(kwargs.get("memory", "STATIC_MEMORY_LAYOUT"), "declared", ""),
        cache_bandwidth_declaration=imbr.CacheBandwidthDeclaration(kwargs.get("cache", "NO_CACHE"), "declared", ""),
        precision_bandwidth_declaration=imbr.PrecisionBandwidthDeclaration(kwargs.get("precision", "FP32_MEMORY"), "declared", ""),
        hardware_bandwidth_boundary=imbr.HardwareBandwidthBoundary(kwargs.get("hw", "ADAPTER_ONLY_HARDWARE"), "declared", ""),
        benchmark_bandwidth_boundary=imbr.BenchmarkBandwidthBoundary(kwargs.get("bench", "NO_BENCHMARK_DECLARED"), "declared", ""),
        reduced_precision_memory=False,
        replay_safe_bandwidth_layout=False,
        adapter_only=kwargs.get("adapter_only", True),
        inference_memory_bandwidth_receipt_hash="",
    )

    def with_hash(obj, key):
        return replace(obj, **{key: imbr._hash_payload(imbr._base_payload(obj.__dict__, key))})

    r = replace(
        receipt,
        memory_bandwidth_identity=with_hash(receipt.memory_bandwidth_identity, "memory_bandwidth_identity_hash"),
        memory_transfer_declaration=with_hash(receipt.memory_transfer_declaration, "memory_transfer_declaration_hash"),
        cache_bandwidth_declaration=with_hash(receipt.cache_bandwidth_declaration, "cache_bandwidth_declaration_hash"),
        precision_bandwidth_declaration=with_hash(receipt.precision_bandwidth_declaration, "precision_bandwidth_declaration_hash"),
        hardware_bandwidth_boundary=with_hash(receipt.hardware_bandwidth_boundary, "hardware_bandwidth_boundary_hash"),
        benchmark_bandwidth_boundary=with_hash(receipt.benchmark_bandwidth_boundary, "benchmark_bandwidth_boundary_hash"),
    )
    reduced = imbr._recompute_reduced_precision_memory(r.precision_bandwidth_declaration.precision_bandwidth_mode, r.cache_bandwidth_declaration.cache_mode)
    replay = imbr._recompute_replay_safe_bandwidth_layout(
        r.memory_transfer_declaration.memory_layout_mode,
        r.benchmark_bandwidth_boundary.benchmark_mode,
        r.hardware_bandwidth_boundary.hardware_boundary_mode,
        r.adapter_only,
        True,
        True,
    )
    r = replace(r, reduced_precision_memory=reduced, replay_safe_bandwidth_layout=replay)
    r = replace(r, inference_memory_bandwidth_receipt_hash=imbr._hash_payload(imbr._base_payload(r.__dict__, "inference_memory_bandwidth_receipt_hash")))
    return r, m, c, t, b


def test_hash_stability_canonical_json_and_idempotent_rebuild_and_upstream_validation():
    r1, m1, c1, t1, b1 = _receipt()
    r2, _, _, _, _ = _receipt()
    assert r1.inference_memory_bandwidth_receipt_hash == r2.inference_memory_bandwidth_receipt_hash
    assert imbr._canonical_json(r1.__dict__) == imbr._canonical_json(r2.__dict__)
    assert imbr.validate_inference_memory_bandwidth_receipt(r1, m1, c1, t1, b1) == r1


def test_recomputed_fields_and_custom_precision_and_dynamic_layout_replay_rejected():
    r, m, c, t, b = _receipt(precision="DECLARED_CUSTOM_PRECISION_MEMORY")
    assert r.reduced_precision_memory is True
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, reduced_precision_memory=False), m, c, t, b)
    rd, md, cd, td, bd = _receipt(memory="DYNAMIC_MEMORY_LAYOUT")
    assert rd.replay_safe_bandwidth_layout is False
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(rd, replay_safe_bandwidth_layout=True), md, cd, td, bd)


def test_malformed_hash_enum_bool_alias_adapter_only_and_hidden_semantics_rejected():
    r, m, c, t, b = _receipt()
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, inference_memory_bandwidth_receipt_hash="x"), m, c, t, b)
    bad = replace(r.memory_transfer_declaration, memory_layout_mode="BAD")
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, memory_transfer_declaration=bad), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, adapter_only=1), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, reduced_precision_memory=1), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, reduced_precision_memory=0), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, replay_safe_bandwidth_layout=1), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, replay_safe_bandwidth_layout=0), m, c, t, b)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, adapter_only=False), m, c, t, b)
    hidden = replace(r.benchmark_bandwidth_boundary, benchmark_reason="hidden replay equivalence")
    hidden = replace(hidden, benchmark_bandwidth_boundary_hash=imbr._hash_payload(imbr._base_payload(hidden.__dict__, "benchmark_bandwidth_boundary_hash")))
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, benchmark_bandwidth_boundary=hidden), m, c, t, b)


@pytest.mark.parametrize(
    "text",
    [
        "runtime_inference",
        "autonomous-evaluation",
        "hidden_hardware_superiority",
        "hidden-semantic-equivalence",
        "hidden_precision_mutation",
        "hidden replay equivalence",
    ],
)
def test_separator_variants_forbidden_semantics_rejected(text):
    r, m, c, t, b = _receipt()
    bad = replace(r.benchmark_bandwidth_boundary, benchmark_reason=text)
    bad = replace(bad, benchmark_bandwidth_boundary_hash=imbr._hash_payload(imbr._base_payload(bad.__dict__, "benchmark_bandwidth_boundary_hash")))
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, benchmark_bandwidth_boundary=bad), m, c, t, b)


def test_type_enforcement_for_top_level_and_child_and_child_before_aggregate():
    r, m, c, t, b = _receipt()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        imbr.validate_inference_memory_bandwidth_receipt({}, m, c, t, b)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, memory_transfer_declaration={}), m, c, t, b)
    bad_child = replace(r.precision_bandwidth_declaration, precision_bandwidth_declaration_hash="a" * 63)
    bad = replace(r, precision_bandwidth_declaration=bad_child, inference_memory_bandwidth_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="precision_bandwidth_declaration_hash"):
        imbr.validate_inference_memory_bandwidth_receipt(bad, m, c, t, b)


def test_custom_layout_and_cache_implied_recompute_enforced():
    rc, mc, cc, tc, bc = _receipt(memory="DECLARED_CUSTOM_MEMORY_LAYOUT")
    assert rc.replay_safe_bandwidth_layout is False
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(rc, replay_safe_bandwidth_layout=True), mc, cc, tc, bc)
    rk, mk, ck, tk, bk = _receipt(cache="KV_CACHE")
    assert rk.reduced_precision_memory is True
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(rk, reduced_precision_memory=False), mk, ck, tk, bk)
    rd, md, cd, td, bd = _receipt(cache="DECLARED_CUSTOM_CACHE")
    assert rd.reduced_precision_memory is True
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(rd, reduced_precision_memory=False), md, cd, td, bd)


def test_child_before_aggregate_immutable_and_no_runtime_imports_hashseed_replay_stability():
    r, m, c, t, b = _receipt()
    bad_child = replace(r.precision_bandwidth_declaration, precision_bandwidth_declaration_hash="a" * 63)
    with pytest.raises(ValueError):
        imbr.validate_inference_memory_bandwidth_receipt(replace(r, precision_bandwidth_declaration=bad_child), m, c, t, b)
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    source = (Path(__file__).parent.parent / "src/qec/analysis/inference_memory_bandwidth_receipts.py").read_text(encoding="utf-8")
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
        "bitsandbytes",
        "requests",
        "urllib",
        "subprocess",
        "eval(",
        "exec(",
        "os.system",
    ):
        assert token not in source
    assert _receipt()[0].inference_memory_bandwidth_receipt_hash == _receipt()[0].inference_memory_bandwidth_receipt_hash
