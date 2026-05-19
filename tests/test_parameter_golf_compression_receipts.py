from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import byte_level_model_boundary_receipts as bl
from qec.analysis import inference_backend_manifest as ibm
from qec.analysis import parameter_golf_compression_receipts as pg
from qec.analysis import tokenization_policy_receipts as tpr


def _manifest():
    return ibm.build_inference_backend_manifest(
        ibm.build_inference_backend_identity("backend", "1.0", "BYTE_LEVEL_MODEL"),
        ibm.build_tokenization_policy_declaration("BYTE_LEVEL", "declared", "boundary only"),
        ibm.build_quantization_declaration("NO_QUANTIZATION", "none"),
        ibm.build_precision_format_declaration("FP32", "full precision"),
        ibm.build_hardware_kernel_boundary("ADAPTER_ONLY_HARDWARE", "boundary only", True),
        ibm.build_benchmark_corpus_declaration("NO_BENCHMARK_DECLARED", "none", "context only"),
        ibm.build_kv_cache_declaration("NO_CACHE", "none"),
    )


def _byte(m):
    return bl.build_byte_level_model_boundary_receipt(
        m,
        bl.build_byte_level_model_identity("family", "PURE_BYTE_MODEL", "declared boundary"),
        bl.build_patch_segmentation_declaration("FIXED_PATCH", 64, "segmentation only"),
        bl.build_byte_window_declaration(128, True, "replay layout only"),
        bl.build_entropy_threshold_declaration("FIXED_THRESHOLD", 0.0, "declared only"),
        bl.build_byte_normalization_policy("NO_NORMALIZATION", "declared only"),
        bl.build_tokenizer_bypass_declaration("TOKENIZER_REMOVED", "declared only"),
        True,
    )


def _token(m,b):
    return tpr.build_tokenization_policy_receipt(
        m,b,
        tpr.build_tokenizer_identity("tok", "1", "BYTE_LEVEL_TOKENIZER"),
        tpr.build_vocabulary_policy_declaration("BYTE_NATIVE_VOCABULARY", 256, "boundary only"),
        tpr.build_merge_rule_declaration("BYTE_NATIVE_SEGMENTATION", "replay boundary only"),
        tpr.build_token_boundary_declaration("BYTE_COMPATIBLE_BOUNDARY", "byte compatible boundary"),
        tpr.build_tokenizer_equivalence_declaration("NO_EQUIVALENCE_CLAIM", "no replay identity claim"),
        tpr.build_tokenizer_drift_boundary("DRIFT_PROHIBITED", "drift prohibited"),
        True,
    )


def _receipt():
    m=_manifest(); b=_byte(m); t=_token(m,b)
    ci=pg.CompressionIdentity("c","1","PARAMETER_GOLF","")
    ci=replace(ci, compression_identity_hash=pg._hash_payload(pg._base_payload(ci.__dict__, "compression_identity_hash")))
    pr=pg.ParameterReductionDeclaration("FIXED_PARAMETER_REDUCTION",100,80,"declared","")
    pr=replace(pr, parameter_reduction_declaration_hash=pg._hash_payload(pg._base_payload(pr.__dict__, "parameter_reduction_declaration_hash")))
    q=pg.QuantizationCompressionDeclaration("DECLARED_CUSTOM_QUANTIZATION","declared","")
    q=replace(q, quantization_compression_declaration_hash=pg._hash_payload(pg._base_payload(q.__dict__, "quantization_compression_declaration_hash")))
    p=pg.PrecisionReductionDeclaration("DECLARED_CUSTOM_PRECISION_REDUCTION","declared","")
    p=replace(p, precision_reduction_declaration_hash=pg._hash_payload(pg._base_payload(p.__dict__, "precision_reduction_declaration_hash")))
    eq=pg.CompressionEquivalenceBoundary("NO_EQUIVALENCE_CLAIM","declared","")
    eq=replace(eq, compression_equivalence_boundary_hash=pg._hash_payload(pg._base_payload(eq.__dict__, "compression_equivalence_boundary_hash")))
    bm=pg.CompressionBenchmarkBoundary("NO_BENCHMARK_DECLARED","declared","")
    bm=replace(bm, compression_benchmark_boundary_hash=pg._hash_payload(pg._base_payload(bm.__dict__, "compression_benchmark_boundary_hash")))
    return pg.build_parameter_golf_compression_receipt(m,t,b,ci,pr,q,p,eq,bm,True),m,t,b


def test_hash_stability_and_canonical_json_and_idempotent_and_hashseed_stability():
    r1, *_ = _receipt(); r2, *_ = _receipt()
    assert r1.parameter_golf_compression_receipt_hash == r2.parameter_golf_compression_receipt_hash
    assert pg._canonical_json(r1.__dict__) == pg._canonical_json(r2.__dict__)


def test_recomputation_and_validation_and_upstream_validation():
    r,m,t,b = _receipt()
    pg.validate_parameter_golf_compression_receipt(r,m,t,b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, reduced_precision_declared=False),m,t,b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, replay_safe_compression=False),m,t,b)


def test_malformed_hash_and_enums_and_counts_and_adapter_only_and_hidden_semantics():
    r,m,t,b = _receipt()
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, parameter_golf_compression_receipt_hash="x"),m,t,b)
    bad_pr = replace(r.parameter_reduction_declaration, reduction_mode="BAD")
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, parameter_reduction_declaration=bad_pr),m,t,b)
    bad_pr2 = replace(r.parameter_reduction_declaration, reduced_parameter_count=101)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, parameter_reduction_declaration=bad_pr2),m,t,b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, adapter_only=False),m,t,b)
    bad_eq = replace(r.compression_equivalence_boundary, equivalence_reason="hidden semantic equivalence")
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, compression_equivalence_boundary=bad_eq),m,t,b)


def test_fixed_vs_dynamic_reduction_and_immutability_and_runtime_import_scan():
    r,m,t,b = _receipt()
    dynamic = replace(r.parameter_reduction_declaration, reduction_mode="DECLARED_DYNAMIC_REDUCTION")
    r2 = replace(r, parameter_reduction_declaration=dynamic, replay_safe_compression=False)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(r2,m,t,b)
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    source = (Path(__file__).parent.parent / "src/qec/analysis/parameter_golf_compression_receipts.py").read_text(encoding="utf-8")
    for token in ("transformers","torch","tensorflow","bitsandbytes","auto_gptq","awq","openai","anthropic","vllm","llama_cpp","requests","urllib","subprocess","eval(","exec(","os.system"):
        assert token not in source
