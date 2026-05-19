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


def _token(m, b):
    return tpr.build_tokenization_policy_receipt(
        m,
        b,
        tpr.build_tokenizer_identity("tok", "1", "BYTE_LEVEL_TOKENIZER"),
        tpr.build_vocabulary_policy_declaration("BYTE_NATIVE_VOCABULARY", 256, "boundary only"),
        tpr.build_merge_rule_declaration("BYTE_NATIVE_SEGMENTATION", "replay boundary only"),
        tpr.build_token_boundary_declaration("BYTE_COMPATIBLE_BOUNDARY", "byte compatible boundary"),
        tpr.build_tokenizer_equivalence_declaration("NO_EQUIVALENCE_CLAIM", "no replay identity claim"),
        tpr.build_tokenizer_drift_boundary("DRIFT_PROHIBITED", "drift prohibited"),
        True,
    )


def _with_hash(obj, key):
    return replace(obj, **{key: pg._hash_payload(pg._base_payload(obj.__dict__, key))})


def _parts(*, reduction_mode="FIXED_PARAMETER_REDUCTION", quant_mode="DECLARED_CUSTOM_QUANTIZATION", precision_mode="DECLARED_CUSTOM_PRECISION_REDUCTION"):
    ci = _with_hash(pg.CompressionIdentity("c", "1", "PARAMETER_GOLF", ""), "compression_identity_hash")
    pr = _with_hash(pg.ParameterReductionDeclaration(reduction_mode, 100, 80, "declared", ""), "parameter_reduction_declaration_hash")
    qd = _with_hash(pg.QuantizationCompressionDeclaration(quant_mode, "declared", ""), "quantization_compression_declaration_hash")
    pd = _with_hash(pg.PrecisionReductionDeclaration(precision_mode, "declared", ""), "precision_reduction_declaration_hash")
    eq = _with_hash(pg.CompressionEquivalenceBoundary("NO_EQUIVALENCE_CLAIM", "declared", ""), "compression_equivalence_boundary_hash")
    bm = _with_hash(pg.CompressionBenchmarkBoundary("NO_BENCHMARK_DECLARED", "declared", ""), "compression_benchmark_boundary_hash")
    return ci, pr, qd, pd, eq, bm


def _receipt(**kwargs):
    m = _manifest()
    b = _byte(m)
    t = _token(m, b)
    ci, pr, qd, pd, eq, bm = _parts(
        reduction_mode=kwargs.get("reduction_mode", "FIXED_PARAMETER_REDUCTION"),
        quant_mode=kwargs.get("quant_mode", "DECLARED_CUSTOM_QUANTIZATION"),
        precision_mode=kwargs.get("precision_mode", "DECLARED_CUSTOM_PRECISION_REDUCTION"),
    )
    return pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, qd, pd, eq, bm, kwargs.get("adapter_only", True)), m, t, b


def test_hash_stability_and_canonical_json_stability_and_idempotent_rebuild():
    r1, *_ = _receipt()
    r2, *_ = _receipt()
    assert r1.parameter_golf_compression_receipt_hash == r2.parameter_golf_compression_receipt_hash
    assert pg._canonical_json(r1.__dict__) == pg._canonical_json(r2.__dict__)


def test_recomputed_booleans_enforced_and_bool_alias_rejected():
    r, m, t, b = _receipt()
    pg.validate_parameter_golf_compression_receipt(r, m, t, b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, replay_safe_compression=1), m, t, b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, reduced_precision_declared=1), m, t, b)


def test_parameter_count_bool_alias_and_bounds_rejected_in_builder_and_validator():
    m = _manifest()
    b = _byte(m)
    t = _token(m, b)
    ci, _, qd, pd, eq, bm = _parts()
    bad_pr = _with_hash(pg.ParameterReductionDeclaration("FIXED_PARAMETER_REDUCTION", True, 80, "declared", ""), "parameter_reduction_declaration_hash")
    with pytest.raises(ValueError):
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, bad_pr, qd, pd, eq, bm, True)

    r, m, t, b = _receipt()
    bad = replace(r.parameter_reduction_declaration, reduced_parameter_count=True)
    bad = _with_hash(bad, "parameter_reduction_declaration_hash")
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, parameter_reduction_declaration=bad), m, t, b)


def test_custom_modes_force_reduced_precision_true():
    r, m, t, b = _receipt(quant_mode="DECLARED_CUSTOM_QUANTIZATION", precision_mode="FP32_TO_FP16")
    assert r.reduced_precision_declared is True
    pg.validate_parameter_golf_compression_receipt(r, m, t, b)
    r2, m2, t2, b2 = _receipt(quant_mode="FP16", precision_mode="DECLARED_CUSTOM_PRECISION_REDUCTION")
    assert r2.reduced_precision_declared is True
    pg.validate_parameter_golf_compression_receipt(r2, m2, t2, b2)


@pytest.mark.parametrize("mode", ["DECLARED_DYNAMIC_REDUCTION", "RESEARCH_REDUCTION_ONLY", "DECLARED_CUSTOM_REDUCTION"])
def test_dynamic_research_custom_reduction_not_replay_safe(mode):
    r, m, t, b = _receipt(reduction_mode=mode)
    assert r.replay_safe_compression is False
    pg.validate_parameter_golf_compression_receipt(r, m, t, b)


@pytest.mark.parametrize(
    "text",
    [
        "hidden semantic-equivalence",
        "hidden semantic equivalence",
        "hidden_semantic_equivalence",
        "hidden replay-equivalence",
        "hidden replay equivalence",
        "hidden_replay_equivalence",
        "hidden benchmark-superiority",
        "hidden benchmark superiority",
        "hidden_benchmark_superiority",
        "hidden precision-mutation",
        "hidden precision mutation",
        "hidden_precision_mutation",
    ],
)
def test_forbidden_hidden_variants_rejected_in_identity_and_reason_fields(text):
    with pytest.raises(ValueError):
        ci = _with_hash(pg.CompressionIdentity(text, "1", "PARAMETER_GOLF", ""), "compression_identity_hash")
        r, m, t, b = _receipt()
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, r.parameter_reduction_declaration, r.quantization_compression_declaration, r.precision_reduction_declaration, r.compression_equivalence_boundary, r.compression_benchmark_boundary, True)
    with pytest.raises(ValueError):
        bad_pr = _with_hash(pg.ParameterReductionDeclaration("FIXED_PARAMETER_REDUCTION", 100, 80, text, ""), "parameter_reduction_declaration_hash")
        r, m, t, b = _receipt()
        pg.build_parameter_golf_compression_receipt(m, t, b, r.compression_identity, bad_pr, r.quantization_compression_declaration, r.precision_reduction_declaration, r.compression_equivalence_boundary, r.compression_benchmark_boundary, True)


def test_forbidden_semantic_text_rejected_in_name_version_and_all_reason_fields():
    m = _manifest()
    b = _byte(m)
    t = _token(m, b)
    ci, pr, qd, pd, eq, bm = _parts()
    with pytest.raises(ValueError):
        bad_ci = _with_hash(replace(ci, compression_name="runtime inference"), "compression_identity_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, bad_ci, pr, qd, pd, eq, bm, True)
    with pytest.raises(ValueError):
        bad_ci = _with_hash(replace(ci, compression_version="autonomous evaluation"), "compression_identity_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, bad_ci, pr, qd, pd, eq, bm, True)
    with pytest.raises(ValueError):
        bad = _with_hash(replace(pr, reduction_reason="hidden replay equivalence"), "parameter_reduction_declaration_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, bad, qd, pd, eq, bm, True)
    with pytest.raises(ValueError):
        bad = _with_hash(replace(qd, quantization_reason="hidden replay equivalence"), "quantization_compression_declaration_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, bad, pd, eq, bm, True)
    with pytest.raises(ValueError):
        bad = _with_hash(replace(pd, precision_reason="hidden replay equivalence"), "precision_reduction_declaration_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, qd, bad, eq, bm, True)
    with pytest.raises(ValueError):
        bad = _with_hash(replace(eq, equivalence_reason="hidden replay equivalence"), "compression_equivalence_boundary_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, qd, pd, bad, bm, True)
    with pytest.raises(ValueError):
        bad = _with_hash(replace(bm, benchmark_reason="hidden replay equivalence"), "compression_benchmark_boundary_hash")
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, qd, pd, eq, bad, True)


def test_builder_rejects_adapter_false_and_invalid_child_hashes_and_child_before_aggregate():
    m = _manifest()
    b = _byte(m)
    t = _token(m, b)
    ci, pr, qd, pd, eq, bm = _parts()
    with pytest.raises(ValueError):
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, pr, qd, pd, eq, bm, False)
    bad_pr = replace(pr, parameter_reduction_declaration_hash="x")
    with pytest.raises(ValueError):
        pg.build_parameter_golf_compression_receipt(m, t, b, ci, bad_pr, qd, pd, eq, bm, True)


def test_validate_catches_tampered_recomputed_fields_and_immutability_and_runtime_import_scan():
    r, m, t, b = _receipt()
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, reduced_precision_declared=False), m, t, b)
    with pytest.raises(ValueError):
        pg.validate_parameter_golf_compression_receipt(replace(r, replay_safe_compression=False), m, t, b)
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    source = (Path(__file__).parent.parent / "src/qec/analysis/parameter_golf_compression_receipts.py").read_text(encoding="utf-8")
    for token in ("transformers", "torch", "tensorflow", "bitsandbytes", "auto_gptq", "awq", "openai", "anthropic", "vllm", "llama_cpp", "requests", "urllib", "subprocess", "eval(", "exec(", "os.system"):
        assert token not in source
