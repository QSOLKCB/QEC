from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import inference_backend_manifest as ibm


def _build_manifest(**kwargs):
    identity = kwargs.get("identity") or ibm.build_inference_backend_identity("backend", "1.0", "TRANSFORMER")
    tok = kwargs.get("tok") or ibm.build_tokenization_policy_declaration("BPE", "tok", "declared policy only")
    quant = kwargs.get("quant") or ibm.build_quantization_declaration("NO_QUANTIZATION", "none")
    prec = kwargs.get("prec") or ibm.build_precision_format_declaration("FP32", "full precision")
    hw = kwargs.get("hw") or ibm.build_hardware_kernel_boundary("CPU_ONLY", "adapter-only boundary", True)
    bench = kwargs.get("bench") or ibm.build_benchmark_corpus_declaration("NO_BENCHMARK_DECLARED", "none", "context only")
    kv = kwargs.get("kv") or ibm.build_kv_cache_declaration("NO_CACHE", "none")
    return ibm.build_inference_backend_manifest(identity, tok, quant, prec, hw, bench, kv)


def test_hash_and_canonical_json_stability_and_idempotent_rebuild():
    m1 = _build_manifest()
    m2 = _build_manifest()
    assert m1.inference_backend_manifest_hash == m2.inference_backend_manifest_hash
    assert ibm._canonical_json(m1.__dict__) == ibm._canonical_json(m2.__dict__)
    assert ibm.validate_inference_backend_manifest(m1) == m1


def test_reduced_precision_and_benchmark_recomputed_and_enforced():
    m = _build_manifest(
        quant=ibm.build_quantization_declaration("INT8", "declared"),
        prec=ibm.build_precision_format_declaration("MIXED_PRECISION", "declared"),
        bench=ibm.build_benchmark_corpus_declaration("TOKEN_SPEED_CONTEXT_ONLY", "ctx", "context only"),
    )
    assert m.reduced_precision_declared is True
    assert m.benchmark_required is True
    tampered = replace(m, reduced_precision_declared=False)
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(tampered)
    tampered_b = replace(m, benchmark_required=False)
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(tampered_b)


@pytest.mark.parametrize(
    "builder,args",
    [
        (ibm.build_inference_backend_identity, ("a", "1", "BAD")),
        (ibm.build_tokenization_policy_declaration, ("BAD", "n", "r")),
        (ibm.build_quantization_declaration, ("BAD", "r")),
        (ibm.build_precision_format_declaration, ("BAD", "r")),
        (ibm.build_hardware_kernel_boundary, ("BAD", "r", True)),
        (ibm.build_benchmark_corpus_declaration, ("BAD", "c", "r")),
        (ibm.build_kv_cache_declaration, ("BAD", "r")),
    ],
)
def test_invalid_enums_rejected(builder, args):
    with pytest.raises(ValueError):
        builder(*args)


def test_malformed_hash_rejected_and_child_validated_first():
    m = _build_manifest()
    bad_child = replace(m.backend_identity, inference_backend_identity_hash="a" * 63)
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(replace(m, backend_identity=bad_child))


def test_immutable_payload_validation():
    m = _build_manifest()
    with pytest.raises(FrozenInstanceError):
        m.adapter_only = False


def test_hidden_quantization_and_quantized_model_require_declaration():
    ident = ibm.build_inference_backend_identity("q", "1", "QUANTIZED_MODEL")
    m = _build_manifest(identity=ident)
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(m)


def test_adapter_only_and_hardware_adapter_enforcement():
    m = _build_manifest()
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(replace(m, adapter_only=False))
    bad_hw = replace(m.hardware_boundary, adapter_only=False)
    with pytest.raises(ValueError):
        ibm.validate_inference_backend_manifest(replace(m, hardware_boundary=bad_hw))


@pytest.mark.parametrize(
    "text",
    [
        "tokenization preserves truth",
        "benchmark proves",
        "autonomous evaluation",
        "hardware superiority established",
        "semantic equivalence guaranteed",
    ],
)
def test_forbidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        ibm.build_tokenization_policy_declaration("BPE", "tok", text)


def test_no_runtime_inference_imports_and_decoder_boundary_and_hashseed_stability():
    source = (Path(__file__).parent.parent / "src/qec/analysis/inference_backend_manifest.py").read_text(encoding="utf-8")
    for token in ("transformers", "torch", "tensorflow", "vllm", "openai", "anthropic", "subprocess", "os.system"):
        assert token not in source
    assert "src/qec/decoder/" not in "src/qec/analysis/inference_backend_manifest.py"
    assert _build_manifest().inference_backend_manifest_hash == _build_manifest().inference_backend_manifest_hash


def test_hidden_tokenization_drift_semantics_rejected():
    with pytest.raises(ValueError):
        ibm.build_tokenization_policy_declaration("DECLARED_CUSTOM_TOKENIZER", "tok", "semantic equivalence guaranteed")


def test_benchmark_context_only_semantics_allowed():
    b = ibm.build_benchmark_corpus_declaration("SYNTHETIC_BENCHMARK_ONLY", "synthetic", "context only; no authority")
    assert b.benchmark_mode == "SYNTHETIC_BENCHMARK_ONLY"
