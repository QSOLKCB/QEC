from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis import decoder_rollback_receipts as rr
from qec.analysis.decoder_rollback_receipts import (
    DecoderRollbackError,
    DecoderRollbackErrorCode,
    build_decoder_rollback_audit_boundary,
    build_decoder_rollback_authority_boundary,
    build_decoder_rollback_execution_boundary,
    build_decoder_rollback_identity,
    build_decoder_rollback_plan,
    build_decoder_rollback_plan_step,
    build_decoder_rollback_receipt,
    build_decoder_rollback_restoration_policy,
    build_decoder_rollback_target,
    build_decoder_rollback_trigger,
    build_decoder_rollback_upstream_binding,
    build_decoder_rollback_verification_boundary,
    validate_decoder_rollback_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/qec/analysis/decoder_rollback_receipts.py"
TEST_MODULE = ROOT / "tests/test_decoder_rollback_receipts.py"
H = {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64, "d": "d" * 64, "e": "e" * 64, "f": "f" * 64, "0": "0" * 64, "1": "1" * 64, "2": "2" * 64, "3": "3" * 64, "4": "4" * 64, "5": "5" * 64}


def assert_code(exc: pytest.ExceptionInfo[DecoderRollbackError], code: DecoderRollbackErrorCode) -> None:
    assert exc.value.code is code
    assert code.value in str(exc.value)
    assert exc.value.detail



def make_upstream(**overrides):
    payload = {
        "upstream_canonical_decoder_baseline_receipt_hash": H["a"],
        "upstream_decoder_candidate_manifest_hash": H["b"],
        "upstream_decoder_replay_equivalence_receipt_hash": H["c"],
        "upstream_decoder_optimization_contract_hash": H["d"],
        "upstream_decoder_fast_path_equivalence_receipt_hash": H["e"],
        "upstream_decoder_implementation_boundary_receipt_hash": H["f"],
        "upstream_decoder_benchmark_ladder_receipt_hash": H["0"],
        "candidate_declaration_hash": H["1"],
        "fast_path_identity_hash": H["2"],
        "implementation_identity_hash": H["3"],
        "candidate_name": "declared adapter candidate",
        "candidate_version": "1",
    }
    payload.update(overrides)
    return build_decoder_rollback_upstream_binding(**payload)


def make_identity(**overrides):
    payload = {
        "rollback_id": "declared-rollback-receipt",
        "rollback_name": "Declared rollback receipt",
        "rollback_version": "1",
        "associated_candidate_declaration_hash": H["1"],
        "associated_fast_path_identity_hash": H["2"],
        "associated_implementation_identity_hash": H["3"],
        "associated_benchmark_ladder_receipt_hash": H["0"],
    }
    payload.update(overrides)
    return build_decoder_rollback_identity(**payload)


def make_plan(**overrides):
    t_replay = build_decoder_rollback_trigger(trigger_id="a-replay", trigger_kind="REPLAY_EQUIVALENCE_FAILURE", trigger_source_receipt_hash=H["c"], trigger_source_release="v166.2", trigger_severity="BLOCK_PROMOTION")
    t_fast = build_decoder_rollback_trigger(trigger_id="b-fast", trigger_kind="FAST_PATH_EQUIVALENCE_FAILURE", trigger_source_receipt_hash=H["e"], trigger_source_release="v166.4", trigger_severity="DISABLE_FAST_PATH")
    t_bench = build_decoder_rollback_trigger(trigger_id="c-benchmark", trigger_kind="BENCHMARK_LADDER_REGRESSION", trigger_source_receipt_hash=H["0"], trigger_source_release="v166.6", trigger_severity="AUDIT_REQUIRED")
    base = build_decoder_rollback_target(target_id="baseline", target_kind="CANONICAL_BASELINE_TARGET", target_hash=H["a"], target_role="RESTORE_BASELINE", pre_rollback_status="DECLARED_ACTIVE", post_rollback_status="BASELINE_RESTORED", disable_required=False, restore_required=True)
    cand = build_decoder_rollback_target(target_id="candidate", target_kind="CANDIDATE_DECLARATION_TARGET", target_hash=H["1"], target_role="DISABLE_CANDIDATE", pre_rollback_status="DECLARED_ADAPTER_ONLY", post_rollback_status="CANDIDATE_DISABLED", disable_required=True, restore_required=False)
    fast = build_decoder_rollback_target(target_id="fast-path", target_kind="FAST_PATH_IDENTITY_TARGET", target_hash=H["2"], target_role="DISABLE_FAST_PATH", pre_rollback_status="DECLARED_ADAPTER_ONLY", post_rollback_status="FAST_PATH_DISABLED", disable_required=True, restore_required=False)
    impl = build_decoder_rollback_target(target_id="implementation", target_kind="IMPLEMENTATION_BOUNDARY_TARGET", target_hash=H["3"], target_role="DISABLE_IMPLEMENTATION", pre_rollback_status="DECLARED_BOUNDARY_ONLY", post_rollback_status="IMPLEMENTATION_DISABLED", disable_required=True, restore_required=False)
    promo = build_decoder_rollback_target(target_id="promotion", target_kind="PROMOTION_GATE_TARGET", target_hash=H["4"], target_role="BLOCK_PROMOTION", pre_rollback_status="DECLARED_PROMOTION_BLOCKED", post_rollback_status="PROMOTION_BLOCKED", disable_required=False, restore_required=False)
    triggers = (t_replay, t_fast, t_bench)
    targets = (base, cand, fast, impl, promo)
    th = tuple(t.decoder_rollback_trigger_hash for t in triggers)
    steps = (
        build_decoder_rollback_plan_step(step_id="baseline", step_index=0, step_kind="DECLARE_BASELINE_RESTORE", target_hash=base.decoder_rollback_target_hash, trigger_hashes=th, precondition_hashes=(H["a"],), postcondition_hashes=(base.decoder_rollback_target_hash,)),
        build_decoder_rollback_plan_step(step_id="candidate", step_index=1, step_kind="DECLARE_CANDIDATE_DISABLE", target_hash=cand.decoder_rollback_target_hash, trigger_hashes=th, precondition_hashes=(H["1"],), postcondition_hashes=(cand.decoder_rollback_target_hash,)),
        build_decoder_rollback_plan_step(step_id="fast", step_index=2, step_kind="DECLARE_FAST_PATH_DISABLE", target_hash=fast.decoder_rollback_target_hash, trigger_hashes=th, precondition_hashes=(H["2"],), postcondition_hashes=(fast.decoder_rollback_target_hash,)),
        build_decoder_rollback_plan_step(step_id="impl", step_index=3, step_kind="DECLARE_IMPLEMENTATION_DISABLE", target_hash=impl.decoder_rollback_target_hash, trigger_hashes=th, precondition_hashes=(H["3"],), postcondition_hashes=(impl.decoder_rollback_target_hash,)),
        build_decoder_rollback_plan_step(step_id="promotion", step_index=4, step_kind="DECLARE_PROMOTION_BLOCK", target_hash=promo.decoder_rollback_target_hash, trigger_hashes=th, precondition_hashes=(H["4"],), postcondition_hashes=(promo.decoder_rollback_target_hash,)),
    )
    payload = {"plan_id": "declared-plan", "plan_version": "1", "rollback_triggers": triggers, "rollback_targets": targets, "rollback_steps": steps}
    payload.update(overrides)
    return build_decoder_rollback_plan(**payload)


def make_receipt(**overrides):
    payload = {
        "upstream_binding": make_upstream(),
        "rollback_identity": make_identity(),
        "rollback_plan": make_plan(),
        "restoration_policy": build_decoder_rollback_restoration_policy(restoration_policy_id="restore", canonical_baseline_receipt_hash=H["a"]),
        "verification_boundary": build_decoder_rollback_verification_boundary(verification_boundary_id="verify"),
        "execution_boundary": build_decoder_rollback_execution_boundary(execution_boundary_id="execute"),
        "audit_boundary": build_decoder_rollback_audit_boundary(audit_boundary_id="audit"),
        "authority_boundary": build_decoder_rollback_authority_boundary(authority_boundary_id="authority"),
    }
    payload.update(overrides)
    return build_decoder_rollback_receipt(**payload)

def test_happy_path_receipt_is_safe_bound_and_immutable():
    receipt = make_receipt()
    validate_decoder_rollback_receipt(receipt)
    assert rr._HASH_RE.fullmatch(receipt.decoder_rollback_receipt_hash)
    assert receipt.rollback_receipt_safe is True
    assert receipt.rollback_ready_for_future_promotion_gate is True
    assert receipt.rollback_execution_performed_by_receipt is False
    assert receipt.candidate_remains_adapter_only is True
    assert receipt.baseline_restore_declared is True
    assert receipt.promotion_allowed is False
    assert receipt.correctness_claim_allowed is False
    assert receipt.global_correctness_claim_allowed is False
    assert receipt.qec_advantage_claim_allowed is False
    assert receipt.hardware_authority_allowed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.promotion_allowed = True


def test_canonical_json_hash_determinism_and_ordering():
    plan = make_plan()
    shuffled = build_decoder_rollback_plan(plan_id="declared-plan", plan_version="1", rollback_triggers=tuple(reversed(plan.rollback_triggers)), rollback_targets=tuple(reversed(plan.rollback_targets)), rollback_steps=tuple(reversed(plan.rollback_steps)))
    assert plan.rollback_plan_hash == shuffled.rollback_plan_hash
    assert plan.decoder_rollback_plan_hash == shuffled.decoder_rollback_plan_hash
    step = plan.rollback_steps[0]
    rebuilt = build_decoder_rollback_plan_step(step_id=step.step_id, step_index=step.step_index, step_kind=step.step_kind, target_hash=step.target_hash, trigger_hashes=tuple(reversed(step.trigger_hashes)), precondition_hashes=tuple(reversed((H["b"], H["a"]))), postcondition_hashes=tuple(reversed((H["d"], H["c"]))))
    rebuilt2 = build_decoder_rollback_plan_step(step_id=step.step_id, step_index=step.step_index, step_kind=step.step_kind, target_hash=step.target_hash, trigger_hashes=tuple(sorted(step.trigger_hashes)), precondition_hashes=(H["a"], H["b"]), postcondition_hashes=(H["c"], H["d"]))
    assert rebuilt.rollback_plan_step_hash == rebuilt2.rollback_plan_step_hash
    assert make_receipt().decoder_rollback_receipt_hash == make_receipt().decoder_rollback_receipt_hash
    assert "builtins.hash" not in MODULE.read_text()


def test_self_hash_exclusion_and_independent_core_hashes_fail_when_stale():
    receipt = make_receipt()
    object.__setattr__(receipt, "decoder_rollback_receipt_hash", H["5"])
    with pytest.raises(DecoderRollbackError) as exc:
        validate_decoder_rollback_receipt(receipt)
    assert_code(exc, DecoderRollbackErrorCode.HASH_MISMATCH)

    plan = make_plan()
    object.__setattr__(plan, "rollback_plan_hash", H["5"])
    with pytest.raises(DecoderRollbackError) as exc2:
        rr.validate_decoder_rollback_plan(plan)
    assert_code(exc2, DecoderRollbackErrorCode.HASH_MISMATCH)

    policy = build_decoder_rollback_restoration_policy(restoration_policy_id="restore", canonical_baseline_receipt_hash=H["a"])
    object.__setattr__(policy, "rollback_restoration_policy_hash", H["5"])
    with pytest.raises(DecoderRollbackError) as exc3:
        rr.validate_decoder_rollback_restoration_policy(policy)
    assert_code(exc3, DecoderRollbackErrorCode.HASH_MISMATCH)


@pytest.mark.parametrize("field,bad,code", [
    ("previous_release_tag", "v166.5", DecoderRollbackErrorCode.INVALID_INPUT),
    ("previous_release_url", "https://example.test/v166.6", DecoderRollbackErrorCode.INVALID_INPUT),
    ("rollback_release", "v166.8", DecoderRollbackErrorCode.INVALID_INPUT),
    ("upstream_canonical_decoder_baseline_receipt_hash", "A" * 64, DecoderRollbackErrorCode.INVALID_HASH),
    ("candidate_declaration_hash", "short", DecoderRollbackErrorCode.INVALID_HASH),
    ("fast_path_identity_hash", "g" * 64, DecoderRollbackErrorCode.INVALID_HASH),
    ("implementation_identity_hash", "", DecoderRollbackErrorCode.INVALID_HASH),
    ("candidate_name", "", DecoderRollbackErrorCode.INVALID_INPUT),
    ("candidate_version", "", DecoderRollbackErrorCode.INVALID_INPUT),
    ("replay_equivalence_proven_for_declared_corpus", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("optimization_contract_safe", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("fast_path_equivalence_proven_for_declared_corpus", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("implementation_boundary_safe", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("benchmark_ladder_safe", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("candidate_adapter_only", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("candidate_promoted", True, DecoderRollbackErrorCode.INVALID_INPUT),
    ("baseline_immutable", False, DecoderRollbackErrorCode.INVALID_INPUT),
    ("baseline_mutation_allowed", True, DecoderRollbackErrorCode.INVALID_INPUT),
    ("runtime_authority_allowed", True, DecoderRollbackErrorCode.INVALID_INPUT),
    ("candidate_adapter_only", 1, DecoderRollbackErrorCode.INVALID_INPUT),
])
def test_upstream_binding_validation(field, bad, code):
    with pytest.raises(DecoderRollbackError) as exc:
        make_upstream(**{field: bad})
    assert_code(exc, code)


@pytest.mark.parametrize("field,bad", [
    ("rollback_id", ""), ("rollback_name", ""), ("rollback_version", ""), ("rollback_kind", "UNKNOWN"),
    ("rollback_status", "ROLLBACK_EXECUTED"), ("rollback_mode", "EXECUTE"), ("associated_candidate_declaration_hash", "bad"),
    ("rollback_receipt_ready_for_future_promotion_gate", False), ("rollback_execution_performed_by_receipt", True),
    ("rollback_authority_allowed", True), ("promotion_allowed", True), ("correctness_claim_allowed", True),
    ("global_correctness_claim_allowed", True), ("hardware_authority_allowed", True), ("qec_advantage_claim_allowed", True),
    ("rollback_name", "candidate decoder promoted"), ("rollback_id", "rollback-executed"),
])
def test_identity_validation_rejects_unsafe_fields(field, bad):
    with pytest.raises(DecoderRollbackError) as exc:
        make_identity(**{field: bad})
    assert exc.value.code in {DecoderRollbackErrorCode.INVALID_INPUT, DecoderRollbackErrorCode.INVALID_HASH}


@pytest.mark.parametrize("field,bad", [
    ("trigger_id", ""), ("trigger_kind", "CUSTOM"), ("trigger_source_receipt_hash", "bad"),
    ("trigger_source_release", "v166.1"), ("trigger_scope", "OTHER"), ("trigger_detection_mode", "RUNTIME_DETECTION"),
    ("trigger_severity", "LOW"), ("rollback_required", False), ("candidate_disable_required", False),
    ("fast_path_disable_required", False), ("implementation_disable_required", False), ("baseline_restore_required", False),
    ("promotion_blocked", False), ("rollback_trigger_authority_allowed", True), ("trigger_id", "runtime promotion"),
])
def test_trigger_validation_rejects_unsafe_fields(field, bad):
    payload = {"trigger_id": "tr", "trigger_kind": "REPLAY_EQUIVALENCE_FAILURE", "trigger_source_receipt_hash": H["c"], "trigger_source_release": "v166.2", "trigger_severity": "BLOCK_PROMOTION"}
    payload[field] = bad
    with pytest.raises(DecoderRollbackError):
        build_decoder_rollback_trigger(**payload)


@pytest.mark.parametrize("field,bad", [
    ("target_id", ""), ("target_kind", "CUSTOM"), ("target_hash", "bad"), ("target_role", "CUSTOM"),
    ("pre_rollback_status", "UNKNOWN"), ("post_rollback_status", "UNKNOWN"), ("mutation_allowed", True),
    ("deletion_allowed", True), ("runtime_disable_only", False), ("rollback_target_authority_allowed", True),
    ("target_id", "rollback path deletion allowed"),
])
def test_target_validation_rejects_unsafe_fields(field, bad):
    payload = {"target_id": "candidate", "target_kind": "CANDIDATE_DECLARATION_TARGET", "target_hash": H["1"], "target_role": "DISABLE_CANDIDATE", "pre_rollback_status": "DECLARED_ADAPTER_ONLY", "post_rollback_status": "CANDIDATE_DISABLED", "disable_required": True, "restore_required": False}
    payload[field] = bad
    with pytest.raises(DecoderRollbackError):
        build_decoder_rollback_target(**payload)
    with pytest.raises(DecoderRollbackError):
        build_decoder_rollback_target(target_id="baseline", target_kind="CANONICAL_BASELINE_TARGET", target_hash=H["a"], target_role="DISABLE_CANDIDATE", pre_rollback_status="DECLARED_ACTIVE", post_rollback_status="BASELINE_RESTORED", disable_required=True, restore_required=True)


@pytest.mark.parametrize("field,bad", [
    ("step_id", ""), ("step_index", True), ("step_index", -1), ("step_kind", "CUSTOM"), ("target_hash", "bad"),
    ("trigger_hashes", ()), ("trigger_hashes", (H["a"], H["a"])), ("trigger_hashes", ("bad",)),
    ("precondition_hashes", ()), ("postcondition_hashes", ()), ("step_mode", "EXECUTE"),
    ("execution_allowed", True), ("filesystem_mutation_allowed", True), ("decoder_import_allowed", True),
    ("runtime_execution_allowed", True), ("deterministic_ordering_required", False),
])
def test_plan_step_validation_rejects_unsafe_fields(field, bad):
    payload = {"step_id": "s", "step_index": 0, "step_kind": "DECLARE_CANDIDATE_DISABLE", "target_hash": H["1"], "trigger_hashes": (H["a"],), "precondition_hashes": (H["b"],), "postcondition_hashes": (H["c"],)}
    payload[field] = bad
    with pytest.raises(DecoderRollbackError):
        build_decoder_rollback_plan_step(**payload)


def test_plan_validation_rejects_counts_order_sparse_duplicates_and_missing_links():
    plan = make_plan()
    for attr, bad in [("plan_id", ""), ("plan_version", ""), ("plan_mode", "OTHER"), ("trigger_count", True), ("target_count", 99), ("terminal_rollback_status", "OTHER"), ("rollback_execution_allowed", True), ("baseline_restore_declared", False)]:
        with pytest.raises(DecoderRollbackError):
            rr.validate_decoder_rollback_plan(dataclasses.replace(plan, **{attr: bad}))
    with pytest.raises(DecoderRollbackError):
        rr.validate_decoder_rollback_plan(dataclasses.replace(plan, rollback_triggers=()))
    with pytest.raises(DecoderRollbackError):
        rr.validate_decoder_rollback_plan(dataclasses.replace(plan, rollback_triggers=(object(),)))
    with pytest.raises(DecoderRollbackError):
        rr.validate_decoder_rollback_plan(dataclasses.replace(plan, rollback_triggers=(plan.rollback_triggers[0], plan.rollback_triggers[0])))
    first = plan.rollback_steps[0]
    bad_step = build_decoder_rollback_plan_step(step_id=first.step_id, step_index=first.step_index, step_kind=first.step_kind, target_hash=H["5"], trigger_hashes=first.trigger_hashes, precondition_hashes=first.precondition_hashes, postcondition_hashes=first.postcondition_hashes)
    with pytest.raises(DecoderRollbackError):
        build_decoder_rollback_plan(plan_id="declared-plan", plan_version="1", rollback_triggers=plan.rollback_triggers, rollback_targets=plan.rollback_targets, rollback_steps=(bad_step,) + plan.rollback_steps[1:])


@pytest.mark.parametrize("builder,base,field_bad", [
    (build_decoder_rollback_restoration_policy, {"restoration_policy_id": "r", "canonical_baseline_receipt_hash": H["a"]}, [("restoration_policy_id", ""), ("restoration_mode", "OTHER"), ("canonical_baseline_receipt_hash", "bad"), ("baseline_restore_required", False), ("baseline_restore_mode", "OTHER"), ("baseline_source_mutation_allowed", True), ("baseline_runtime_replacement_allowed", True), ("candidate_disable_required", False), ("fast_path_disable_required", False), ("implementation_disable_required", False), ("benchmark_result_reuse_allowed_after_rollback", True)]),
    (build_decoder_rollback_verification_boundary, {"verification_boundary_id": "v"}, [("verification_boundary_id", ""), ("verification_mode", "OTHER"), ("static_plan_validation_required", False), ("runtime_rollback_execution_required", True), ("verification_complete_for_declaration", False)]),
    (build_decoder_rollback_execution_boundary, {"execution_boundary_id": "e"}, [("execution_boundary_id", ""), ("execution_boundary_mode", "OTHER"), ("declared_rollback_receipt_only", False), ("rollback_execution_allowed", True), ("decoder_import_allowed", True), ("candidate_import_allowed", True), ("fast_path_import_allowed", True), ("implementation_import_allowed", True), ("benchmark_import_allowed", True), ("runtime_decoder_execution_allowed", True), ("rollback_runtime_execution_allowed", True), ("filesystem_mutation_allowed", True), ("git_operation_allowed", True), ("subprocess_rollback_allowed", True), ("network_allowed", True), ("heavy_backend_import_allowed", True), ("hardware_sdk_allowed", True), ("hardware_sdk_allowed", 1)]),
    (build_decoder_rollback_audit_boundary, {"audit_boundary_id": "a"}, [("audit_boundary_id", ""), ("audit_mode", "OTHER"), ("upstream_receipts_review_required", False), ("future_promotion_receipt_required", False), ("audit_complete_for_rollback_receipt", False), ("audit_authority_allowed", True)]),
    (build_decoder_rollback_authority_boundary, {"authority_boundary_id": "auth"}, [("authority_boundary_id", ""), ("authority_mode", "OTHER"), ("candidate_adapter_only", False), ("rollback_receipt_authority_allowed", True), ("rollback_execution_authority_allowed", True), ("runtime_authority_allowed", True), ("implementation_authority_allowed", True), ("benchmark_authority_allowed", True), ("promotion_authority_allowed", True), ("hardware_authority_allowed", True), ("ml_decoder_authority_allowed", True), ("probabilistic_decoder_authority_allowed", True), ("qec_advantage_claim_allowed", True), ("global_correctness_claim_allowed", True), ("silent_replacement_allowed", True), ("baseline_mutation_allowed", True), ("rollback_path_deletion_allowed", True), ("candidate_promotion_allowed", True)]),
])
def test_boundary_builders_reject_unsafe_fields(builder, base, field_bad):
    for field, bad in field_bad:
        with pytest.raises(DecoderRollbackError):
            builder(**{**base, field: bad})


def test_aggregate_receipt_rejects_mismatches_and_forged_flags():
    receipt = make_receipt()
    for attr, bad in [("receipt_version", "v166.8"), ("receipt_kind", "Other"), ("previous_release_tag", "v166.5"), ("previous_release_url", "url"), ("trigger_count", True), ("target_count", 99), ("rollback_execution_performed_by_receipt", True), ("promotion_allowed", True), ("correctness_claim_allowed", True), ("global_correctness_claim_allowed", True), ("qec_advantage_claim_allowed", True), ("hardware_authority_allowed", True)]:
        with pytest.raises(DecoderRollbackError):
            rr.validate_decoder_rollback_receipt(dataclasses.replace(receipt, **{attr: bad}))
    with pytest.raises(DecoderRollbackError):
        make_receipt(rollback_identity=make_identity(associated_candidate_declaration_hash=H["5"]))
    with pytest.raises(DecoderRollbackError):
        make_receipt(rollback_identity=make_identity(associated_fast_path_identity_hash=H["5"]))
    with pytest.raises(DecoderRollbackError):
        make_receipt(rollback_identity=make_identity(associated_implementation_identity_hash=H["5"]))
    with pytest.raises(DecoderRollbackError):
        make_receipt(rollback_identity=make_identity(associated_benchmark_ladder_receipt_hash=H["5"]))
    with pytest.raises(DecoderRollbackError):
        make_receipt(restoration_policy=build_decoder_rollback_restoration_policy(restoration_policy_id="restore", canonical_baseline_receipt_hash=H["5"]))
    leaking = build_decoder_rollback_authority_boundary(authority_boundary_id="authority")
    object.__setattr__(leaking, "runtime_authority_allowed", True)
    object.__setattr__(leaking, "decoder_rollback_authority_boundary_hash", rr._hash_payload(rr._payload_without(leaking, "decoder_rollback_authority_boundary_hash")))
    with pytest.raises(DecoderRollbackError):
        make_receipt(authority_boundary=leaking)


@pytest.mark.parametrize("phrase", [
    "silent_decoder_replacement", "candidate-replaces-baseline", "decoder replaced because faster",
    "speed   proves correctness", "benchmark\\nproves\\tcorrectness", "benchmark marketing", "runtime promotion",
    "candidate decoder promoted", "probabilistic decoder authority", "ML decoder authority", "hardware authority",
    "QEC advantage proven", "hidden precision drift", "undeclared approximation policy",
    "output accepted as universal canonical truth", "global correctness\nproven", "replay equivalence implies promotion",
    "replay equivalence implies speedup", "optimization implies correctness", "optimization grants execution authority",
    "contract permits implementation", "fast path accepted", "fast path implemented", "fast path runtime enabled",
    "fast path proves speedup", "benchmark proves fast path", "benchmark proves decoder correctness",
    "benchmark replaces replay equivalence", "benchmark replaces rollback", "implementation permission granted",
    "implementation enabled", "implementation proves correctness", "implementation replaces baseline",
    "runtime implementation authority", "rollback permits promotion", "rollback receipt promotes candidate",
    "promotion without receipt", "rollback path deletion allowed", "baseline mutation allowed",
])
def test_forbidden_semantic_hardening(phrase):
    with pytest.raises(DecoderRollbackError) as exc:
        make_identity(rollback_name=f"declared {phrase} boundary")
    assert_code(exc, DecoderRollbackErrorCode.INVALID_INPUT)


@pytest.mark.parametrize("phrase", ["rollback_ready_for_future_promotion_gate", "promotion_blocked", "baseline_restore_declared", "candidate_disable_declared", "rollback_receipt_safe", "rollback_receipt_required_before_promotion"])
def test_positive_semantic_controls_are_allowed(phrase):
    rr._check_forbidden_declaration_semantics(phrase, "positive")


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_static_boundary_no_decoder_heavy_runtime_network_timing_or_git_markers():
    text = MODULE.read_text()
    imports = _imports(MODULE)
    test_imports = _imports(TEST_MODULE)
    banned_prefixes = ("qec.decoder", "numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax")
    for name in imports + test_imports:
        assert not any(name == p or name.startswith(p + ".") for p in banned_prefixes)
    banned_markers = ["importlib", "requests", "urllib", "socket", "time.perf_counter", "time.time", "datetime.now", "run_decoder", "execute_decoder", "fast_path_runtime", "optimization_runtime", "implementation_runtime"]
    for marker in banned_markers:
        assert marker not in text
    created = subprocess.check_output(["git", "diff", "--name-only", "--", "src/qec/decoder/"], cwd=ROOT, text=True).strip()
    assert created == ""
    names = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    assert not any("candidate_decoder" in n or "fast_path" in n and "decoder_rollback_receipts" not in n or "rollback_runtime" in n for n in names)


def test_hash_seed_stability_for_receipt_hash():
    code = "from tests.test_decoder_rollback_receipts import make_receipt; print(make_receipt().decoder_rollback_receipt_hash)"
    env = os.environ.copy(); env["PYTHONPATH"] = f"src{os.pathsep}."
    outputs = []
    for seed in ("0", "1"):
        run_env = {**env, "PYTHONHASHSEED": seed}
        outputs.append(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=run_env, text=True).strip())
    assert outputs[0] == outputs[1] == make_receipt().decoder_rollback_receipt_hash
