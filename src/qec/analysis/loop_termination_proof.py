from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .loop_termination_contract import LoopTerminationContract, validate_loop_termination_contract
from .recursive_proof_receipt import (
    LoopIterationRecord,
    RecursiveProofReceipt,
    get_allowed_loop_iteration_statuses,
    validate_loop_iteration_record,
    validate_recursive_proof_receipt,
    validate_recursive_proof_receipt_with_contract,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_LOOP_CONTRACT_MISMATCH = "LOOP_CONTRACT_MISMATCH"
_ERR_RECURSIVE_PROOF_RECEIPT_MISMATCH = "RECURSIVE_PROOF_RECEIPT_MISMATCH"
_ERR_INVALID_TERMINATION_CLASS = "INVALID_TERMINATION_CLASS"
_ERR_TERMINATION_CLASS_MISMATCH = "TERMINATION_CLASS_MISMATCH"
_ERR_TERMINATION_SATISFACTION_MISMATCH = "TERMINATION_SATISFACTION_MISMATCH"
_ERR_INVALID_CONVERGENCE_MODE = "INVALID_CONVERGENCE_MODE"
_ERR_INVALID_CONVERGENCE_CLASS = "INVALID_CONVERGENCE_CLASS"
_ERR_CONVERGENCE_CLASS_MISMATCH = "CONVERGENCE_CLASS_MISMATCH"
_ERR_CONVERGENCE_STABILITY_MISMATCH = "CONVERGENCE_STABILITY_MISMATCH"
_ERR_LOOP_TERMINATION_PROOF_MISMATCH = "LOOP_TERMINATION_PROOF_MISMATCH"
_ERR_OUROBORIC_CONVERGENCE_RECEIPT_MISMATCH = "OUROBORIC_CONVERGENCE_RECEIPT_MISMATCH"
_ERR_ITERATION_COUNT_MISMATCH = "ITERATION_COUNT_MISMATCH"
_ERR_FINAL_HASH_MISMATCH = "FINAL_HASH_MISMATCH"

_MAX_ITERATION_COUNT = 10_000
_MAX_LOOP_LABEL_LENGTH = 96

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_ALLOWED_TERMINATION_CLASSES = frozenset({
    "LOOP_TERMINATED_BY_MAX_DEPTH",
    "LOOP_TERMINATED_BY_FIXED_POINT",
    "LOOP_TERMINATED_BY_TARGET_HASH",
    "LOOP_TERMINATED_BY_STATUS",
    "LOOP_TERMINATED_BY_DIVERGENCE_BOUND",
    "LOOP_TERMINATION_UNSATISFIED",
})

_ALLOWED_CONVERGENCE_MODES = frozenset({"DETERMINISTIC_OUROBORIC_HASH_CONVERGENCE"})
_ALLOWED_CONVERGENCE_CLASSES = frozenset({
    "OUROBORIC_CLOSED_LOOP",
    "OUROBORIC_FIXED_POINT",
    "OUROBORIC_TERMINATED_NONCONVERGED",
    "OUROBORIC_INCOMPLETE",
})


def get_allowed_loop_termination_classes() -> frozenset[str]:
    return _ALLOWED_TERMINATION_CLASSES


def get_allowed_ouroboric_convergence_modes() -> frozenset[str]:
    return _ALLOWED_CONVERGENCE_MODES


def get_allowed_ouroboric_convergence_classes() -> frozenset[str]:
    return _ALLOWED_CONVERGENCE_CLASSES


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_int(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    return v


def _validate_loop_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_LOOP_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _derive_termination(termination_policy: str, canonical_termination_parameters: str, final_input_receipt_hash: str, final_output_receipt_hash: str, terminal_iteration_status: str, iteration_count: int, max_depth: int, changed_iteration_count: int) -> tuple[bool, str]:
    params = json.loads(canonical_termination_parameters)
    if termination_policy == "MAX_DEPTH_ONLY":
        ok = terminal_iteration_status == "ITERATION_MAX_DEPTH_REACHED" and iteration_count == max_depth
        return ok, "LOOP_TERMINATED_BY_MAX_DEPTH" if ok else "LOOP_TERMINATION_UNSATISFIED"
    if termination_policy == "FIXED_POINT_HASH":
        ok = terminal_iteration_status == "ITERATION_FIXED_POINT" and final_input_receipt_hash == final_output_receipt_hash
        return ok, "LOOP_TERMINATED_BY_FIXED_POINT" if ok else "LOOP_TERMINATION_UNSATISFIED"
    if termination_policy == "TARGET_HASH_REACHED":
        ok = terminal_iteration_status == "ITERATION_TARGET_REACHED" and final_output_receipt_hash == params["target_hash"]
        return ok, "LOOP_TERMINATED_BY_TARGET_HASH" if ok else "LOOP_TERMINATION_UNSATISFIED"
    if termination_policy == "STATUS_FIELD_MATCH":
        ok = terminal_iteration_status == "ITERATION_STATUS_TERMINAL"
        return ok, "LOOP_TERMINATED_BY_STATUS" if ok else "LOOP_TERMINATION_UNSATISFIED"
    if termination_policy == "BOUNDED_DIVERGENCE_COUNT":
        ok = terminal_iteration_status == "ITERATION_DIVERGENCE_LIMIT" and changed_iteration_count <= params["max_divergence_count"]
        return ok, "LOOP_TERMINATED_BY_DIVERGENCE_BOUND" if ok else "LOOP_TERMINATION_UNSATISFIED"
    return False, "LOOP_TERMINATION_UNSATISFIED"


def _loop_termination_proof_payload(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _ouroboric_convergence_receipt_payload(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

@dataclass(frozen=True)
class LoopTerminationProof:
    loop_termination_contract_hash: str
    recursive_proof_receipt_hash: str
    source_artifact_type: str
    source_artifact_hash: str
    loop_label: str
    loop_mode: str
    termination_policy: str
    max_depth: int
    iteration_count: int
    terminal_iteration_index: int
    terminal_iteration_status: str
    first_input_receipt_hash: str
    final_output_receipt_hash: str
    changed_iteration_count: int
    termination_satisfied: bool
    termination_class: str
    loop_termination_proof_hash: str
    def __post_init__(self) -> None:
        validate_loop_termination_proof(self)
    def to_dict(self) -> dict[str, Any]:
        p = _loop_termination_proof_payload(**{k: v for k, v in self.__dict__.items() if k != 'loop_termination_proof_hash'})
        return {**p, 'loop_termination_proof_hash': self.loop_termination_proof_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class OuroboricConvergenceReceipt:
    loop_termination_contract_hash: str
    recursive_proof_receipt_hash: str
    loop_termination_proof_hash: str
    loop_label: str
    convergence_mode: str
    first_input_receipt_hash: str
    final_output_receipt_hash: str
    final_input_receipt_hash: str
    cycle_closed: bool
    fixed_point_reached: bool
    convergence_stable: bool
    convergence_class: str
    iteration_count: int
    ouroboric_convergence_receipt_hash: str
    def __post_init__(self) -> None:
        validate_ouroboric_convergence_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        p = _ouroboric_convergence_receipt_payload(**{k: v for k, v in self.__dict__.items() if k != 'ouroboric_convergence_receipt_hash'})
        return {**p, 'ouroboric_convergence_receipt_hash': self.ouroboric_convergence_receipt_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_loop_termination_proof(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt) -> LoopTerminationProof:
    validate_loop_termination_contract(loop_termination_contract)
    validate_recursive_proof_receipt_with_contract(recursive_proof_receipt, loop_termination_contract)
    if recursive_proof_receipt.loop_termination_contract_hash != loop_termination_contract.loop_termination_contract_hash:
        raise ValueError(_ERR_LOOP_CONTRACT_MISMATCH)
    first = recursive_proof_receipt.iteration_records[0]
    final = recursive_proof_receipt.iteration_records[-1]
    changed_count = sum(1 for r in recursive_proof_receipt.iteration_records if r.changed)
    term_ok, term_class = _derive_termination(loop_termination_contract.termination_policy, loop_termination_contract.canonical_termination_parameters, final.input_receipt_hash, recursive_proof_receipt.final_output_receipt_hash, final.iteration_status, recursive_proof_receipt.iteration_count, loop_termination_contract.max_depth, changed_count)
    payload = _loop_termination_proof_payload(
        loop_termination_contract_hash=loop_termination_contract.loop_termination_contract_hash,
        recursive_proof_receipt_hash=recursive_proof_receipt.recursive_proof_receipt_hash,
        source_artifact_type=recursive_proof_receipt.source_artifact_type,
        source_artifact_hash=recursive_proof_receipt.source_artifact_hash,
        loop_label=recursive_proof_receipt.loop_label,
        loop_mode=recursive_proof_receipt.loop_mode,
        termination_policy=loop_termination_contract.termination_policy,
        max_depth=loop_termination_contract.max_depth,
        iteration_count=recursive_proof_receipt.iteration_count,
        terminal_iteration_index=recursive_proof_receipt.terminal_iteration_index,
        terminal_iteration_status=final.iteration_status,
        first_input_receipt_hash=first.input_receipt_hash,
        final_output_receipt_hash=recursive_proof_receipt.final_output_receipt_hash,
        changed_iteration_count=changed_count,
        termination_satisfied=term_ok,
        termination_class=term_class,
    )
    return LoopTerminationProof(**payload, loop_termination_proof_hash=sha256_hex(payload))


def validate_loop_termination_proof(proof: LoopTerminationProof) -> bool:
    if not isinstance(proof, LoopTerminationProof):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(proof.loop_termination_contract_hash); _validate_sha(proof.recursive_proof_receipt_hash); _validate_sha(proof.source_artifact_hash); _validate_loop_label(proof.loop_label)
    _validate_int(proof.max_depth); _validate_int(proof.iteration_count); _validate_int(proof.terminal_iteration_index); _validate_int(proof.changed_iteration_count)
    if not (1 <= proof.iteration_count <= _MAX_ITERATION_COUNT) or proof.changed_iteration_count < 0 or proof.changed_iteration_count > proof.iteration_count or proof.terminal_iteration_index != proof.iteration_count - 1:
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    _validate_sha(proof.first_input_receipt_hash); _validate_sha(proof.final_output_receipt_hash)
    if proof.terminal_iteration_status not in get_allowed_loop_iteration_statuses():
        raise ValueError(_ERR_INVALID_INPUT)
    if proof.termination_class not in _ALLOWED_TERMINATION_CLASSES:
        raise ValueError(_ERR_INVALID_TERMINATION_CLASS)
    if not isinstance(proof.termination_satisfied, bool):
        raise ValueError(_ERR_TERMINATION_SATISFACTION_MISMATCH)
    _validate_sha(proof.loop_termination_proof_hash)
    payload = _loop_termination_proof_payload(**{k: v for k, v in proof.__dict__.items() if k != 'loop_termination_proof_hash'})
    if sha256_hex(payload) != proof.loop_termination_proof_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def build_ouroboric_convergence_receipt(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof) -> OuroboricConvergenceReceipt:
    validate_loop_termination_proof_with_receipts(loop_termination_proof, loop_termination_contract, recursive_proof_receipt)
    first = recursive_proof_receipt.iteration_records[0]
    final = recursive_proof_receipt.iteration_records[-1]
    cycle_closed = recursive_proof_receipt.final_output_receipt_hash == first.input_receipt_hash
    fixed_point = recursive_proof_receipt.final_output_receipt_hash == final.input_receipt_hash
    if not loop_termination_proof.termination_satisfied:
        stable = False; klass = "OUROBORIC_INCOMPLETE"
    elif cycle_closed:
        stable = True; klass = "OUROBORIC_CLOSED_LOOP"
    elif fixed_point:
        stable = True; klass = "OUROBORIC_FIXED_POINT"
    else:
        stable = False; klass = "OUROBORIC_TERMINATED_NONCONVERGED"
    payload = _ouroboric_convergence_receipt_payload(
        loop_termination_contract_hash=loop_termination_contract.loop_termination_contract_hash,
        recursive_proof_receipt_hash=recursive_proof_receipt.recursive_proof_receipt_hash,
        loop_termination_proof_hash=loop_termination_proof.loop_termination_proof_hash,
        loop_label=loop_termination_proof.loop_label,
        convergence_mode="DETERMINISTIC_OUROBORIC_HASH_CONVERGENCE",
        first_input_receipt_hash=first.input_receipt_hash,
        final_output_receipt_hash=recursive_proof_receipt.final_output_receipt_hash,
        final_input_receipt_hash=final.input_receipt_hash,
        cycle_closed=cycle_closed,
        fixed_point_reached=fixed_point,
        convergence_stable=stable,
        convergence_class=klass,
        iteration_count=recursive_proof_receipt.iteration_count,
    )
    return OuroboricConvergenceReceipt(**payload, ouroboric_convergence_receipt_hash=sha256_hex(payload))


def validate_ouroboric_convergence_receipt(receipt: OuroboricConvergenceReceipt) -> bool:
    if not isinstance(receipt, OuroboricConvergenceReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(receipt.loop_termination_contract_hash); _validate_sha(receipt.recursive_proof_receipt_hash); _validate_sha(receipt.loop_termination_proof_hash)
    _validate_loop_label(receipt.loop_label)
    if receipt.convergence_mode not in _ALLOWED_CONVERGENCE_MODES:
        raise ValueError(_ERR_INVALID_CONVERGENCE_MODE)
    _validate_sha(receipt.first_input_receipt_hash); _validate_sha(receipt.final_output_receipt_hash); _validate_sha(receipt.final_input_receipt_hash)
    if not isinstance(receipt.cycle_closed, bool) or not isinstance(receipt.fixed_point_reached, bool) or not isinstance(receipt.convergence_stable, bool):
        raise ValueError(_ERR_CONVERGENCE_STABILITY_MISMATCH)
    if receipt.cycle_closed != (receipt.final_output_receipt_hash == receipt.first_input_receipt_hash):
        raise ValueError(_ERR_CONVERGENCE_STABILITY_MISMATCH)
    if receipt.fixed_point_reached != (receipt.final_output_receipt_hash == receipt.final_input_receipt_hash):
        raise ValueError(_ERR_CONVERGENCE_STABILITY_MISMATCH)
    _validate_int(receipt.iteration_count)
    if not (1 <= receipt.iteration_count <= _MAX_ITERATION_COUNT):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    if receipt.convergence_class not in _ALLOWED_CONVERGENCE_CLASSES:
        raise ValueError(_ERR_INVALID_CONVERGENCE_CLASS)
    _validate_sha(receipt.ouroboric_convergence_receipt_hash)
    payload = _ouroboric_convergence_receipt_payload(**{k: v for k, v in receipt.__dict__.items() if k != 'ouroboric_convergence_receipt_hash'})
    if sha256_hex(payload) != receipt.ouroboric_convergence_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_loop_termination_proof_with_receipts(proof: LoopTerminationProof, loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt) -> bool:
    validate_loop_termination_contract(loop_termination_contract)
    validate_recursive_proof_receipt_with_contract(recursive_proof_receipt, loop_termination_contract)
    validate_loop_termination_proof(proof)
    expected = build_loop_termination_proof(loop_termination_contract, recursive_proof_receipt)
    if proof.to_dict() != expected.to_dict():
        raise ValueError(_ERR_LOOP_TERMINATION_PROOF_MISMATCH)
    return True


def validate_ouroboric_convergence_receipt_with_receipts(receipt: OuroboricConvergenceReceipt, loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof) -> bool:
    validate_loop_termination_proof_with_receipts(loop_termination_proof, loop_termination_contract, recursive_proof_receipt)
    validate_ouroboric_convergence_receipt(receipt)
    expected = build_ouroboric_convergence_receipt(loop_termination_contract, recursive_proof_receipt, loop_termination_proof)
    if receipt.to_dict() != expected.to_dict():
        raise ValueError(_ERR_OUROBORIC_CONVERGENCE_RECEIPT_MISMATCH)
    return True
