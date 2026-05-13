from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .loop_termination_contract import LoopTerminationContract, validate_loop_termination_contract
from .loop_termination_proof import (
    LoopTerminationProof,
    OuroboricConvergenceReceipt,
    validate_loop_termination_proof_with_receipts,
    validate_ouroboric_convergence_receipt_with_receipts,
)
from .recursive_proof_receipt import LoopIterationRecord, RecursiveProofReceipt, validate_recursive_proof_receipt_with_contract

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_SUBSYSTEM_LOOP_TYPE = "INVALID_SUBSYSTEM_LOOP_TYPE"
_ERR_INVALID_SUBSYSTEM_LOOP_LABEL = "INVALID_SUBSYSTEM_LOOP_LABEL"
_ERR_INVALID_SUBSYSTEM_LOOP_STABILITY_CLASS = "INVALID_SUBSYSTEM_LOOP_STABILITY_CLASS"
_ERR_SUBSYSTEM_LOOP_CLASSIFICATION_AMBIGUOUS = "SUBSYSTEM_LOOP_CLASSIFICATION_AMBIGUOUS"
_ERR_SUBSYSTEM_LOOP_MISMATCH = "SUBSYSTEM_LOOP_MISMATCH"
_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH = "SUBSYSTEM_LOOP_COUNT_MISMATCH"
_ERR_SUBSYSTEM_LOOP_STABILITY_CLASS_MISMATCH = "SUBSYSTEM_LOOP_STABILITY_CLASS_MISMATCH"
_ERR_DUPLICATE_SUBSYSTEM_ITERATION = "DUPLICATE_SUBSYSTEM_ITERATION"
_ERR_SUBSYSTEM_LOOP_RECEIPT_MISMATCH = "SUBSYSTEM_LOOP_RECEIPT_MISMATCH"
_ERR_LOOP_ARTIFACT_MISMATCH = "LOOP_ARTIFACT_MISMATCH"

_MAX_SUBSYSTEM_ITERATIONS = 10_000
_MAX_LOOP_LABEL_LENGTH = 96
_MAX_SUBSYSTEM_LOOP_LABEL_LENGTH = 128
_MAX_ITERATION_INDEX = 9_999
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_SUBSYSTEM_LOOP_TYPES = frozenset({"ROUTER", "READOUT", "MARKOV"})
_ALLOWED_SUBSYSTEM_LOOP_STABILITY_CLASSES = frozenset({"SUBSYSTEM_LOOP_STABLE", "SUBSYSTEM_LOOP_TERMINATED_NONCONVERGED", "SUBSYSTEM_LOOP_INCOMPLETE", "SUBSYSTEM_LOOP_EMPTY"})


def get_allowed_subsystem_loop_types() -> frozenset[str]: return _ALLOWED_SUBSYSTEM_LOOP_TYPES

def get_allowed_subsystem_loop_stability_classes() -> frozenset[str]: return _ALLOWED_SUBSYSTEM_LOOP_STABILITY_CLASSES

def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v

def _validate_i(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool): raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    return v

def _validate_loop_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_LOOP_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_INPUT)
    return v

def _validate_subsystem_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_SUBSYSTEM_LOOP_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_SUBSYSTEM_LOOP_LABEL)
    return v

def _subsystem_loop_receipt_payload(**k: Any) -> dict[str, Any]:
    p = dict(k); p["iteration_record_hashes"] = list(k["iteration_record_hashes"]); return p

def _derive_stability(c: int, stable: bool, klass: str) -> tuple[bool, str]:
    if c == 0: return True, "SUBSYSTEM_LOOP_EMPTY"
    if stable is True: return True, "SUBSYSTEM_LOOP_STABLE"
    if klass == "OUROBORIC_INCOMPLETE": return False, "SUBSYSTEM_LOOP_INCOMPLETE"
    return False, "SUBSYSTEM_LOOP_TERMINATED_NONCONVERGED"

def _token_match(v: str, t: str) -> bool:
    lp = [x.lower() for x in v.split("_")]; low = v.lower()
    if t == "ROUTER": return "router" in lp or "routers" in lp or v.startswith("Router")
    if t == "READOUT": return "readout" in lp or "readouts" in lp or v.startswith("Readout")
    return "markov" in lp or "markov_basis" in lp or "markovbasis" in lp or v.startswith("Markov") or "markov" in low.split("_")

def _classify_loop_subsystem(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt) -> str | None:
    if not isinstance(loop_termination_contract, LoopTerminationContract) or not isinstance(recursive_proof_receipt, RecursiveProofReceipt): raise ValueError(_ERR_INVALID_INPUT)
    fs = [loop_termination_contract.source_artifact_type, loop_termination_contract.loop_label, loop_termination_contract.input_receipt_hash_field, loop_termination_contract.output_receipt_hash_field, recursive_proof_receipt.loop_label]
    m = {t for t in _ALLOWED_SUBSYSTEM_LOOP_TYPES if any(_token_match(f, t) for f in fs)}
    if len(m) > 1: raise ValueError(_ERR_SUBSYSTEM_LOOP_CLASSIFICATION_AMBIGUOUS)
    return next(iter(m)) if m else None

def _select_iteration_records_for_subsystem(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, subsystem_loop_type: str) -> tuple[LoopIterationRecord, ...]:
    if subsystem_loop_type not in _ALLOWED_SUBSYSTEM_LOOP_TYPES: raise ValueError(_ERR_INVALID_SUBSYSTEM_LOOP_TYPE)
    s = _classify_loop_subsystem(loop_termination_contract, recursive_proof_receipt)
    if s is None: return ()
    if s != subsystem_loop_type: raise ValueError(_ERR_SUBSYSTEM_LOOP_MISMATCH)
    return recursive_proof_receipt.iteration_records

@dataclass(frozen=True)
class RouterLoopReceipt:
    loop_termination_contract_hash: str; recursive_proof_receipt_hash: str; loop_termination_proof_hash: str; ouroboric_convergence_receipt_hash: str
    source_artifact_type: str; source_artifact_hash: str; subsystem_loop_type: str; subsystem_loop_label: str; loop_label: str; loop_mode: str; max_depth: int
    iteration_record_hashes: tuple[str, ...]; subsystem_iteration_count: int; changed_iteration_count: int; terminal_iteration_index: int | None; final_output_receipt_hash: str | None
    termination_class: str; convergence_class: str; subsystem_loop_stable: bool; subsystem_loop_stability_class: str; router_loop_receipt_hash: str
    def __post_init__(self) -> None: validate_router_loop_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        p = _subsystem_loop_receipt_payload(**{k:getattr(self,k) for k in self.__dataclass_fields__ if k!="router_loop_receipt_hash"})
        return {**p, "router_loop_receipt_hash": self.router_loop_receipt_hash}
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class ReadoutLoopReceipt:
    loop_termination_contract_hash: str; recursive_proof_receipt_hash: str; loop_termination_proof_hash: str; ouroboric_convergence_receipt_hash: str
    source_artifact_type: str; source_artifact_hash: str; subsystem_loop_type: str; subsystem_loop_label: str; loop_label: str; loop_mode: str; max_depth: int
    iteration_record_hashes: tuple[str, ...]; subsystem_iteration_count: int; changed_iteration_count: int; terminal_iteration_index: int | None; final_output_receipt_hash: str | None
    termination_class: str; convergence_class: str; subsystem_loop_stable: bool; subsystem_loop_stability_class: str; readout_loop_receipt_hash: str
    def __post_init__(self) -> None: validate_readout_loop_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        p = _subsystem_loop_receipt_payload(**{k:getattr(self,k) for k in self.__dataclass_fields__ if k!="readout_loop_receipt_hash"})
        return {**p, "readout_loop_receipt_hash": self.readout_loop_receipt_hash}
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class MarkovLoopStabilityReceipt(ReadoutLoopReceipt):
    markov_loop_stability_receipt_hash: str
    def __post_init__(self) -> None: validate_markov_loop_stability_receipt(self)


def _build_common(contract: LoopTerminationContract, recursive: RecursiveProofReceipt, proof: LoopTerminationProof, conv: OuroboricConvergenceReceipt, subtype: str, sublabel: str):
    validate_loop_termination_contract(contract)
    validate_recursive_proof_receipt_with_contract(recursive, contract)
    validate_loop_termination_proof_with_receipts(proof, contract, recursive)
    validate_ouroboric_convergence_receipt_with_receipts(conv, contract, recursive, proof)
    if recursive.loop_termination_contract_hash != contract.loop_termination_contract_hash or proof.recursive_proof_receipt_hash != recursive.recursive_proof_receipt_hash or conv.loop_termination_proof_hash != proof.loop_termination_proof_hash:
        raise ValueError(_ERR_LOOP_ARTIFACT_MISMATCH)
    rs = _select_iteration_records_for_subsystem(contract, recursive, subtype)
    hashes = tuple(r.loop_iteration_record_hash for r in rs)
    cc = sum(1 for r in rs if r.changed)
    cnt = len(hashes)
    terminal = recursive.terminal_iteration_index if cnt else None
    fout = recursive.final_output_receipt_hash if cnt else None
    stable, stability_class = _derive_stability(cnt, conv.convergence_stable, conv.convergence_class)
    payload = dict(
        loop_termination_contract_hash=contract.loop_termination_contract_hash,
        recursive_proof_receipt_hash=recursive.recursive_proof_receipt_hash,
        loop_termination_proof_hash=proof.loop_termination_proof_hash,
        ouroboric_convergence_receipt_hash=conv.ouroboric_convergence_receipt_hash,
        source_artifact_type=contract.source_artifact_type,
        source_artifact_hash=contract.source_artifact_hash,
        subsystem_loop_type=subtype,
        subsystem_loop_label=sublabel,
        loop_label=contract.loop_label,
        loop_mode=contract.loop_mode,
        max_depth=contract.max_depth,
        iteration_record_hashes=hashes,
        subsystem_iteration_count=cnt,
        changed_iteration_count=cc,
        terminal_iteration_index=terminal,
        final_output_receipt_hash=fout,
        termination_class=proof.termination_class,
        convergence_class=conv.convergence_class,
        subsystem_loop_stable=stable,
        subsystem_loop_stability_class=stability_class,
    )
    return payload

def build_router_loop_receipt(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> RouterLoopReceipt:
    p = _build_common(loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, "ROUTER", "ROUTER_LOOP")
    return RouterLoopReceipt(**p, router_loop_receipt_hash=sha256_hex(_subsystem_loop_receipt_payload(**p)))

def build_readout_loop_receipt(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> ReadoutLoopReceipt:
    p = _build_common(loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, "READOUT", "READOUT_LOOP")
    return ReadoutLoopReceipt(**p, readout_loop_receipt_hash=sha256_hex(_subsystem_loop_receipt_payload(**p)))

def build_markov_loop_stability_receipt(loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> MarkovLoopStabilityReceipt:
    p = _build_common(loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, "MARKOV", "MARKOV_LOOP_STABILITY")
    return MarkovLoopStabilityReceipt(**p, markov_loop_stability_receipt_hash=sha256_hex(_subsystem_loop_receipt_payload(**p)))

def _validate_common(r: Any, t: str, l: str, hf: str) -> bool:
    for f in ["loop_termination_contract_hash","recursive_proof_receipt_hash","loop_termination_proof_hash","ouroboric_convergence_receipt_hash",hf]: _validate_sha(getattr(r,f))
    _validate_sha(r.source_artifact_hash); _validate_subsystem_label(r.subsystem_loop_label); _validate_loop_label(r.loop_label); _validate_i(r.max_depth)
    if r.subsystem_loop_type != t: raise ValueError(_ERR_INVALID_SUBSYSTEM_LOOP_TYPE)
    if r.subsystem_loop_label != l: raise ValueError(_ERR_INVALID_SUBSYSTEM_LOOP_LABEL)
    if not isinstance(r.iteration_record_hashes, tuple): raise ValueError(_ERR_INVALID_INPUT)
    if len(r.iteration_record_hashes) > _MAX_SUBSYSTEM_ITERATIONS: raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if len(set(r.iteration_record_hashes)) != len(r.iteration_record_hashes): raise ValueError(_ERR_DUPLICATE_SUBSYSTEM_ITERATION)
    for h in r.iteration_record_hashes: _validate_sha(h)
    c = _validate_i(r.subsystem_iteration_count); ch = _validate_i(r.changed_iteration_count)
    if c != len(r.iteration_record_hashes): raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if ch > c: raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if r.terminal_iteration_index is not None and (not isinstance(r.terminal_iteration_index,int) or isinstance(r.terminal_iteration_index,bool) or not (0 <= r.terminal_iteration_index <= _MAX_ITERATION_INDEX)): raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if c==0 and (r.final_output_receipt_hash is not None or r.terminal_iteration_index is not None): raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if c>0 and (r.final_output_receipt_hash is None or r.terminal_iteration_index is None): raise ValueError(_ERR_SUBSYSTEM_LOOP_COUNT_MISMATCH)
    if r.final_output_receipt_hash is not None: _validate_sha(r.final_output_receipt_hash)
    if r.subsystem_loop_stability_class not in _ALLOWED_SUBSYSTEM_LOOP_STABILITY_CLASSES: raise ValueError(_ERR_INVALID_SUBSYSTEM_LOOP_STABILITY_CLASS)
    exp_s, exp_c = _derive_stability(c, r.subsystem_loop_stable, r.convergence_class)
    if exp_s != r.subsystem_loop_stable or exp_c != r.subsystem_loop_stability_class: raise ValueError(_ERR_SUBSYSTEM_LOOP_STABILITY_CLASS_MISMATCH)
    d = r.to_dict(); got = d.pop(hf); expected = sha256_hex(d)
    if expected != got: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_router_loop_receipt(receipt: RouterLoopReceipt) -> bool:
    if not isinstance(receipt, RouterLoopReceipt): raise ValueError(_ERR_INVALID_INPUT)
    return _validate_common(receipt, "ROUTER", "ROUTER_LOOP", "router_loop_receipt_hash")

def validate_readout_loop_receipt(receipt: ReadoutLoopReceipt) -> bool:
    if not isinstance(receipt, ReadoutLoopReceipt): raise ValueError(_ERR_INVALID_INPUT)
    return _validate_common(receipt, "READOUT", "READOUT_LOOP", "readout_loop_receipt_hash")

def validate_markov_loop_stability_receipt(receipt: MarkovLoopStabilityReceipt) -> bool:
    if not isinstance(receipt, MarkovLoopStabilityReceipt): raise ValueError(_ERR_INVALID_INPUT)
    return _validate_common(receipt, "MARKOV", "MARKOV_LOOP_STABILITY", "markov_loop_stability_receipt_hash")

def _validate_with_artifacts(r: Any, c: LoopTerminationContract, rr: RecursiveProofReceipt, p: LoopTerminationProof, o: OuroboricConvergenceReceipt, builder: Any, intrinsic: Any) -> bool:
    intrinsic(r)
    expected = builder(c, rr, p, o)
    if r.to_dict() != expected.to_dict(): raise ValueError(_ERR_SUBSYSTEM_LOOP_RECEIPT_MISMATCH)
    return True

def validate_router_loop_receipt_with_artifacts(receipt: RouterLoopReceipt, loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> bool:
    return _validate_with_artifacts(receipt, loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, build_router_loop_receipt, validate_router_loop_receipt)

def validate_readout_loop_receipt_with_artifacts(receipt: ReadoutLoopReceipt, loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> bool:
    return _validate_with_artifacts(receipt, loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, build_readout_loop_receipt, validate_readout_loop_receipt)

def validate_markov_loop_stability_receipt_with_artifacts(receipt: MarkovLoopStabilityReceipt, loop_termination_contract: LoopTerminationContract, recursive_proof_receipt: RecursiveProofReceipt, loop_termination_proof: LoopTerminationProof, ouroboric_convergence_receipt: OuroboricConvergenceReceipt) -> bool:
    return _validate_with_artifacts(receipt, loop_termination_contract, recursive_proof_receipt, loop_termination_proof, ouroboric_convergence_receipt, build_markov_loop_stability_receipt, validate_markov_loop_stability_receipt)
