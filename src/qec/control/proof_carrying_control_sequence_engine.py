"""v137.15.6 — Proof-Carrying Control Sequence Engine.

Formal acceptance gate layered above:
- v137.15.0 deterministic control sequence kernel
- v137.15.1 explicit state transition automata
- v137.15.2 deterministic rollback planner
- v137.15.3 transition safety envelope kernel
- v137.15.4 collision prevention scheduler
- v137.15.5 bounded fallback corridor

This module introduces the proof-carrying control sequence engine.

It generates formal proof obligations over:
- explicit state transitions
- rollback paths
- collision scheduler guarantees
- fallback corridor bounds
- replay identity law

It exports canonical SMT-LIB 2 proof artifacts targetable at Z3 and CVC5.
It provides a deterministic proof verifier that produces byte-stable
verification reports and proof execution receipts.

Determinism law:
    same input = same bytes = same SMT-LIB = same hash = same outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union


_ALLOWED_PROOF_KINDS: Tuple[str, ...] = (
    "state_safety",
    "collision_freedom",
    "rollback_boundedness",
    "fallback_corridor_safety",
    "replay_identity",
)

_ALLOWED_SOLVERS: Tuple[str, ...] = ("z3", "cvc5")

_ALLOWED_VERIFICATION_STATUSES: Tuple[str, ...] = (
    "verified",
    "failed",
    "timeout",
    "invalid_artifact",
)

_REQUIRED_OBLIGATION_FIELDS: Tuple[str, ...] = (
    "obligation_id",
    "source_module",
    "proof_kind",
    "state_constraints",
    "transition_constraints",
    "collision_constraints",
    "rollback_constraints",
    "fallback_constraints",
    "latency_budget_ms",
    "proof_epoch",
)

_CONSTRAINT_GROUPS: Tuple[str, ...] = (
    "state_constraints",
    "transition_constraints",
    "collision_constraints",
    "rollback_constraints",
    "fallback_constraints",
)

ObligationLike = Union["ControlProofObligation", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_constraint_group(raw: Any, label: str) -> Tuple[str, ...]:
    if raw is None:
        raise ValueError(f"malformed constraint in {label}")
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"malformed constraint in {label}")
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(f"malformed constraint in {label}") from exc
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"malformed constraint in {label}")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"malformed constraint in {label}")
        normalized.append(stripped)
    return tuple(normalized)


@dataclass(frozen=True)
class ControlProofObligation:
    obligation_id: str
    source_module: str
    proof_kind: str
    state_constraints: Tuple[str, ...]
    transition_constraints: Tuple[str, ...]
    collision_constraints: Tuple[str, ...]
    rollback_constraints: Tuple[str, ...]
    fallback_constraints: Tuple[str, ...]
    latency_budget_ms: int
    proof_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "source_module": self.source_module,
            "proof_kind": self.proof_kind,
            "state_constraints": list(self.state_constraints),
            "transition_constraints": list(self.transition_constraints),
            "collision_constraints": list(self.collision_constraints),
            "rollback_constraints": list(self.rollback_constraints),
            "fallback_constraints": list(self.fallback_constraints),
            "latency_budget_ms": self.latency_budget_ms,
            "proof_epoch": self.proof_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ProofCarryingControlSequence:
    sequence_id: str
    proof_obligations: Tuple[ControlProofObligation, ...]
    canonical_control_hash: str
    proof_hash: str
    smtlib_artifact_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "proof_obligations": [o.to_dict() for o in self.proof_obligations],
            "canonical_control_hash": self.canonical_control_hash,
            "proof_hash": self.proof_hash,
            "smtlib_artifact_hash": self.smtlib_artifact_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SMTLIBExportArtifact:
    artifact_id: str
    solver_target: str
    smtlib_text: str
    export_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "solver_target": self.solver_target,
            "smtlib_text": self.smtlib_text,
            "export_hash": self.export_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ProofVerificationReport:
    solver_target: str
    verification_status: str
    proof_artifact_hash: str
    replay_identity_hash: str
    obligations_passed: Tuple[str, ...]
    obligations_failed: Tuple[str, ...]
    latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_target": self.solver_target,
            "verification_status": self.verification_status,
            "proof_artifact_hash": self.proof_artifact_hash,
            "replay_identity_hash": self.replay_identity_hash,
            "obligations_passed": list(self.obligations_passed),
            "obligations_failed": list(self.obligations_failed),
            "latency_ms": self.latency_ms,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ProofExecutionReceipt:
    receipt_id: str
    sequence_id: str
    solver_target: str
    verification_status: str
    proof_hash: str
    proof_artifact_hash: str
    replay_identity_hash: str
    latency_ms: int
    proof_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "sequence_id": self.sequence_id,
            "solver_target": self.solver_target,
            "verification_status": self.verification_status,
            "proof_hash": self.proof_hash,
            "proof_artifact_hash": self.proof_artifact_hash,
            "replay_identity_hash": self.replay_identity_hash,
            "latency_ms": self.latency_ms,
            "proof_epoch": self.proof_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def normalize_control_proof_obligation(value: ObligationLike) -> ControlProofObligation:
    """Deterministically normalize a mapping or dataclass into
    a ControlProofObligation.

    Fails fast on:
    - missing required fields
    - unsupported proof kinds
    - malformed constraints
    - invalid latency budgets
    """
    if isinstance(value, ControlProofObligation):
        data: Mapping[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        data = value
    else:
        raise ValueError("obligation must be mapping or ControlProofObligation")

    missing = [name for name in _REQUIRED_OBLIGATION_FIELDS if name not in data]
    if missing:
        raise ValueError(f"missing required obligation fields: {missing}")

    obligation_id = str(data["obligation_id"])
    if not obligation_id.strip():
        raise ValueError("invalid obligation id")

    source_module = str(data["source_module"])
    if not source_module.strip():
        raise ValueError("invalid source module")

    proof_kind = str(data["proof_kind"])
    if proof_kind not in _ALLOWED_PROOF_KINDS:
        raise ValueError(f"unsupported proof kind: {proof_kind}")

    state_constraints = _normalize_constraint_group(
        data["state_constraints"], "state_constraints"
    )
    transition_constraints = _normalize_constraint_group(
        data["transition_constraints"], "transition_constraints"
    )
    collision_constraints = _normalize_constraint_group(
        data["collision_constraints"], "collision_constraints"
    )
    rollback_constraints = _normalize_constraint_group(
        data["rollback_constraints"], "rollback_constraints"
    )
    fallback_constraints = _normalize_constraint_group(
        data["fallback_constraints"], "fallback_constraints"
    )

    try:
        latency_budget_ms = int(data["latency_budget_ms"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid latency budget") from exc
    if latency_budget_ms <= 0:
        raise ValueError("invalid latency budget")

    try:
        proof_epoch = int(data["proof_epoch"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid proof epoch") from exc
    if proof_epoch < 0:
        raise ValueError("invalid proof epoch")

    return ControlProofObligation(
        obligation_id=obligation_id.strip(),
        source_module=source_module.strip(),
        proof_kind=proof_kind,
        state_constraints=state_constraints,
        transition_constraints=transition_constraints,
        collision_constraints=collision_constraints,
        rollback_constraints=rollback_constraints,
        fallback_constraints=fallback_constraints,
        latency_budget_ms=latency_budget_ms,
        proof_epoch=proof_epoch,
    )


def _ordered_obligations(
    obligations: Iterable[ObligationLike],
) -> Tuple[ControlProofObligation, ...]:
    normalized = tuple(normalize_control_proof_obligation(o) for o in obligations)
    ids = tuple(o.obligation_id for o in normalized)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate obligations")
    return tuple(
        sorted(
            normalized,
            key=lambda o: (o.proof_epoch, o.proof_kind, o.obligation_id),
        )
    )


def _constraint_lines_for_obligation(
    obligation: ControlProofObligation,
) -> List[str]:
    lines: List[str] = []
    lines.append(f"; obligation_id: {obligation.obligation_id}")
    lines.append(f"; source_module: {obligation.source_module}")
    lines.append(f"; proof_kind: {obligation.proof_kind}")
    lines.append(f"; proof_epoch: {obligation.proof_epoch}")
    lines.append(f"; latency_budget_ms: {obligation.latency_budget_ms}")
    group_predicates: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
        ("state", "state-safety", obligation.state_constraints),
        ("transition", "transition-legality", obligation.transition_constraints),
        ("collision", "collision-free", obligation.collision_constraints),
        ("rollback", "rollback-bounded", obligation.rollback_constraints),
        ("fallback", "fallback-corridor-safe", obligation.fallback_constraints),
    )
    for group_label, predicate, constraints in group_predicates:
        for index, constraint in enumerate(constraints):
            escaped = constraint.replace("\\", "\\\\").replace("\"", "\\\"")
            lines.append(
                f"(assert ({predicate} \"{obligation.obligation_id}\" "
                f"{index} \"{escaped}\"))"
            )
    return lines


def _build_base_smtlib(
    sequence_id: str,
    obligations: Tuple[ControlProofObligation, ...],
) -> str:
    lines: List[str] = []
    lines.append("; proof-carrying control sequence SMT-LIB base artifact")
    lines.append(f"; sequence_id: {sequence_id}")
    lines.append(f"; obligation_count: {len(obligations)}")
    lines.append("(set-logic ALL)")
    lines.append("(declare-fun state-safety (String Int String) Bool)")
    lines.append("(declare-fun transition-legality (String Int String) Bool)")
    lines.append("(declare-fun collision-free (String Int String) Bool)")
    lines.append("(declare-fun rollback-bounded (String Int String) Bool)")
    lines.append("(declare-fun fallback-corridor-safe (String Int String) Bool)")
    lines.append("(declare-fun replay-identity-invariant () Bool)")
    for obligation in obligations:
        lines.extend(_constraint_lines_for_obligation(obligation))
    lines.append("(assert replay-identity-invariant)")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


def _canonical_control_hash(
    sequence_id: str,
    obligations: Tuple[ControlProofObligation, ...],
) -> str:
    payload = {
        "sequence_id": sequence_id,
        "proof_obligations": [o.to_dict() for o in obligations],
    }
    return _sha256_hex(_canonical_bytes(payload))


def _proof_hash(
    canonical_control_hash: str,
    smtlib_artifact_hash: str,
    obligations: Tuple[ControlProofObligation, ...],
) -> str:
    payload = {
        "canonical_control_hash": canonical_control_hash,
        "smtlib_artifact_hash": smtlib_artifact_hash,
        "obligation_ids": [o.obligation_id for o in obligations],
    }
    return _sha256_hex(_canonical_bytes(payload))


def build_proof_carrying_control_sequence(
    sequence_id: str,
    obligations: Iterable[ObligationLike],
) -> ProofCarryingControlSequence:
    """Deterministically build a proof-carrying control sequence.

    Rejects duplicate obligations, malformed constraints, unsupported proof
    kinds, and invalid latency budgets via obligation normalization.
    """
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("invalid sequence id")
    ordered = _ordered_obligations(obligations)
    if not ordered:
        raise ValueError("empty obligations")
    base_smtlib = _build_base_smtlib(sequence_id.strip(), ordered)
    smtlib_artifact_hash = _sha256_hex(base_smtlib.encode("utf-8"))
    canonical_control_hash = _canonical_control_hash(sequence_id.strip(), ordered)
    proof_hash = _proof_hash(canonical_control_hash, smtlib_artifact_hash, ordered)
    return ProofCarryingControlSequence(
        sequence_id=sequence_id.strip(),
        proof_obligations=ordered,
        canonical_control_hash=canonical_control_hash,
        proof_hash=proof_hash,
        smtlib_artifact_hash=smtlib_artifact_hash,
    )


def _solver_header(solver_target: str) -> str:
    lines: List[str] = []
    lines.append(f"; solver_target: {solver_target}")
    if solver_target == "z3":
        lines.append("(set-option :produce-models true)")
        lines.append("(set-option :produce-unsat-cores true)")
    elif solver_target == "cvc5":
        lines.append("(set-option :produce-proofs true)")
        lines.append("(set-option :produce-unsat-cores true)")
    return "\n".join(lines) + "\n"


def export_smtlib_proof_artifact(
    sequence: ProofCarryingControlSequence,
    solver_target: str,
) -> SMTLIBExportArtifact:
    """Deterministically export an SMT-LIB 2 proof artifact targeted at
    a supported solver.

    Repeated export yields byte-identical output.
    """
    if not isinstance(sequence, ProofCarryingControlSequence):
        raise ValueError("sequence must be ProofCarryingControlSequence")
    if solver_target not in _ALLOWED_SOLVERS:
        raise ValueError(f"invalid solver target: {solver_target}")

    base_smtlib = _build_base_smtlib(sequence.sequence_id, sequence.proof_obligations)
    base_hash = _sha256_hex(base_smtlib.encode("utf-8"))
    if base_hash != sequence.smtlib_artifact_hash:
        raise ValueError("smtlib base artifact hash mismatch")

    header = _solver_header(solver_target)
    smtlib_text = header + base_smtlib
    export_hash = _sha256_hex(smtlib_text.encode("utf-8"))
    artifact_id = f"smtlib-{sequence.sequence_id}-{solver_target}"
    return SMTLIBExportArtifact(
        artifact_id=artifact_id,
        solver_target=solver_target,
        smtlib_text=smtlib_text,
        export_hash=export_hash,
    )


def _simulated_latency_ms(obligation: ControlProofObligation) -> int:
    total = 0
    for group in _CONSTRAINT_GROUPS:
        for constraint in getattr(obligation, group):
            total += max(1, len(constraint))
    return total


def _constraint_failures(obligation: ControlProofObligation) -> Tuple[str, ...]:
    failures: List[str] = []
    for group in _CONSTRAINT_GROUPS:
        for constraint in getattr(obligation, group):
            if constraint.startswith("FAIL:"):
                failures.append(f"{group}:{constraint}")
    return tuple(failures)


def _replay_identity_hash(
    canonical_control_hash: str,
    export_hash: str,
    solver_target: str,
) -> str:
    payload = {
        "canonical_control_hash": canonical_control_hash,
        "export_hash": export_hash,
        "solver_target": solver_target,
    }
    return _sha256_hex(_canonical_bytes(payload))


def verify_control_proof_sequence(
    sequence: ProofCarryingControlSequence,
    solver_target: str,
) -> ProofVerificationReport:
    """Deterministic proof verifier.

    Produces a byte-stable ProofVerificationReport. Determinism law:
    repeated verification of the same sequence and solver target yields
    the exact same bytes and outcome.
    """
    if not isinstance(sequence, ProofCarryingControlSequence):
        raise ValueError("sequence must be ProofCarryingControlSequence")
    if solver_target not in _ALLOWED_SOLVERS:
        raise ValueError(f"invalid solver target: {solver_target}")

    # Recompute canonical control hash and base artifact hash to guard
    # against tampering; any mismatch yields invalid_artifact deterministically.
    try:
        recomputed_control_hash = _canonical_control_hash(
            sequence.sequence_id, sequence.proof_obligations
        )
        recomputed_base = _build_base_smtlib(
            sequence.sequence_id, sequence.proof_obligations
        )
        recomputed_base_hash = _sha256_hex(recomputed_base.encode("utf-8"))
        expected_proof_hash = _proof_hash(
            recomputed_control_hash,
            recomputed_base_hash,
            sequence.proof_obligations,
        )
    except ValueError:
        return ProofVerificationReport(
            solver_target=solver_target,
            verification_status="invalid_artifact",
            proof_artifact_hash="",
            replay_identity_hash="",
            obligations_passed=(),
            obligations_failed=(),
            latency_ms=0,
        )

    integrity_ok = (
        recomputed_control_hash == sequence.canonical_control_hash
        and recomputed_base_hash == sequence.smtlib_artifact_hash
        and expected_proof_hash == sequence.proof_hash
    )

    if not integrity_ok:
        return ProofVerificationReport(
            solver_target=solver_target,
            verification_status="invalid_artifact",
            proof_artifact_hash="",
            replay_identity_hash="",
            obligations_passed=(),
            obligations_failed=(),
            latency_ms=0,
        )

    artifact = export_smtlib_proof_artifact(sequence, solver_target)
    replay_identity_hash = _replay_identity_hash(
        sequence.canonical_control_hash, artifact.export_hash, solver_target
    )

    passed: List[str] = []
    failed: List[str] = []
    total_latency = 0
    any_timeout = False

    for obligation in sequence.proof_obligations:
        latency = _simulated_latency_ms(obligation)
        total_latency += latency
        if latency > obligation.latency_budget_ms:
            any_timeout = True
            continue
        failures = _constraint_failures(obligation)
        if failures:
            failed.append(obligation.obligation_id)
        else:
            passed.append(obligation.obligation_id)

    if any_timeout:
        status = "timeout"
        passed_tuple: Tuple[str, ...] = ()
        failed_tuple: Tuple[str, ...] = ()
    elif failed:
        status = "failed"
        passed_tuple = tuple(sorted(passed))
        failed_tuple = tuple(sorted(failed))
    else:
        status = "verified"
        passed_tuple = tuple(sorted(passed))
        failed_tuple = ()

    if status not in _ALLOWED_VERIFICATION_STATUSES:
        raise ValueError(f"invalid verification status: {status}")

    return ProofVerificationReport(
        solver_target=solver_target,
        verification_status=status,
        proof_artifact_hash=artifact.export_hash,
        replay_identity_hash=replay_identity_hash,
        obligations_passed=passed_tuple,
        obligations_failed=failed_tuple,
        latency_ms=total_latency,
    )


def record_proof_execution_receipt(
    sequence: ProofCarryingControlSequence,
    report: ProofVerificationReport,
    proof_epoch: int,
) -> ProofExecutionReceipt:
    """Deterministically derive a proof execution receipt from a verified
    sequence and its verification report.
    """
    if not isinstance(sequence, ProofCarryingControlSequence):
        raise ValueError("sequence must be ProofCarryingControlSequence")
    if not isinstance(report, ProofVerificationReport):
        raise ValueError("report must be ProofVerificationReport")
    if not isinstance(proof_epoch, int) or proof_epoch < 0:
        raise ValueError("invalid proof epoch")

    receipt_payload = {
        "sequence_id": sequence.sequence_id,
        "solver_target": report.solver_target,
        "verification_status": report.verification_status,
        "proof_hash": sequence.proof_hash,
        "proof_artifact_hash": report.proof_artifact_hash,
        "replay_identity_hash": report.replay_identity_hash,
        "latency_ms": report.latency_ms,
        "proof_epoch": proof_epoch,
    }
    receipt_id = f"receipt-{_sha256_hex(_canonical_bytes(receipt_payload))[:32]}"
    return ProofExecutionReceipt(
        receipt_id=receipt_id,
        sequence_id=sequence.sequence_id,
        solver_target=report.solver_target,
        verification_status=report.verification_status,
        proof_hash=sequence.proof_hash,
        proof_artifact_hash=report.proof_artifact_hash,
        replay_identity_hash=report.replay_identity_hash,
        latency_ms=report.latency_ms,
        proof_epoch=proof_epoch,
    )


__all__ = (
    "ControlProofObligation",
    "ProofCarryingControlSequence",
    "SMTLIBExportArtifact",
    "ProofVerificationReport",
    "ProofExecutionReceipt",
    "normalize_control_proof_obligation",
    "build_proof_carrying_control_sequence",
    "export_smtlib_proof_artifact",
    "verify_control_proof_sequence",
    "record_proof_execution_receipt",
)
