from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.functional_kernel_readout_shell import (
    CoreKernelSpec,
    DerivedKernelSpec,
    KernelCompatibilityReceipt,
    KernelDerivationReceipt,
    ReadoutCompositionReceipt,
    ReadoutOrderReceipt,
    ReadoutShellStack,
    build_readout_composition_receipt,
    validate_readout_composition_receipt,
)
from qec.analysis.hilber_shift_projection import (
    FilterCompatibilityReceipt,
    HilberShiftSpec,
    ShiftProjectionReceipt,
    ShiftStabilityReceipt,
    build_shift_stability_receipt,
    validate_shift_stability_receipt,
)
from qec.analysis.layer_spec_contract import _ensure_json_safe
from qec.analysis.readout_combination_matrix import MarkovBasisReceipt, ReadoutCombinationMatrix, ReadoutMatrixReceipt, ReadoutTransitionReceipt, build_markov_basis_receipt, build_readout_matrix_receipt, validate_markov_basis_receipt, validate_readout_matrix_receipt
from qec.analysis.readout_projection_receipts import ReadoutProjectionReceipt, ReadoutProjectionSpec, build_readout_projection_receipt, validate_readout_projection_receipt
from qec.analysis.router_lattice_paths import ResolvedLatticePathSet, RouterLatticePathReceipt, RouterPathSpec, SpecialPathIndex, build_router_lattice_path_receipt, validate_router_lattice_path_receipt
from qec.analysis.search_mask64_contract import MaskCollisionReceipt, MaskCompatibilityReceipt, MaskReductionReceipt, build_mask_compatibility_receipt, validate_mask_compatibility_receipt

LATTICE_REPLAY_VERSION = "v153.9"
REPLAY_RULE = "DETERMINISTIC_REPLAY_ALIGNMENT_V1"
DRIFT_RULE = "EXACT_HASH_MISMATCH_V1"
_ALLOWED_ARTIFACT_TYPES = {"LATTICE_GRAPH", "ROUTER_PATH", "READOUT_PROJECTION", "MASK_COMPATIBILITY", "SHIFT_PROJECTION", "KERNEL_COMPOSITION", "READOUT_MATRIX"}
_STATUS_REASON = {"REPLAY_MATCH": "HASH_MATCH", "REPLAY_DRIFT": "HASH_MISMATCH", "REPLAY_MISSING": "MISSING_RECEIPT", "REPLAY_INCONSISTENT": "INCONSISTENT_CHAIN"}


def _is_sha(v: str) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _status(expected_missing: bool, expected_inconsistent: bool, expected_hash: str, recomputed_hash: str) -> str:
    if expected_missing:
        return "REPLAY_MISSING"
    if expected_inconsistent:
        return "REPLAY_INCONSISTENT"
    return "REPLAY_MATCH" if expected_hash == recomputed_hash else "REPLAY_DRIFT"


def _proof_status(statuses: Sequence[str]) -> str:
    if any(s == "REPLAY_INCONSISTENT" for s in statuses):
        return "REPLAY_INCONSISTENT"
    if any(s == "REPLAY_MISSING" for s in statuses):
        return "REPLAY_INCOMPLETE"
    if any(s == "REPLAY_DRIFT" for s in statuses):
        return "REPLAY_DRIFT_DETECTED"
    return "REPLAY_ALIGNED"


@dataclass(frozen=True)
class LatticeDriftReceipt:
    drift_id: str; replay_version: str; artifact_type: str; artifact_id: str; expected_hash: str; recomputed_hash: str; source_receipt_hash: str; replay_status: str; drift_reason: str; drift_detected: bool; drift_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {"drift_id": self.drift_id, "replay_version": self.replay_version, "artifact_type": self.artifact_type, "artifact_id": self.artifact_id, "expected_hash": self.expected_hash, "recomputed_hash": self.recomputed_hash, "source_receipt_hash": self.source_receipt_hash, "replay_status": self.replay_status, "drift_reason": self.drift_reason, "drift_detected": self.drift_detected}
    def _c(self) -> dict[str, Any]: p = {**self._d(), "drift_hash": self.drift_hash}; _ensure_json_safe(p); return p
    def stable_hash(self) -> str: return sha256_hex(self._c())
    def to_dict(self) -> dict[str, Any]: return dict(self._c(), receipt_hash=self.receipt_hash)
    def __post_init__(self) -> None:
        if not self.drift_id or self.replay_version != LATTICE_REPLAY_VERSION or self.artifact_type not in _ALLOWED_ARTIFACT_TYPES or not self.artifact_id: raise ValueError("INVALID_INPUT")
        if self.replay_status not in _STATUS_REASON or self.drift_reason != _STATUS_REASON[self.replay_status]: raise ValueError("INVALID_INPUT")
        if self.replay_status == "REPLAY_MISSING":
            if self.expected_hash or self.recomputed_hash or self.source_receipt_hash: raise ValueError("INVALID_INPUT")
        else:
            if not _is_sha(self.expected_hash) or not _is_sha(self.recomputed_hash) or not _is_sha(self.source_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.drift_detected != (self.replay_status != "REPLAY_MATCH"): raise ValueError("INVALID_INPUT")
        if self.replay_status == "REPLAY_MATCH" and self.expected_hash != self.recomputed_hash: raise ValueError("INVALID_INPUT")
        if self.replay_status == "REPLAY_DRIFT" and self.expected_hash == self.recomputed_hash: raise ValueError("INVALID_INPUT")
        if self.drift_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")


def _build_drift(drift_id: str, artifact_type: str, artifact_id: str, expected_hash: str, recomputed_hash: str, source_receipt_hash: str, expected_missing: bool, expected_inconsistent: bool) -> LatticeDriftReceipt:
    status = _status(expected_missing, expected_inconsistent, expected_hash, recomputed_hash)
    eh = "" if status == "REPLAY_MISSING" else expected_hash
    rh = "" if status == "REPLAY_MISSING" else recomputed_hash
    sh = "" if status == "REPLAY_MISSING" else source_receipt_hash
    d = {"drift_id": drift_id, "replay_version": LATTICE_REPLAY_VERSION, "artifact_type": artifact_type, "artifact_id": artifact_id, "expected_hash": eh, "recomputed_hash": rh, "source_receipt_hash": sh, "replay_status": status, "drift_reason": _STATUS_REASON[status], "drift_detected": status != "REPLAY_MATCH"}
    dh = sha256_hex(d)
    return LatticeDriftReceipt(**d, drift_hash=dh, receipt_hash=sha256_hex({**d, "drift_hash": dh}))


def build_lattice_drift_receipt(drift_id: str, artifact_type: str, artifact_id: str, expected_hash: str, recomputed_hash: str, source_receipt_hash: str, *, missing: bool = False, inconsistent: bool = False) -> LatticeDriftReceipt:
    return _build_drift(drift_id, artifact_type, artifact_id, expected_hash, recomputed_hash, source_receipt_hash, missing, inconsistent)


# explicit subsystem dataclasses
@dataclass(frozen=True)
class RouterReplayReceipt:
    replay_id: str; replay_version: str; semantic_lattice_graph_hash: str; router_path_spec_hash: str; special_path_index_hash: str; resolved_path_hash: str; expected_router_receipt_hash: str; recomputed_router_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return {"replay_id": self.replay_id, "replay_version": self.replay_version, "semantic_lattice_graph_hash": self.semantic_lattice_graph_hash, "router_path_spec_hash": self.router_path_spec_hash, "special_path_index_hash": self.special_path_index_hash, "resolved_path_hash": self.resolved_path_hash, "expected_router_receipt_hash": self.expected_router_receipt_hash, "recomputed_router_receipt_hash": self.recomputed_router_receipt_hash, "drift_receipt_hash": self.drift_receipt_hash, "replay_status": self.replay_status, "replay_hash": self.replay_hash, "receipt_hash": self.receipt_hash}
    def __post_init__(self) -> None:
        if self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.expected_router_receipt_hash and not _is_sha(self.expected_router_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.recomputed_router_receipt_hash and not _is_sha(self.recomputed_router_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")

@dataclass(frozen=True)
class ReadoutReplayReceipt:
    replay_id: str; replay_version: str; semantic_lattice_graph_hash: str; router_path_spec_hash: str; special_path_index_hash: str; resolved_path_hash: str; readout_projection_spec_hash: str; expected_readout_receipt_hash: str; recomputed_readout_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def __post_init__(self) -> None:
        if self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON or not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.expected_readout_receipt_hash and not _is_sha(self.expected_readout_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.recomputed_readout_receipt_hash and not _is_sha(self.recomputed_readout_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")

@dataclass(frozen=True)
class MaskReplayReceipt:
    replay_id: str; replay_version: str; mask_compatibility_receipt_hash: str; mask_reduction_receipt_hashes: tuple[str, ...]; mask_collision_receipt_hashes: tuple[str, ...]; expected_mask_compatibility_receipt_hash: str; recomputed_mask_compatibility_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__, mask_reduction_receipt_hashes=list(self.mask_reduction_receipt_hashes), mask_collision_receipt_hashes=list(self.mask_collision_receipt_hashes))
    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_reduction_receipt_hashes", tuple(self.mask_reduction_receipt_hashes)); object.__setattr__(self, "mask_collision_receipt_hashes", tuple(self.mask_collision_receipt_hashes))
        if not self.replay_id or self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.mask_reduction_receipt_hashes != tuple(sorted(self.mask_reduction_receipt_hashes)) or len(set(self.mask_reduction_receipt_hashes)) != len(self.mask_reduction_receipt_hashes): raise ValueError("INVALID_INPUT")
        if self.mask_collision_receipt_hashes != tuple(sorted(self.mask_collision_receipt_hashes)) or len(set(self.mask_collision_receipt_hashes)) != len(self.mask_collision_receipt_hashes): raise ValueError("INVALID_INPUT")
        if any(not _is_sha(x) for x in self.mask_reduction_receipt_hashes + self.mask_collision_receipt_hashes): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")

@dataclass(frozen=True)
class ShiftReplayReceipt:
    replay_id: str; replay_version: str; shift_spec_hash: str; shift_projection_receipt_hash: str; filter_compatibility_receipt_hash: str; expected_shift_stability_receipt_hash: str; recomputed_shift_stability_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def __post_init__(self) -> None:
        if self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.expected_shift_stability_receipt_hash and not _is_sha(self.expected_shift_stability_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.recomputed_shift_stability_receipt_hash and not _is_sha(self.recomputed_shift_stability_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")

@dataclass(frozen=True)
class KernelReplayReceipt:
    replay_id: str; replay_version: str; core_kernel_hash: str; derived_kernel_hash: str; kernel_derivation_receipt_hash: str; kernel_compatibility_receipt_hash: str; readout_shell_stack_hash: str; readout_order_receipt_hash: str; expected_readout_composition_receipt_hash: str; recomputed_readout_composition_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def __post_init__(self) -> None:
        if not self.replay_id or self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.expected_readout_composition_receipt_hash and not _is_sha(self.expected_readout_composition_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.recomputed_readout_composition_receipt_hash and not _is_sha(self.recomputed_readout_composition_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")

@dataclass(frozen=True)
class ReadoutMatrixReplayReceipt:
    replay_id: str; replay_version: str; readout_combination_matrix_hash: str; readout_matrix_receipt_hash: str; markov_basis_receipt_hash: str; transition_receipt_hashes: tuple[str, ...]; expected_markov_basis_receipt_hash: str; recomputed_markov_basis_receipt_hash: str; drift_receipt_hash: str; replay_status: str; replay_hash: str; receipt_hash: str
    def _d(self) -> dict[str, Any]: return {k: v for k, v in self.to_dict().items() if k not in {"replay_hash", "receipt_hash"}}
    def stable_hash(self) -> str: return sha256_hex({**self._d(), "replay_hash": self.replay_hash})
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__, transition_receipt_hashes=list(self.transition_receipt_hashes))
    def __post_init__(self) -> None:
        object.__setattr__(self, "transition_receipt_hashes", tuple(self.transition_receipt_hashes))
        if self.replay_version != LATTICE_REPLAY_VERSION or self.replay_status not in _STATUS_REASON: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.drift_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.expected_markov_basis_receipt_hash and not _is_sha(self.expected_markov_basis_receipt_hash): raise ValueError("INVALID_INPUT")
        if self.recomputed_markov_basis_receipt_hash and not _is_sha(self.recomputed_markov_basis_receipt_hash): raise ValueError("INVALID_INPUT")
        if any(not _is_sha(x) for x in self.transition_receipt_hashes): raise ValueError("INVALID_INPUT")
        if self.replay_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")


def build_router_replay_receipt(replay_id: str, graph: SemanticLatticeGraph, router_path_spec: RouterPathSpec, special_path_index: SpecialPathIndex, resolved_path_set: ResolvedLatticePathSet, expected_router_receipt: RouterLatticePathReceipt | None) -> RouterReplayReceipt:
    recomputed = build_router_lattice_path_receipt(replay_id, graph, router_path_spec, special_path_index, resolved_path_set)
    missing = expected_router_receipt is None
    inconsistent = False
    expected_hash = ""
    if not missing:
        expected_hash = expected_router_receipt.receipt_hash
        try: validate_router_lattice_path_receipt(expected_router_receipt, graph, router_path_spec, special_path_index, resolved_path_set)
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "ROUTER_PATH", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    base = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "semantic_lattice_graph_hash": graph.graph_hash, "router_path_spec_hash": router_path_spec.spec_hash, "special_path_index_hash": special_path_index.index_hash, "resolved_path_hash": resolved_path_set.resolved_path_hash, "expected_router_receipt_hash": expected_hash, "recomputed_router_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex(base)
    return RouterReplayReceipt(**base, replay_hash=rh, receipt_hash=sha256_hex({**base, "replay_hash": rh}))

# explicit distinct builders with subsystem signatures

def build_readout_replay_receipt(replay_id: str, graph: SemanticLatticeGraph, router_path_spec: RouterPathSpec, special_path_index: SpecialPathIndex, resolved_path_set: ResolvedLatticePathSet, readout_projection_spec: ReadoutProjectionSpec, expected_readout_receipt: ReadoutProjectionReceipt | None) -> ReadoutReplayReceipt:
    recomputed = build_readout_projection_receipt(replay_id, graph, router_path_spec, special_path_index, resolved_path_set, readout_projection_spec)
    missing = expected_readout_receipt is None; inconsistent = False; expected_hash = "" if missing else expected_readout_receipt.receipt_hash
    if not missing:
        try: validate_readout_projection_receipt(expected_readout_receipt, graph, router_path_spec, special_path_index, resolved_path_set, readout_projection_spec)
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "READOUT_PROJECTION", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    b = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "semantic_lattice_graph_hash": graph.graph_hash, "router_path_spec_hash": router_path_spec.spec_hash, "special_path_index_hash": special_path_index.index_hash, "resolved_path_hash": resolved_path_set.resolved_path_hash, "readout_projection_spec_hash": readout_projection_spec.spec_hash, "expected_readout_receipt_hash": expected_hash, "recomputed_readout_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex(b); return ReadoutReplayReceipt(**b, replay_hash=rh, receipt_hash=sha256_hex({**b, "replay_hash": rh}))

def build_mask_replay_receipt(replay_id: str, expected_mask_compatibility_receipt: MaskCompatibilityReceipt | None, mask_reduction_receipts: Sequence[MaskReductionReceipt], mask_collision_receipts: Sequence[MaskCollisionReceipt]) -> MaskReplayReceipt:
    recomputed = build_mask_compatibility_receipt(replay_id, tuple(mask_reduction_receipts), tuple(mask_collision_receipts))
    missing = expected_mask_compatibility_receipt is None; inconsistent = False; expected_hash = "" if missing else expected_mask_compatibility_receipt.receipt_hash
    if not missing:
        try: validate_mask_compatibility_receipt(expected_mask_compatibility_receipt, tuple(mask_reduction_receipts), tuple(mask_collision_receipts))
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "MASK_COMPATIBILITY", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    red_hashes = tuple(sorted(r.receipt_hash for r in mask_reduction_receipts)); col_hashes = tuple(sorted(c.receipt_hash for c in mask_collision_receipts))
    b = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "mask_compatibility_receipt_hash": recomputed.receipt_hash, "mask_reduction_receipt_hashes": red_hashes, "mask_collision_receipt_hashes": col_hashes, "expected_mask_compatibility_receipt_hash": expected_hash, "recomputed_mask_compatibility_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex({**b, "mask_reduction_receipt_hashes": list(red_hashes), "mask_collision_receipt_hashes": list(col_hashes)}); return MaskReplayReceipt(**b, replay_hash=rh, receipt_hash=sha256_hex({**b, "mask_reduction_receipt_hashes": list(red_hashes), "mask_collision_receipt_hashes": list(col_hashes), "replay_hash": rh}))

def build_shift_replay_receipt(replay_id: str, shift_spec: HilberShiftSpec, shift_projection_receipt: ShiftProjectionReceipt, filter_compatibility_receipt: FilterCompatibilityReceipt, expected_shift_stability_receipt: ShiftStabilityReceipt | None) -> ShiftReplayReceipt:
    recomputed = build_shift_stability_receipt(replay_id, shift_spec, shift_projection_receipt, filter_compatibility_receipt)
    missing = expected_shift_stability_receipt is None; inconsistent = False; expected_hash = "" if missing else expected_shift_stability_receipt.receipt_hash
    if not missing:
        try: validate_shift_stability_receipt(expected_shift_stability_receipt, shift_spec, shift_projection_receipt, filter_compatibility_receipt)
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "SHIFT_PROJECTION", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    b = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "shift_spec_hash": shift_spec.spec_hash, "shift_projection_receipt_hash": shift_projection_receipt.receipt_hash, "filter_compatibility_receipt_hash": filter_compatibility_receipt.receipt_hash, "expected_shift_stability_receipt_hash": expected_hash, "recomputed_shift_stability_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex(b); return ShiftReplayReceipt(**b, replay_hash=rh, receipt_hash=sha256_hex({**b, "replay_hash": rh}))

def build_kernel_replay_receipt(replay_id: str, core_kernel_spec: CoreKernelSpec, derived_kernel_spec: DerivedKernelSpec, kernel_derivation_receipt: KernelDerivationReceipt, kernel_compatibility_receipt: KernelCompatibilityReceipt, readout_shell_stack: ReadoutShellStack, readout_order_receipt: ReadoutOrderReceipt, expected_readout_composition_receipt: ReadoutCompositionReceipt | None, input_identity_hash: str) -> KernelReplayReceipt:
    recomputed = build_readout_composition_receipt(replay_id, core_kernel_spec, derived_kernel_spec, kernel_derivation_receipt, kernel_compatibility_receipt, readout_shell_stack, readout_order_receipt, input_identity_hash)
    missing = expected_readout_composition_receipt is None; inconsistent = False; expected_hash = "" if missing else expected_readout_composition_receipt.receipt_hash
    if not missing:
        try: validate_readout_composition_receipt(expected_readout_composition_receipt, core_kernel_spec, derived_kernel_spec, kernel_derivation_receipt, kernel_compatibility_receipt, readout_shell_stack, readout_order_receipt, input_identity_hash)
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "KERNEL_COMPOSITION", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    b = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "core_kernel_hash": core_kernel_spec.kernel_hash, "derived_kernel_hash": derived_kernel_spec.derived_kernel_hash, "kernel_derivation_receipt_hash": kernel_derivation_receipt.receipt_hash, "kernel_compatibility_receipt_hash": kernel_compatibility_receipt.receipt_hash, "readout_shell_stack_hash": readout_shell_stack.stack_hash, "readout_order_receipt_hash": readout_order_receipt.receipt_hash, "expected_readout_composition_receipt_hash": expected_hash, "recomputed_readout_composition_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex(b); return KernelReplayReceipt(**b, replay_hash=rh, receipt_hash=sha256_hex({**b, "replay_hash": rh}))

def build_readout_matrix_replay_receipt(replay_id: str, readout_combination_matrix: ReadoutCombinationMatrix, readout_matrix_receipt: ReadoutMatrixReceipt, expected_markov_basis_receipt: MarkovBasisReceipt | None, transition_receipts: Sequence[ReadoutTransitionReceipt] = ()) -> ReadoutMatrixReplayReceipt:
    validate_readout_matrix_receipt(readout_matrix_receipt, readout_combination_matrix)
    recomputed = build_markov_basis_receipt(replay_id, readout_combination_matrix, readout_matrix_receipt)
    missing = expected_markov_basis_receipt is None; inconsistent = False; expected_hash = "" if missing else expected_markov_basis_receipt.receipt_hash
    if not missing:
        try: validate_markov_basis_receipt(expected_markov_basis_receipt, readout_combination_matrix, readout_matrix_receipt)
        except ValueError: inconsistent = True
    drift = _build_drift(replay_id, "READOUT_MATRIX", replay_id, expected_hash, recomputed.receipt_hash, expected_hash, missing, inconsistent)
    th = tuple(t.receipt_hash for t in transition_receipts)
    b = {"replay_id": replay_id, "replay_version": LATTICE_REPLAY_VERSION, "readout_combination_matrix_hash": readout_combination_matrix.matrix_hash, "readout_matrix_receipt_hash": readout_matrix_receipt.receipt_hash, "markov_basis_receipt_hash": recomputed.receipt_hash, "transition_receipt_hashes": th, "expected_markov_basis_receipt_hash": expected_hash, "recomputed_markov_basis_receipt_hash": recomputed.receipt_hash, "drift_receipt_hash": drift.receipt_hash, "replay_status": drift.replay_status}
    rh = sha256_hex({**b, "transition_receipt_hashes": list(th)}); return ReadoutMatrixReplayReceipt(**b, replay_hash=rh, receipt_hash=sha256_hex({**b, "transition_receipt_hashes": list(th), "replay_hash": rh}))

@dataclass(frozen=True)
class LatticeReplayProofReceipt:
    proof_id: str; replay_version: str; replay_rule: str; drift_rule: str; semantic_lattice_graph_hash: str; router_replay_receipt_hash: str; readout_replay_receipt_hash: str; mask_replay_receipt_hash: str; shift_replay_receipt_hash: str; kernel_replay_receipt_hash: str; readout_matrix_replay_receipt_hash: str; drift_receipt_hashes: tuple[str, ...]; replay_receipt_hashes: tuple[str, ...]; replay_receipt_count: int; drift_receipt_count: int; replay_alignment_status: str; lattice_replay_proof_hash: str; receipt_hash: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__, drift_receipt_hashes=list(self.drift_receipt_hashes), replay_receipt_hashes=list(self.replay_receipt_hashes))
    def __post_init__(self) -> None:
        object.__setattr__(self, "drift_receipt_hashes", tuple(self.drift_receipt_hashes)); object.__setattr__(self, "replay_receipt_hashes", tuple(self.replay_receipt_hashes))
        if isinstance(self.replay_receipt_count, bool) or isinstance(self.drift_receipt_count, bool): raise ValueError("INVALID_INPUT")
        if self.replay_receipt_count != len(self.replay_receipt_hashes) or self.drift_receipt_count != len(self.drift_receipt_hashes): raise ValueError("INVALID_INPUT")
        if self.replay_version != LATTICE_REPLAY_VERSION or self.replay_rule != REPLAY_RULE or self.drift_rule != DRIFT_RULE: raise ValueError("INVALID_INPUT")
        if not _is_sha(self.semantic_lattice_graph_hash): raise ValueError("INVALID_INPUT")
        for h in (self.router_replay_receipt_hash, self.readout_replay_receipt_hash, self.mask_replay_receipt_hash, self.shift_replay_receipt_hash, self.kernel_replay_receipt_hash, self.readout_matrix_replay_receipt_hash):
            if not _is_sha(h): raise ValueError("INVALID_INPUT")
        if any(not _is_sha(h) for h in self.drift_receipt_hashes + self.replay_receipt_hashes): raise ValueError("INVALID_INPUT")
        payload = {k: v for k, v in self.to_dict().items() if k not in {"lattice_replay_proof_hash", "receipt_hash"}}
        if self.lattice_replay_proof_hash != sha256_hex(payload): raise ValueError("INVALID_INPUT")
        if self.receipt_hash != sha256_hex({**payload, "lattice_replay_proof_hash": self.lattice_replay_proof_hash}): raise ValueError("INVALID_INPUT")


def build_lattice_replay_proof_receipt(proof_id: str, semantic_lattice_graph_hash: str, router: RouterReplayReceipt, readout: ReadoutReplayReceipt, mask: MaskReplayReceipt, shift: ShiftReplayReceipt, kernel: KernelReplayReceipt, readout_matrix: ReadoutMatrixReplayReceipt) -> LatticeReplayProofReceipt:
    statuses = (router.replay_status, readout.replay_status, mask.replay_status, shift.replay_status, kernel.replay_status, readout_matrix.replay_status)
    replay_hashes = (router.receipt_hash, readout.receipt_hash, mask.receipt_hash, shift.receipt_hash, kernel.receipt_hash, readout_matrix.receipt_hash)
    drift_hashes = (router.drift_receipt_hash, readout.drift_receipt_hash, mask.drift_receipt_hash, shift.drift_receipt_hash, kernel.drift_receipt_hash, readout_matrix.drift_receipt_hash)
    b = {"proof_id": proof_id, "replay_version": LATTICE_REPLAY_VERSION, "replay_rule": REPLAY_RULE, "drift_rule": DRIFT_RULE, "semantic_lattice_graph_hash": semantic_lattice_graph_hash, "router_replay_receipt_hash": router.receipt_hash, "readout_replay_receipt_hash": readout.receipt_hash, "mask_replay_receipt_hash": mask.receipt_hash, "shift_replay_receipt_hash": shift.receipt_hash, "kernel_replay_receipt_hash": kernel.receipt_hash, "readout_matrix_replay_receipt_hash": readout_matrix.receipt_hash, "drift_receipt_hashes": drift_hashes, "replay_receipt_hashes": replay_hashes, "replay_receipt_count": 6, "drift_receipt_count": 6, "replay_alignment_status": _proof_status(statuses)}
    ph = sha256_hex({**b, "drift_receipt_hashes": list(drift_hashes), "replay_receipt_hashes": list(replay_hashes)})
    rh = sha256_hex({**b, "drift_receipt_hashes": list(drift_hashes), "replay_receipt_hashes": list(replay_hashes), "lattice_replay_proof_hash": ph})
    return LatticeReplayProofReceipt(**b, lattice_replay_proof_hash=ph, receipt_hash=rh)


def validate_lattice_replay_proof_receipt(receipt: LatticeReplayProofReceipt, router: RouterReplayReceipt | None, readout: ReadoutReplayReceipt | None, mask: MaskReplayReceipt | None, shift: ShiftReplayReceipt | None, kernel: KernelReplayReceipt | None, readout_matrix: ReadoutMatrixReplayReceipt | None) -> None:
    if any(x is None for x in (router, readout, mask, shift, kernel, readout_matrix)): raise ValueError("INCOMPLETE_REPLAY")
    expected = build_lattice_replay_proof_receipt(receipt.proof_id, receipt.semantic_lattice_graph_hash, router, readout, mask, shift, kernel, readout_matrix)
    if expected.to_dict() != receipt.to_dict(): raise ValueError("HASH_MISMATCH")


def _assert_no_v153_9_forbidden_scope() -> None:
    forbidden = ("apply", "execute", "run", "dispatch", "route", "traverse", "pathfind", "resolve", "project", "search", "filter", "shift", "sample", "random", "perturb", "decay", "scale")
    for cls in (LatticeDriftReceipt, RouterReplayReceipt, ReadoutReplayReceipt, MaskReplayReceipt, ShiftReplayReceipt, KernelReplayReceipt, ReadoutMatrixReplayReceipt, LatticeReplayProofReceipt):
        for name in forbidden:
            if hasattr(cls, name):
                raise RuntimeError("INVALID_STATE")


__all__ = [
    "LatticeDriftReceipt", "RouterReplayReceipt", "ReadoutReplayReceipt", "MaskReplayReceipt", "ShiftReplayReceipt", "KernelReplayReceipt", "ReadoutMatrixReplayReceipt", "LatticeReplayProofReceipt",
    "build_lattice_drift_receipt", "build_router_replay_receipt", "build_readout_replay_receipt", "build_mask_replay_receipt", "build_shift_replay_receipt", "build_kernel_replay_receipt", "build_readout_matrix_replay_receipt", "build_lattice_replay_proof_receipt", "validate_lattice_replay_proof_receipt",
]

_assert_no_v153_9_forbidden_scope()
