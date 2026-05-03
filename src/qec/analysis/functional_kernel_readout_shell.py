from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe

FUNCTIONAL_KERNEL_VERSION = "v153.7"
MAX_KERNEL_PARAMETERS = 128
MAX_DERIVATION_PARAMETERS = 128
MAX_READOUT_SHELLS = 128
MAX_SHELL_PARAMETERS = 128

_ALLOWED_KERNEL_TYPES = {
    "READOUT_CORE",
    "ROUTER_READOUT_CORE",
    "LAYERED_READOUT_CORE",
    "MASK_COMPATIBILITY_CORE",
    "SHIFT_COMPATIBILITY_CORE",
    "GENERIC_FUNCTIONAL_CORE",
}
_ALLOWED_DERIVATION_RULES = {
    "IDENTITY_DERIVATION",
    "READOUT_BOUND_DERIVATION",
    "MASK_BOUND_DERIVATION",
    "SHIFT_BOUND_DERIVATION",
    "SHELL_COMPATIBILITY_DERIVATION",
}
_ALLOWED_SHELL_KINDS = {
    "IDENTITY_SHELL",
    "READOUT_BINDING_SHELL",
    "MASK_BINDING_SHELL",
    "SHIFT_BINDING_SHELL",
    "ORDER_BINDING_SHELL",
    "COMPATIBILITY_SHELL",
}
_ALLOWED_COMPATIBILITY_STATUS = {"KERNEL_COMPATIBLE", "KERNEL_INCOMPATIBLE"}
_ALLOWED_COMPATIBILITY_REASON = {
    "DERIVATION_BOUND",
    "UPSTREAM_IDENTITY_MISMATCH",
    "INVALID_UPSTREAM_IDENTITIES",
}


def _is_sha256_hex(v: str) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _freeze_map(params: Mapping[str, Any], max_count: int) -> Mapping[str, Any]:
    if not isinstance(params, Mapping) or len(params) > max_count:
        raise ValueError("INVALID_INPUT")
    frozen = MappingProxyType(dict(_deep_freeze(dict(params))))
    _ensure_json_safe(frozen)
    return frozen


def _thaw(v: Any) -> Any:
    if isinstance(v, MappingProxyType):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


def _composed_hash(
    core_hash: str,
    derived_hash: str,
    stack_hash: str,
    input_hash: str,
    policy: str,
) -> str:
    return sha256_hex({
        "composition_policy": policy,
        "core_kernel_hash": core_hash,
        "derived_kernel_hash": derived_hash,
        "readout_shell_stack_hash": stack_hash,
        "input_identity_hash": input_hash,
    })


@dataclass(frozen=True)
class CoreKernelSpec:
    kernel_id: str
    kernel_version: str
    kernel_type: str
    input_identity_hash: str
    output_contract_hash: str
    kernel_parameters: Mapping[str, Any]
    kernel_hash: str

    def __post_init__(self) -> None:
        if not self.kernel_id:
            raise ValueError("INVALID_INPUT")
        if self.kernel_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.kernel_type not in _ALLOWED_KERNEL_TYPES:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.input_identity_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.output_contract_hash):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "kernel_parameters", _freeze_map(self.kernel_parameters, MAX_KERNEL_PARAMETERS))
        if self.kernel_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "kernel_id": self.kernel_id,
            "kernel_version": self.kernel_version,
            "kernel_type": self.kernel_type,
            "input_identity_hash": self.input_identity_hash,
            "output_contract_hash": self.output_contract_hash,
            "kernel_parameters": _thaw(self.kernel_parameters),
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), kernel_hash=self.kernel_hash)


@dataclass(frozen=True)
class DerivedKernelSpec:
    derived_kernel_id: str
    derived_kernel_version: str
    parent_kernel_hash: str
    derivation_rule: str
    derivation_parameters: Mapping[str, Any]
    derived_input_identity_hash: str
    derived_output_contract_hash: str
    derived_kernel_hash: str

    def __post_init__(self) -> None:
        if not self.derived_kernel_id:
            raise ValueError("INVALID_INPUT")
        if self.derived_kernel_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.derivation_rule not in _ALLOWED_DERIVATION_RULES:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.parent_kernel_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.derived_input_identity_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.derived_output_contract_hash):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "derivation_parameters", _freeze_map(self.derivation_parameters, MAX_DERIVATION_PARAMETERS))
        if self.derived_kernel_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "derived_kernel_id": self.derived_kernel_id,
            "derived_kernel_version": self.derived_kernel_version,
            "parent_kernel_hash": self.parent_kernel_hash,
            "derivation_rule": self.derivation_rule,
            "derivation_parameters": _thaw(self.derivation_parameters),
            "derived_input_identity_hash": self.derived_input_identity_hash,
            "derived_output_contract_hash": self.derived_output_contract_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), derived_kernel_hash=self.derived_kernel_hash)


@dataclass(frozen=True)
class KernelDerivationReceipt:
    derivation_id: str
    kernel_version: str
    core_kernel_hash: str
    derived_kernel_hash: str
    derivation_rule: str
    derivation_parameters_hash: str
    derivation_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.derivation_id:
            raise ValueError("INVALID_INPUT")
        if self.kernel_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.derivation_rule not in _ALLOWED_DERIVATION_RULES:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.core_kernel_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.derived_kernel_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.derivation_parameters_hash):
            raise ValueError("INVALID_INPUT")
        if self.derivation_hash and self.derivation_hash != sha256_hex(self._d()):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "kernel_version": self.kernel_version,
            "core_kernel_hash": self.core_kernel_hash,
            "derived_kernel_hash": self.derived_kernel_hash,
            "derivation_rule": self.derivation_rule,
            "derivation_parameters_hash": self.derivation_parameters_hash,
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), derivation_hash=self.derivation_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class KernelCompatibilityReceipt:
    compatibility_id: str
    kernel_version: str
    core_kernel_hash: str
    derived_kernel_hash: str
    upstream_identity_hashes: tuple[str, ...]
    compatibility_status: str
    compatibility_reason: str
    compatibility_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream_identity_hashes", tuple(self.upstream_identity_hashes))
        if not self.compatibility_id:
            raise ValueError("INVALID_INPUT")
        if self.kernel_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.core_kernel_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.derived_kernel_hash):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.upstream_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if len(set(self.upstream_identity_hashes)) != len(self.upstream_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.upstream_identity_hashes)) != self.upstream_identity_hashes:
            raise ValueError("INVALID_INPUT")
        if self.compatibility_status not in _ALLOWED_COMPATIBILITY_STATUS:
            raise ValueError("INVALID_INPUT")
        if self.compatibility_reason not in _ALLOWED_COMPATIBILITY_REASON:
            raise ValueError("INVALID_INPUT")
        if self.compatibility_hash and self.compatibility_hash != sha256_hex(self._d()):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "compatibility_id": self.compatibility_id,
            "kernel_version": self.kernel_version,
            "core_kernel_hash": self.core_kernel_hash,
            "derived_kernel_hash": self.derived_kernel_hash,
            "upstream_identity_hashes": list(self.upstream_identity_hashes),
            "compatibility_status": self.compatibility_status,
            "compatibility_reason": self.compatibility_reason,
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), compatibility_hash=self.compatibility_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class ReadoutShell:
    shell_id: str
    shell_version: str
    shell_kind: str
    shell_input_hash: str
    shell_output_hash: str
    shell_parameters: Mapping[str, Any]
    shell_hash: str

    def __post_init__(self) -> None:
        if not self.shell_id:
            raise ValueError("INVALID_INPUT")
        if self.shell_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.shell_kind not in _ALLOWED_SHELL_KINDS:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.shell_input_hash):
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.shell_output_hash):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "shell_parameters", _freeze_map(self.shell_parameters, MAX_SHELL_PARAMETERS))
        if self.shell_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "shell_id": self.shell_id,
            "shell_version": self.shell_version,
            "shell_kind": self.shell_kind,
            "shell_input_hash": self.shell_input_hash,
            "shell_output_hash": self.shell_output_hash,
            "shell_parameters": _thaw(self.shell_parameters),
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), shell_hash=self.shell_hash)


@dataclass(frozen=True)
class ReadoutShellStack:
    stack_id: str
    stack_version: str
    ordered_shell_hashes: tuple[str, ...]
    ordered_shell_ids: tuple[str, ...]
    stack_order_hash: str
    stack_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_shell_hashes", tuple(self.ordered_shell_hashes))
        object.__setattr__(self, "ordered_shell_ids", tuple(self.ordered_shell_ids))
        if not self.stack_id:
            raise ValueError("INVALID_INPUT")
        if self.stack_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if not self.ordered_shell_hashes:
            raise ValueError("INVALID_INPUT")
        if len(self.ordered_shell_hashes) != len(self.ordered_shell_ids):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.ordered_shell_hashes):
            raise ValueError("INVALID_INPUT")
        if any(not isinstance(x, str) or not x for x in self.ordered_shell_ids):
            raise ValueError("INVALID_INPUT")
        expected_order_hash = sha256_hex({
            "ordered_shell_hashes": list(self.ordered_shell_hashes),
            "ordered_shell_ids": list(self.ordered_shell_ids),
        })
        if self.stack_order_hash != expected_order_hash:
            raise ValueError("INVALID_INPUT")
        if self.stack_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _order(self) -> dict[str, Any]:
        return {
            "ordered_shell_hashes": list(self.ordered_shell_hashes),
            "ordered_shell_ids": list(self.ordered_shell_ids),
        }

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "stack_id": self.stack_id,
            "stack_version": self.stack_version,
            "ordered_shell_hashes": list(self.ordered_shell_hashes),
            "ordered_shell_ids": list(self.ordered_shell_ids),
            "stack_order_hash": self.stack_order_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), stack_hash=self.stack_hash)


@dataclass(frozen=True)
class ReadoutOrderReceipt:
    order_receipt_id: str
    stack_id: str
    stack_version: str
    stack_hash: str
    ordered_shell_hashes: tuple[str, ...]
    ordered_shell_ids: tuple[str, ...]
    order_policy: str
    order_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_shell_hashes", tuple(self.ordered_shell_hashes))
        object.__setattr__(self, "ordered_shell_ids", tuple(self.ordered_shell_ids))
        if not self.order_receipt_id:
            raise ValueError("INVALID_INPUT")
        if not self.stack_id:
            raise ValueError("INVALID_INPUT")
        if self.stack_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.stack_hash):
            raise ValueError("INVALID_INPUT")
        if len(self.ordered_shell_hashes) != len(self.ordered_shell_ids):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.ordered_shell_hashes):
            raise ValueError("INVALID_INPUT")
        if any(not isinstance(x, str) or not x for x in self.ordered_shell_ids):
            raise ValueError("INVALID_INPUT")
        if self.order_policy != "EXPLICIT_ORDER_PRESERVED":
            raise ValueError("INVALID_INPUT")
        if self.order_hash and self.order_hash != sha256_hex(self._d()):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "order_receipt_id": self.order_receipt_id,
            "stack_id": self.stack_id,
            "stack_version": self.stack_version,
            "stack_hash": self.stack_hash,
            "ordered_shell_hashes": list(self.ordered_shell_hashes),
            "ordered_shell_ids": list(self.ordered_shell_ids),
            "order_policy": self.order_policy,
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), order_hash=self.order_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class ReadoutCompositionReceipt:
    composition_id: str
    kernel_version: str
    core_kernel_hash: str
    derived_kernel_hash: str
    kernel_derivation_receipt_hash: str
    kernel_compatibility_receipt_hash: str
    readout_shell_stack_hash: str
    readout_order_receipt_hash: str
    input_identity_hash: str
    composed_readout_identity_hash: str
    composition_policy: str
    composition_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.composition_id:
            raise ValueError("INVALID_INPUT")
        if self.kernel_version != FUNCTIONAL_KERNEL_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.composition_policy != "ORDERED_SHELL_COMPOSITION_V1":
            raise ValueError("INVALID_INPUT")
        fields = (
            self.core_kernel_hash,
            self.derived_kernel_hash,
            self.kernel_derivation_receipt_hash,
            self.kernel_compatibility_receipt_hash,
            self.readout_shell_stack_hash,
            self.readout_order_receipt_hash,
            self.input_identity_hash,
            self.composed_readout_identity_hash,
        )
        if not all(_is_sha256_hex(x) for x in fields):
            raise ValueError("INVALID_INPUT")
        if self.composition_hash and self.composition_hash != sha256_hex(self._d()):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "composition_id": self.composition_id,
            "kernel_version": self.kernel_version,
            "core_kernel_hash": self.core_kernel_hash,
            "derived_kernel_hash": self.derived_kernel_hash,
            "kernel_derivation_receipt_hash": self.kernel_derivation_receipt_hash,
            "kernel_compatibility_receipt_hash": self.kernel_compatibility_receipt_hash,
            "readout_shell_stack_hash": self.readout_shell_stack_hash,
            "readout_order_receipt_hash": self.readout_order_receipt_hash,
            "input_identity_hash": self.input_identity_hash,
            "composed_readout_identity_hash": self.composed_readout_identity_hash,
            "composition_policy": self.composition_policy,
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), composition_hash=self.composition_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


def build_core_kernel_spec(
    kernel_id: str,
    kernel_type: str,
    input_identity_hash: str,
    output_contract_hash: str,
    kernel_parameters: Mapping[str, Any],
) -> CoreKernelSpec:
    payload = {
        "kernel_id": kernel_id,
        "kernel_version": FUNCTIONAL_KERNEL_VERSION,
        "kernel_type": kernel_type,
        "input_identity_hash": input_identity_hash,
        "output_contract_hash": output_contract_hash,
        "kernel_parameters": _thaw(_freeze_map(kernel_parameters, MAX_KERNEL_PARAMETERS)),
    }
    return CoreKernelSpec(**payload, kernel_hash=sha256_hex(payload))


def build_derived_kernel_spec(
    core: CoreKernelSpec,
    derived_kernel_id: str,
    derivation_rule: str,
    derivation_parameters: Mapping[str, Any],
    derived_input_identity_hash: str,
    derived_output_contract_hash: str,
    parent_kernel_hash: str | None = None,
) -> DerivedKernelSpec:
    pkh = core.stable_hash() if parent_kernel_hash is None else parent_kernel_hash
    if pkh != core.stable_hash():
        raise ValueError("INVALID_INPUT")
    payload = {
        "derived_kernel_id": derived_kernel_id,
        "derived_kernel_version": FUNCTIONAL_KERNEL_VERSION,
        "parent_kernel_hash": pkh,
        "derivation_rule": derivation_rule,
        "derivation_parameters": _thaw(_freeze_map(derivation_parameters, MAX_DERIVATION_PARAMETERS)),
        "derived_input_identity_hash": derived_input_identity_hash,
        "derived_output_contract_hash": derived_output_contract_hash,
    }
    return DerivedKernelSpec(**payload, derived_kernel_hash=sha256_hex(payload))


def build_kernel_derivation_receipt(
    core: CoreKernelSpec,
    derived: DerivedKernelSpec,
    derivation_id: str,
) -> KernelDerivationReceipt:
    if derived.parent_kernel_hash != core.stable_hash():
        raise ValueError("INVALID_INPUT")
    dp = sha256_hex(_thaw(derived.derivation_parameters))
    d = {
        "derivation_id": derivation_id,
        "kernel_version": FUNCTIONAL_KERNEL_VERSION,
        "core_kernel_hash": core.stable_hash(),
        "derived_kernel_hash": derived.stable_hash(),
        "derivation_rule": derived.derivation_rule,
        "derivation_parameters_hash": dp,
    }
    dh = sha256_hex(d)
    c = dict(d, derivation_hash=dh)
    return KernelDerivationReceipt(**c, receipt_hash=sha256_hex(c))


def build_kernel_compatibility_receipt(
    core: CoreKernelSpec,
    derived: DerivedKernelSpec,
    compatibility_id: str,
    upstream_identity_hashes: Sequence[str] = (),
) -> KernelCompatibilityReceipt:
    uh = tuple(upstream_identity_hashes)
    if any(not _is_sha256_hex(x) for x in uh):
        raise ValueError("INVALID_INPUT")
    if len(set(uh)) != len(uh):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(uh)) != uh:
        raise ValueError("INVALID_INPUT")
    if derived.parent_kernel_hash == core.stable_hash():
        status, reason = "KERNEL_COMPATIBLE", "DERIVATION_BOUND"
    else:
        status, reason = "KERNEL_INCOMPATIBLE", "UPSTREAM_IDENTITY_MISMATCH"
    d = {
        "compatibility_id": compatibility_id,
        "kernel_version": FUNCTIONAL_KERNEL_VERSION,
        "core_kernel_hash": core.stable_hash(),
        "derived_kernel_hash": derived.stable_hash(),
        "upstream_identity_hashes": list(uh),
        "compatibility_status": status,
        "compatibility_reason": reason,
    }
    ch = sha256_hex(d)
    c = dict(d, compatibility_hash=ch)
    return KernelCompatibilityReceipt(**{**c, "upstream_identity_hashes": uh, "receipt_hash": sha256_hex(c)})


def validate_kernel_compatibility_receipt(
    core: CoreKernelSpec,
    derived: DerivedKernelSpec,
    receipt: KernelCompatibilityReceipt,
) -> None:
    r = build_kernel_compatibility_receipt(core, derived, receipt.compatibility_id, receipt.upstream_identity_hashes)
    if r.to_dict() != receipt.to_dict():
        raise ValueError("INVALID_INPUT")


def build_readout_shell(
    shell_id: str,
    shell_kind: str,
    shell_input_hash: str,
    shell_output_hash: str,
    shell_parameters: Mapping[str, Any],
) -> ReadoutShell:
    p = {
        "shell_id": shell_id,
        "shell_version": FUNCTIONAL_KERNEL_VERSION,
        "shell_kind": shell_kind,
        "shell_input_hash": shell_input_hash,
        "shell_output_hash": shell_output_hash,
        "shell_parameters": _thaw(_freeze_map(shell_parameters, MAX_SHELL_PARAMETERS)),
    }
    return ReadoutShell(**p, shell_hash=sha256_hex(p))


def build_readout_shell_stack(
    stack_id: str,
    shells: Sequence[ReadoutShell],
) -> ReadoutShellStack:
    if not stack_id or not shells or len(shells) > MAX_READOUT_SHELLS:
        raise ValueError("INVALID_INPUT")
    ids = tuple(s.shell_id for s in shells)
    hashes = tuple(s.stable_hash() for s in shells)
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("INVALID_INPUT")
    op = {"ordered_shell_hashes": list(hashes), "ordered_shell_ids": list(ids)}
    order_hash = sha256_hex(op)
    p = {
        "stack_id": stack_id,
        "stack_version": FUNCTIONAL_KERNEL_VERSION,
        "ordered_shell_hashes": list(hashes),
        "ordered_shell_ids": list(ids),
        "stack_order_hash": order_hash,
    }
    return ReadoutShellStack(**{**p, "ordered_shell_hashes": hashes, "ordered_shell_ids": ids, "stack_hash": sha256_hex(p)})


def build_readout_order_receipt(
    stack: ReadoutShellStack,
    order_receipt_id: str,
) -> ReadoutOrderReceipt:
    d = {
        "order_receipt_id": order_receipt_id,
        "stack_id": stack.stack_id,
        "stack_version": stack.stack_version,
        "stack_hash": stack.stable_hash(),
        "ordered_shell_hashes": list(stack.ordered_shell_hashes),
        "ordered_shell_ids": list(stack.ordered_shell_ids),
        "order_policy": "EXPLICIT_ORDER_PRESERVED",
    }
    oh = sha256_hex(d)
    c = dict(d, order_hash=oh)
    return ReadoutOrderReceipt(**{**c, "ordered_shell_hashes": stack.ordered_shell_hashes, "ordered_shell_ids": stack.ordered_shell_ids, "receipt_hash": sha256_hex(c)})


def validate_readout_order_receipt(
    stack: ReadoutShellStack,
    receipt: ReadoutOrderReceipt,
) -> None:
    ReadoutOrderReceipt(**receipt.to_dict())
    if receipt.order_hash != sha256_hex(receipt._d()):
        raise ValueError("INVALID_INPUT")
    if receipt.receipt_hash != receipt.stable_hash():
        raise ValueError("INVALID_INPUT")
    if receipt.stack_hash != stack.stable_hash():
        raise ValueError("INVALID_INPUT")
    if receipt.stack_id != stack.stack_id:
        raise ValueError("INVALID_INPUT")
    if receipt.stack_version != stack.stack_version:
        raise ValueError("INVALID_INPUT")
    if receipt.ordered_shell_hashes != stack.ordered_shell_hashes:
        raise ValueError("INVALID_INPUT")
    if receipt.ordered_shell_ids != stack.ordered_shell_ids:
        raise ValueError("INVALID_INPUT")
    if receipt.order_policy != "EXPLICIT_ORDER_PRESERVED":
        raise ValueError("INVALID_INPUT")
    r = build_readout_order_receipt(stack, receipt.order_receipt_id)
    if r.to_dict() != receipt.to_dict():
        raise ValueError("INVALID_INPUT")


def build_readout_composition_receipt(
    composition_id: str,
    core: CoreKernelSpec,
    derived: DerivedKernelSpec,
    derivation_receipt: KernelDerivationReceipt,
    compatibility_receipt: KernelCompatibilityReceipt,
    shell_stack: ReadoutShellStack,
    order_receipt: ReadoutOrderReceipt,
    input_identity_hash: str,
    composition_policy: str = "ORDERED_SHELL_COMPOSITION_V1",
) -> ReadoutCompositionReceipt:
    if composition_policy != "ORDERED_SHELL_COMPOSITION_V1":
        raise ValueError("INVALID_INPUT")
    if not _is_sha256_hex(input_identity_hash):
        raise ValueError("INVALID_INPUT")
    validate_readout_order_receipt(shell_stack, order_receipt)
    validate_kernel_compatibility_receipt(core, derived, compatibility_receipt)
    expected_dr = build_kernel_derivation_receipt(core, derived, derivation_receipt.derivation_id)
    if expected_dr.to_dict() != derivation_receipt.to_dict():
        raise ValueError("INVALID_INPUT")
    if compatibility_receipt.compatibility_status != "KERNEL_COMPATIBLE":
        raise ValueError("INVALID_INPUT")
    crih = _composed_hash(
        core.stable_hash(),
        derived.stable_hash(),
        shell_stack.stable_hash(),
        input_identity_hash,
        composition_policy,
    )
    d = {
        "composition_id": composition_id,
        "kernel_version": FUNCTIONAL_KERNEL_VERSION,
        "core_kernel_hash": core.stable_hash(),
        "derived_kernel_hash": derived.stable_hash(),
        "kernel_derivation_receipt_hash": derivation_receipt.stable_hash(),
        "kernel_compatibility_receipt_hash": compatibility_receipt.stable_hash(),
        "readout_shell_stack_hash": shell_stack.stable_hash(),
        "readout_order_receipt_hash": order_receipt.stable_hash(),
        "input_identity_hash": input_identity_hash,
        "composed_readout_identity_hash": crih,
        "composition_policy": composition_policy,
    }
    ch = sha256_hex(d)
    c = dict(d, composition_hash=ch)
    return ReadoutCompositionReceipt(**c, receipt_hash=sha256_hex(c))


def validate_readout_composition_receipt(
    receipt: ReadoutCompositionReceipt,
    core: CoreKernelSpec,
    derived: DerivedKernelSpec,
    derivation_receipt: KernelDerivationReceipt,
    compatibility_receipt: KernelCompatibilityReceipt,
    shell_stack: ReadoutShellStack,
    order_receipt: ReadoutOrderReceipt,
) -> None:
    rebuilt = build_readout_composition_receipt(
        receipt.composition_id,
        core,
        derived,
        derivation_receipt,
        compatibility_receipt,
        shell_stack,
        order_receipt,
        receipt.input_identity_hash,
        receipt.composition_policy,
    )
    if rebuilt.to_dict() != receipt.to_dict():
        raise ValueError("INVALID_INPUT")


def _scope_guard() -> None:
    forbidden = {
        "apply", "execute", "run", "dispatch", "route", "traverse",
        "pathfind", "resolve", "project", "search", "filter", "shift",
        "matrix", "markov", "sample", "random",
    }
    for cls in (
        CoreKernelSpec,
        DerivedKernelSpec,
        KernelDerivationReceipt,
        KernelCompatibilityReceipt,
        ReadoutShell,
        ReadoutShellStack,
        ReadoutOrderReceipt,
        ReadoutCompositionReceipt,
    ):
        for n in forbidden:
            if hasattr(cls, n):
                raise RuntimeError("INVALID_STATE")


_scope_guard()
