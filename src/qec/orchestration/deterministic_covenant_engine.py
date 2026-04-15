"""v137.18.1 — Deterministic Covenant Engine.

Minimal deterministic covenant runtime for bounded, replay-safe state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple


Scalar = str | int | float | bool | None
StateValue = Scalar | Tuple["StateValue", ...] | Mapping[str, "StateValue"]


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _normalize_value(value: Any) -> StateValue:
    if _is_scalar(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("state values must be finite floats")
        return value
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, Mapping):
        normalized: Dict[str, StateValue] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            if not isinstance(key, str):
                raise ValueError("state dict keys must be strings")
            if key == "":
                raise ValueError("state dict keys must be non-empty")
            normalized[key] = _normalize_value(value[key])
        return normalized
    raise ValueError("state values must be scalar, list, tuple, or dict")


def _normalize_state(state: Mapping[str, Any]) -> Dict[str, StateValue]:
    if not isinstance(state, Mapping):
        raise ValueError("state must be a mapping")

    validated_keys: list[str] = []
    for key in state.keys():
        if not isinstance(key, str):
            raise ValueError("state keys must be strings")
        if key == "":
            raise ValueError("state keys must be non-empty")
        validated_keys.append(key)

    normalized: Dict[str, StateValue] = {}
    for key in sorted(validated_keys):
        normalized[key] = _normalize_value(state[key])
    return normalized


def _state_hash(state_data: Mapping[str, StateValue]) -> str:
    return _sha256_hex(_canonical_json(state_data).encode("utf-8"))


def _transition_id(prior_state_hash: str, rule_id: str, replay_identity: str) -> str:
    payload = _canonical_json(
        {
            "prior_state_hash": prior_state_hash,
            "replay_identity": replay_identity,
            "rule_id": rule_id,
        }
    ).encode("utf-8")
    return _sha256_hex(payload)


@dataclass(frozen=True)
class CovenantRule:
    rule_id: str
    target_key: str
    delta: float
    min_value: float | None
    max_value: float | None
    replay_identity: str
    invariant_keys: Tuple[str, ...]
    precondition_keys: Tuple[str, ...]
    postcondition_keys: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "target_key": self.target_key,
            "delta": self.delta,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "replay_identity": self.replay_identity,
            "invariant_keys": list(self.invariant_keys),
            "precondition_keys": list(self.precondition_keys),
            "postcondition_keys": list(self.postcondition_keys),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class CovenantState:
    state_data: Mapping[str, StateValue]
    state_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_data": self.state_data,
            "state_hash": self.state_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.state_hash


@dataclass(frozen=True)
class CovenantTransitionReceipt:
    transition_id: str
    prior_state_hash: str
    next_state_hash: str
    rule_id: str
    replay_identity: str
    accepted: bool
    violations: Tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "prior_state_hash": self.prior_state_hash,
            "next_state_hash": self.next_state_hash,
            "rule_id": self.rule_id,
            "replay_identity": self.replay_identity,
            "accepted": self.accepted,
            "violations": list(self.violations),
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class DeterministicCovenantExecution:
    prior_state: CovenantState
    next_state: CovenantState
    receipt: CovenantTransitionReceipt
    rule: CovenantRule

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_state": self.prior_state.to_dict(),
            "next_state": self.next_state.to_dict(),
            "receipt": self.receipt.to_dict(),
            "rule": self.rule.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


def _normalize_keys(keys: Tuple[str, ...] | list[str]) -> Tuple[str, ...]:
    normalized = []
    for key in keys:
        if not isinstance(key, str):
            raise ValueError("rule keys must be strings")
        key_text = key.strip()
        if not key_text:
            raise ValueError("rule keys must be non-empty strings")
        normalized.append(key_text)
    return tuple(sorted(set(normalized)))


def build_covenant_rule(
    *,
    rule_id: str,
    target_key: str,
    delta: float,
    min_value: float | None = None,
    max_value: float | None = None,
    replay_identity: str,
    invariant_keys: Tuple[str, ...] | list[str] = (),
    precondition_keys: Tuple[str, ...] | list[str] = (),
    postcondition_keys: Tuple[str, ...] | list[str] = (),
) -> CovenantRule:
    if not str(rule_id).strip():
        raise ValueError("rule_id must be non-empty")
    if not str(target_key).strip():
        raise ValueError("target_key must be non-empty")
    if not str(replay_identity).strip():
        raise ValueError("replay_identity must be non-empty")
    delta_f = float(delta)
    if not math.isfinite(delta_f):
        raise ValueError("delta must be finite")
    if min_value is not None:
        if not math.isfinite(float(min_value)):
            raise ValueError("min_value must be finite")
    if max_value is not None:
        if not math.isfinite(float(max_value)):
            raise ValueError("max_value must be finite")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value cannot exceed max_value")

    return CovenantRule(
        rule_id=str(rule_id).strip(),
        target_key=str(target_key).strip(),
        delta=delta_f,
        min_value=min_value,
        max_value=max_value,
        replay_identity=str(replay_identity).strip(),
        invariant_keys=_normalize_keys(tuple(invariant_keys)),
        precondition_keys=_normalize_keys(tuple(precondition_keys)),
        postcondition_keys=_normalize_keys(tuple(postcondition_keys)),
    )


def validate_covenant_state(state: Any) -> Tuple[str, ...]:
    violations = []
    try:
        _normalize_state(state)
    except ValueError as exc:
        violations.append(f"malformed_state:{exc}")
    except Exception as exc:  # pragma: no cover - defensive, required non-raising behavior
        violations.append(f"malformed_state:{type(exc).__name__}")
    return tuple(violations)


def execute_covenant_transition(
    state_t: Mapping[str, Any],
    action_capsule: Mapping[str, Any],
    covenant_rule: CovenantRule,
) -> DeterministicCovenantExecution:
    violations = list(validate_covenant_state(state_t))

    normalized_prior: Dict[str, StateValue] = {}
    if not violations:
        normalized_prior = _normalize_state(state_t)

    # Normalize replay_identity from action_capsule consistently in both paths.
    _raw_ri = str(action_capsule.get("replay_identity", "") or "").strip()
    replay_identity = _raw_ri if _raw_ri else covenant_rule.replay_identity

    if violations:
        prior = CovenantState(state_data=normalized_prior, state_hash=_state_hash(normalized_prior))
        transition_id = _transition_id(prior.state_hash, covenant_rule.rule_id, replay_identity)
        receipt = CovenantTransitionReceipt(
            transition_id=transition_id,
            prior_state_hash=prior.state_hash,
            next_state_hash=prior.state_hash,
            rule_id=covenant_rule.rule_id,
            replay_identity=replay_identity,
            accepted=False,
            violations=tuple(violations),
            receipt_hash=_sha256_hex(
                _canonical_json(
                    {
                        "accepted": False,
                        "next_state_hash": prior.state_hash,
                        "prior_state_hash": prior.state_hash,
                        "replay_identity": replay_identity,
                        "rule_id": covenant_rule.rule_id,
                        "transition_id": transition_id,
                        "violations": tuple(violations),
                    }
                ).encode("utf-8")
            ),
        )
        return DeterministicCovenantExecution(
            prior_state=prior,
            next_state=prior,
            receipt=receipt,
            rule=covenant_rule,
        )

    prior = CovenantState(state_data=normalized_prior, state_hash=_state_hash(normalized_prior))
    transition_id = str(action_capsule.get("transition_id", "")).strip() or _transition_id(
        prior.state_hash,
        covenant_rule.rule_id,
        replay_identity,
    )

    if replay_identity != covenant_rule.replay_identity:
        violations.append("replay_drift:identity_mismatch")

    for key in covenant_rule.precondition_keys:
        if key not in normalized_prior:
            violations.append(f"precondition_failure:missing_key:{key}")

    target_value = normalized_prior.get(covenant_rule.target_key)
    if target_value is None or isinstance(target_value, bool) or not isinstance(target_value, (int, float)):
        violations.append(f"precondition_failure:target_not_numeric:{covenant_rule.target_key}")

    next_data: Dict[str, StateValue] = dict(normalized_prior)
    if not violations:
        updated_value = float(target_value) + covenant_rule.delta
        next_data[covenant_rule.target_key] = updated_value

        for key in covenant_rule.invariant_keys:
            if key not in normalized_prior:
                violations.append(f"invariant_failure:missing_key:{key}")
                continue
            if next_data.get(key) != normalized_prior.get(key):
                violations.append(f"invariant_failure:key_changed:{key}")

        for key in covenant_rule.postcondition_keys:
            if key not in next_data:
                violations.append(f"postcondition_failure:missing_key:{key}")

        bounded_value = next_data[covenant_rule.target_key]
        if covenant_rule.min_value is not None and bounded_value < covenant_rule.min_value:
            violations.append("postcondition_failure:min_bound")
        if covenant_rule.max_value is not None and bounded_value > covenant_rule.max_value:
            violations.append("postcondition_failure:max_bound")

    accepted = len(violations) == 0
    final_next_data = next_data if accepted else dict(normalized_prior)

    next_state = CovenantState(
        state_data=final_next_data,
        state_hash=_state_hash(final_next_data),
    )

    receipt_hash = _sha256_hex(
        _canonical_json(
            {
                "accepted": accepted,
                "next_state_hash": next_state.state_hash,
                "prior_state_hash": prior.state_hash,
                "replay_identity": replay_identity,
                "rule_id": covenant_rule.rule_id,
                "transition_id": transition_id,
                "violations": tuple(violations),
            }
        ).encode("utf-8")
    )
    receipt = CovenantTransitionReceipt(
        transition_id=transition_id,
        prior_state_hash=prior.state_hash,
        next_state_hash=next_state.state_hash,
        rule_id=covenant_rule.rule_id,
        replay_identity=replay_identity,
        accepted=accepted,
        violations=tuple(violations),
        receipt_hash=receipt_hash,
    )
    return DeterministicCovenantExecution(
        prior_state=prior,
        next_state=next_state,
        receipt=receipt,
        rule=covenant_rule,
    )


def compare_covenant_replay(
    baseline: DeterministicCovenantExecution,
    replay: DeterministicCovenantExecution,
) -> Tuple[bool, Tuple[str, ...]]:
    violations = []
    if baseline.stable_hash() != replay.stable_hash():
        violations.append("replay_drift:execution_hash")
    if baseline.receipt.stable_hash() != replay.receipt.stable_hash():
        violations.append("replay_drift:receipt_hash")
    if baseline.next_state.stable_hash() != replay.next_state.stable_hash():
        violations.append("replay_drift:next_state_hash")
    if baseline.receipt.replay_identity != replay.receipt.replay_identity:
        violations.append("replay_drift:replay_identity")
    return len(violations) == 0, tuple(violations)
