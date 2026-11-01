#!/usr/bin/env python3
"""Verify the executed Phase 623 JUnit test inventory and emit a local receipt.

The receipt attests only to tests recorded in the supplied JUnit file for
the supplied source SHA. It is not cryptographically signed or independent
external custody; CI must record the source checkout and upload the inputs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

EXPECTED = {
    "test_phase623_wire_allow_once_replay_denied_and_receipts": 1,
    "test_phase623_wire_transplant_individually_valid_bundle_is_denied": 1,
    "test_phase623_wire_reused_operation_identity_denied_after_rotation": 1,
    "test_phase623_wire_each_revocation_denies_before_any_output": 4,
    "test_phase623_wire_wrong_client_and_channel_have_zero_protected_effect": 1,
    "test_phase623_wire_tampered_proof_and_policy_fail_closed_then_valid": 1,
    "test_phase623_wire_saturated_ledger_blocks_otherwise_valid_action": 1,
    "test_phase623_ledger_chain_independently_rejects_tampering": 1,
}

# Exact pytest node names observed in the previously successful Phase 623
# JUnit artifact (workflow run 41355448547). The revocation matrix requires
# all four *different* dimensions, not four arbitrary parameter instances.
REVOCATION_CASE = "test_phase623_wire_each_revocation_denies_before_any_output"
REVOCATION_PARAMETERS = (
    "operation-REVOKED_OPERATION",
    "signer-REVOKED_SIGNER",
    "capability-REVOKED_CAPABILITY",
    "policy-REVOKED_POLICY",
)
EXPECTED_TEST_NAMES = frozenset(
    {name for name, count in EXPECTED.items() if count == 1}
    | {f"{REVOCATION_CASE}[{parameter}]" for parameter in REVOCATION_PARAMETERS}
)
assert len(EXPECTED_TEST_NAMES) == sum(EXPECTED.values())


def inspect(xml_bytes: bytes, commit: str) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full lowercase SHA-1")
    root = ET.fromstring(xml_bytes)
    cases = list(root.iter("testcase"))
    if len(cases) != sum(EXPECTED.values()):
        raise ValueError("incomplete or unexpected Phase 623 test inventory")
    observed: Counter[str] = Counter()
    names: set[str] = set()
    for case in cases:
        name = case.attrib.get("name", "")
        base = name.split("[", 1)[0]
        if name not in EXPECTED_TEST_NAMES or name in names:
            raise ValueError("unrecognized, substituted, or repeated Phase 623 test")
        if not "test_phase623_end_to_end" in case.attrib.get("classname", ""):
            raise ValueError("wrong test module in executed JUnit")
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise ValueError("failed, errored, or skipped trial cannot pass")
        names.add(name)
        observed[base] += 1
    if dict(observed) != EXPECTED or names != EXPECTED_TEST_NAMES:
        raise ValueError("missing, duplicated, or substituted adversarial scenario")
    return {
        "schema": "qec-phase623-junit-inventory-v1",
        "claim": "pytest_executed_adversarial_trial_inventory",
        "source_commit": commit,
        "junit_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "test_count": len(cases),
        "scenario_count": len(EXPECTED),
        "tests": sorted(names),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "limitations": (
            "Receipt is based on CI-generated JUnit data, not an independent signature; "
            "XML durations and digest can vary across runs; real-world custody, "
            "human authentication, durable replay and external hardware are not proven."
        ),
    }


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: verify_phase623_trial.py INPUT.xml OUTPUT.json COMMIT", file=sys.stderr)
        return 2
    src, dst, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    try:
        data = inspect(src.read_bytes(), commit)
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        dst.write_text(payload, encoding="utf-8")
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"PHASE623_RECEIPT_REJECTED: {exc}", file=sys.stderr)
        return 1
    print(f"PHASE623_RECEIPT_VERIFIED: {data['test_count']} real JUnit trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
