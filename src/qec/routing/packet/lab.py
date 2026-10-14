# SPDX-License-Identifier: MPL-2.0
"""Derived, explicitly non-authoritative packet laboratory exports."""
from .core import validate_packet_receipt


def create_packet_demo_export(receipt: object) -> dict:
    validate_packet_receipt(receipt)
    # Deliberately exclude canonical receipt hashes and schemas. The browser
    # sees useful event observations but cannot issue canonical evidence.
    packet_names = {e["sha256"]: f'{e["request_id"]}:{e["sequence"]}' for e in receipt["envelopes"]}
    topology = receipt["input"]["topology"]
    edges = sorted({(r["node"], n) for r in topology["routes"] for n in r["next_hops"]})
    return {
        "schema": "qec.packet-demo.v1", "contract_version": "175.6", "demo_only": True,
        "cryptographic_proof": False, "hardware_evidence": False,
        "nodes": list(topology["nodes"]), "edges": [list(edge) for edge in edges],
        "queue_capacity": receipt["input"]["queue_policy"]["capacity"],
        "events": [{"index": e["index"], "logical_tick": e["logical_tick"], "node": e["node"],
                    "packet": packet_names.get(e["packet_sha256"]), "kind": e["kind"],
                    "next_hop": e["details"].get("next_hop") or e["details"].get("selected_next_hop"),
                    "reason": e["details"].get("reason"),
                    "queue_size": len(e["details"]["queue_after"]) if "queue_after" in e["details"] else None}
                   for e in receipt["hop_receipt_chain"]["events"]],
        "results": [{"request_id": r["request_id"], "destination": r["destination"],
                     "outcome": r["outcome"], "reason": r["reason"]} for r in receipt["results"]],
        "claim": "Offline visualization and sonification only. No cryptographic proof or hardware claims.",
    }
