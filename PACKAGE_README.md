# QEC 172.3.0 — Contention and Busy-Link Receipts Development Package

This package describes the **172.3.0 development/package candidate**.
The authoritative published stable release is **v172.2**. Package metadata
and published tags remain separate; v172.3 is not yet released.

The candidate adds bounded logical contention over immutable layered Crossbar
fabrics. Canonical commands release, quarantine and reserve resources with exact
request-hash tie-breaking. Successful selection reserves the whole path atomically;
rejections leave no partial reservation. Owned releases check reservation and
request identities. Quarantine revokes a held path in full and blocks the target.

Receipts include busy-resource inventories, ownership, state transitions, nested
path-search evidence, marker release and replay validation with optional trusted
input hashes. A shared search budget bounds the complete batch. Initial external
busy links have no invented owner; marker release does not release a held path.

New commands: `qec-crossbar contention-demo`, `contend`, `contention-validate`.
Primary identity: `crossbar_contention_receipt_hash`.

Published v172.0, v172.1 and v172.2 contracts remain unchanged. This is a closed
classical model batch, not a persistent or concurrent reservation service.
Connection commit, separate continuity receipts and cross-era equivalence remain
later milestones. Payload bytes and declared decoder identity remain unchanged.

See [the contention contract](docs/CROSSBAR_CONTENTION.md) for ordering, ownership,
bounds, lifecycle semantics, CLI examples and the evidence boundary.

Replay does not establish authenticated provenance, physical switching behaviour,
decoder correctness or quantum advantage.
