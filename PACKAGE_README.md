# QEC 172.4.0 — Crossbar Continuity Verification Development Package

This package describes the **172.4.0 development/package candidate**.
The authoritative published stable release is **v172.3**. Package metadata
and published tags remain separate; v172.4 is not yet released.

The candidate adds independent forward continuity verification over replayed
multi-stage selection and contention receipts. Every selected coordinate and
wire must form one complete idle route with exact requested endpoints. Payload
bytes and the caller-declared decoder identity retain their upstream bindings.

Contention evidence covers every reserve attempt, binds each effective snapshot
and reservation, and distinguishes continuity at selection from reservation
activity at batch end. Released and quarantined routes retain historical proof
without a current-connection claim. Rejected attempts have no witness.

New commands: `qec-crossbar continuity`, `continuity-validate`.
Primary identity: `crossbar_continuity_receipt_hash`.

Published v172.0–v172.3 contracts remain unchanged. This is bounded classical
software-model verification, not connection commit or physical actuation.
Cross-era equivalence remains v172.5.

See [the continuity contract](docs/CROSSBAR_CONTINUITY.md) for the walk,
source coverage, bounds, trusted bindings, CLI examples and evidence boundary.
Replay does not establish authenticated provenance, current liveness, physical
switching behaviour, decoder correctness or quantum advantage.
