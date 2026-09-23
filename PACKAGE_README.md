# QEC 173.0.0 — Stored-Program Switch Skeleton Development Package

This package describes the **173.0.0 development/package candidate**.
The authoritative published stable release is **v172.5**. Package metadata
and published tags remain separate; v173.0 is not yet released.

The candidate begins the v173.x ESS arc with separate immutable programme and
call stores, a bounded logical event queue and a switching-fabric adapter boundary.
The fixed built-in programme dispatches one independent planning command per
call through the existing Crossbar path selector. The controller requires native
replay and continuity verification against the exact request, programme and fabric.

Complete snapshots, ordered commands, native evidence and derived result counts
are hash-bound and replayable. Malformed queues, substituted adapter receipts and
rehashed evidence mutations are rejected. Native planning rejections remain valid
recorded outcomes; a passing validation does not imply that every route exists.

New commands: `qec-ess demo`, `qec-ess run`, `qec-ess validate`.
Primary identities: `ess_switch_skeleton_receipt_hash`, `ess_command_stream_hash`.

Every call plans against the same immutable fabric snapshot. This skeleton does
not reserve resources or commit connections. Full programme source manifests and
translation tables, call-processing transitions, priority scheduling, feature
modules and ESS/Fabric equivalence remain v173.1–v173.5 work.

Published Strowger, Panel and v172.0–v172.5 contracts and decoder code remain
unchanged. See [the ESS skeleton contract](docs/ESS_STORED_PROGRAM_SKELETON.md)
for APIs, bounds, replay and trusted input pins. These software checks do not
establish physical switching fidelity, authenticated provenance, decoder
correctness or quantum advantage.
