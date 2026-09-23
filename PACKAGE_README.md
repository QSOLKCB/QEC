# QEC 172.1.0 — Marker/Common-Control Development Package

This package description applies to the **172.1.0 development/package
candidate** for the v172.x Crossbar coordinate-switching phase.

The authoritative published stable release is **v172.0**. Package-version
metadata and published tags are separate; v172.1 is not yet a published release.

## v172.1 Marker/Common-Control Contract

The candidate adds a sealed input register, a matrix/request-bound marker
programme, and a deterministic exact-coordinate plan or rejection receipt.
Replay validation reconstructs every event and decision. Optional trusted input
hashes bind validation to a known run.

The marker reads one immutable matrix snapshot and evaluates at most one
requested intersection. Both links must be idle. It cannot mutate payloads,
change the caller-declared decoder identity, reserve links, commit connections,
or choose fallback routes. Marker release is recorded on selection and rejection.

`qec-crossbar marker` emits the register, programme, common-control receipt and
validation artifact. `qec-crossbar marker-validate` replays a receipt.

Primary new identity: `crossbar_common_control_receipt_hash`.

The published v172.0 matrix schemas, hashes and CLI commands are preserved.
Multi-stage selection, reservation/contention, continuity proof and cross-era
equivalence remain assigned to later milestones.

See [the marker contract](docs/CROSSBAR_COMMON_CONTROL.md) for the API, CLI,
ordering, failure modes, trust boundary and examples.

This software proves declared classical software behaviour only. It does not
establish physical Crossbar fidelity, carrier-grade reliability, decoder
correctness, quantum hardware behaviour or quantum advantage.
