# QEC 172.2.0 — Multi-Stage Link Selection Development Package

This package description applies to the **172.2.0 development/package
candidate** for the v172.x Crossbar coordinate-switching phase.

The authoritative published stable release is **v172.1**. Package-version
metadata and published tags remain separate; v172.2 is not yet released.

## v172.2 Multi-Stage Link Selection

The candidate adds immutable layered fabrics, explicit adjacent-stage wiring,
and deterministic reverse reachability followed by first-complete-path
selection. Each stage can contain multiple published v172.0 matrices.

Search binds the fabric, complete request, policy and evaluation budget.
Receipts distinguish selection, no admissible complete path, unknown endpoints
and budget exhaustion. Every decision is replayed; optional trusted input
hashes bind validation to the intended run. All outcomes record marker release.

New CLI commands: `qec-crossbar fabric-demo`, `path-search`, and `path-validate`.
Primary identity: `crossbar_path_search_receipt_hash`.

The published v172.0 matrix and v172.1 single-matrix marker contracts are
preserved. Selected paths are plans: reservations, contention, connection
commit, separate continuity receipts and cross-era equivalence remain later
milestones. Payload bytes and declared decoder identity remain unchanged.

See [Multi-Stage Link Selection](docs/CROSSBAR_MULTISTAGE_SELECTION.md) for
ordering, bounds, failure modes, APIs, CLI examples and the trust boundary.

This software establishes declared classical software behaviour only. It does
not establish physical Crossbar fidelity, carrier-grade reliability, decoder
correctness, authenticated provenance, quantum hardware behaviour or advantage.
