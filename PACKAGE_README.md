# QEC 171.5.0 — Panel Separated-Control Development Package

This package description applies to the **171.5.0 development/package candidate** implemented on the v171.x Panel separated-control branch.

The authoritative published stable release remains **v170.3.0** until the corresponding v171.x release tags are created. Package-version metadata and published-tag status are intentionally separate so an unreleased candidate is not presented as an already-published stable tag.

## v171.x Panel separated-control phase

The 171.5.0 candidate implements the planned v171.0 through v171.5 sequence:

- **v171.0 — Panel Exchange Skeleton:** canonical motor groups, banks, bounded selector positions, path inventory, and unique bank/selector actuation coordinates.
- **v171.1 — Sender and Register Contract:** complete request sealing, deterministic sender-program compilation, strict canonical sender-program validation, and sender/register binding receipts.
- **v171.2 — Separated Control-Path Receipt:** separate identities for request, payload, compiled control intent, actuation, verification, and outcome, with replay validation.
- **v171.3 — Deterministic Panel Translation Tables:** versioned exact translation tables and declared deterministic fallback rules with no ambient lookup.
- **v171.4 — Panel Fault and Capacity Battery:** explicit busy-bank, path-unavailable, motor-stall, translation-corruption, sender-disagreement, and capacity cases.
- **v171.5 — Offline Panel Laboratory:** dependency-free visualization and sonification labelled as demonstration evidence rather than canonical cryptographic evidence.

Primary Panel identities are `panel_sender_register_receipt_hash` and `panel_separated_control_receipt_hash`.

Cross-era Strowger/Panel equivalence is limited to declared destination/outcome equivalence. Both source receipts are replay-validated before an equivalence artifact can be issued; identical internal event traces are not required.

This software makes deterministic classical software-routing and replay claims only. It does not establish physical telephone-switch fidelity, hardware fault tolerance, decoder correctness merely because routing succeeds, quantum hardware behavior, physical truth, or quantum advantage.
