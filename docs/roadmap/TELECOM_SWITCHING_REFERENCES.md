<!-- SPDX-License-Identifier: MPL-2.0 -->
# Telecommunications Switching Design References

This document records the historical design lineage that motivates the
post-v170 QEC switching programme in [`ROADMAP.md`](../../ROADMAP.md).

These references are design context. They are not QEC proof inputs, benchmark
evidence or authority for decoder correctness.

## Architectural lineage

```text
Strowger step-by-step switching
→ Panel sender/register separated control
→ Crossbar coordinate switching and common control
→ Electronic Switching Systems with stored-program control
→ Digital time-division switching
→ Packet switching and software-controlled routing
```

The sequence is used as a systems-design grammar. It does not claim that every
carrier, country or exchange migrated through one identical progression.

## Design concepts adopted by QEC

| Historical era | Design concept | QEC interpretation |
|---|---|---|
| Strowger | Direct step-by-step route progression | Explicit state transitions, pulse decoding, first-free selection and route receipts |
| Panel | Sender/register separates collected intent from path actuation | Seal the request and compile the route programme before touching the path |
| Crossbar | Coordinate matrix and common control | Bounded central path computation over an immutable switching matrix |
| ESS | Stored-program control | Versioned immutable programmes, logical scheduling and bounded feature modules |
| Digital circuit switching | Time-division multiplexing | Canonical frames, fixed slots, explicit alignment and deterministic interchange |
| Packet switching and softswitches | Independently routed software-defined envelopes | Canonical packets, deterministic route tables, bounded queues and replay-safe failover |

## Core references

The following works motivate the architecture programme:

- *A History of Engineering and Science in the Bell System: Switching Technology* — Panel, coordinate and common-control development.
- Western Electric, *Fundamentals of Telephone Communication Systems* — Panel senders, registers and Crossbar common control.
- Bell System Technical Journal, *Organization of the No. 1 ESS Stored Program* — stored-program switching control.
- ITU historical and technical material covering digital circuit-switched data networks, X.25 and the transition toward packet networks.

Implementation documents should provide exact edition, author, publication and
page references for any historical statement that affects a design decision.

## Historical claim boundary

Historical switching ideas may motivate software architecture, but a QEC receipt
proves only the declared software model under its bound inputs and assumptions.

The programme does not claim:

- physical fidelity to a deployed telephone exchange;
- carrier-grade capacity, latency or reliability;
- that electromechanical switching is quantum hardware;
- that a historical analogy proves decoder correctness;
- universal superiority of newer switching systems;
- that historical sources are substitutes for generated QEC evidence.

## Citation rule for future phases

Each implementation document should separate:

1. **Historical source claims** — statements about real switching technology,
   accompanied by bibliographic citations.
2. **QEC design interpretation** — the software contract inspired by the source.
3. **Generated evidence** — canonical artifacts, hashes, tests and receipts
   produced by the implementation.

No historical citation may be used to bypass QEC validation or claim boundaries.
