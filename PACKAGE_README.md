# QEC 172.5.0 — Strowger/Panel/Crossbar Equivalence Battery Development Package

This package describes the **172.5.0 development/package candidate**.
The authoritative published stable release is **v172.4**. Package metadata
and published tags remain separate; v172.5 is not yet released.

The candidate completes the planned v172.x implementation with a bounded shared
corpus comparing Strowger, Panel and Crossbar route decisions. Fixed adapters map
four destinations and two lanes into the unchanged native models. Full native
receipts, Crossbar continuity evidence and an independent admission oracle bind
every result to its exact case, initial state and adapter contract.

The default battery contains 36 equivalent availability cases and five native
fault controls that must remain different. A passing battery therefore does not
mean every case is equivalent. Exact default-corpus coverage is explicit.

New commands: `qec-crossbar equivalence-demo`, `equivalence`, `equivalence-validate`.
Primary identity: `switch_equivalence_matrix_hash`.

The comparison covers requested/reached destination, routing outcome and selected
lane. Native commit semantics, event traces and reservation lifecycles are not
claimed equivalent. Strowger has no native payload field; payload preservation
is checked natively for Panel and Crossbar only. Only Crossbar carries a native
caller-declared decoder-output reference.

Published Strowger, Panel and v172.0–v172.4 contracts and decoder code remain
unchanged. The next implementation phase is v173.x ESS stored-program control;
full cross-era migration remains v176.x.

See [the equivalence contract](docs/SWITCH_EQUIVALENCE_BATTERY.md) for mappings,
bounds, replay, trusted input pins and the evidence boundary. These software
checks do not establish universal equivalence, authenticated provenance,
physical switching fidelity, decoder correctness or quantum advantage.
