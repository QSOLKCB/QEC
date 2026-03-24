# IMPLEMENTATION.md  
QEC Implementation Notes, Legacy Components & Reproducibility Hooks

Author: Trent Slade — QSOL-IMC  
ORCID: 0009-0002-4515-9237  

---

## ⚠️ Scope

This document is **NOT used by Claude Code**.

It exists for:

- human reference  
- onboarding  
- legacy system context  
- reproducibility experiments  

It may contain **historical components** that are no longer part of the core architecture.

The authoritative implementation constraints are defined in:

- `CLAUDE.md`
- `ROADMAP.md`
- `CURRENT_TASK.md`

---

# 🧠 System Evolution Context

QEC has evolved from:
simulation demos → diagnostic framework → spectral analysis → deterministic discovery system


This file preserves **early-stage execution pathways** and auxiliary tools.

---

# ⚙️ Repository Setup

Clone the repository:

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC

Install dependencies:

pip install -r requirements.txt

Or (preferred modern workflow):

pip install -e .
🧪 Legacy Simulation Demo (Historical)

⚠️ This is a legacy entry point
Not part of the modern spectral/diagnostic pipeline

Run:

python src/qec_steane.py

Example output:

=== Steane Code QEC Simulation Demo ===
Initialized Steane [[7,1,3]] code
...
=== Demo Complete ===

Purpose:

early validation of QEC simulation stack
sanity check for encoding / noise / decoding
🎵 MIDI Sonification Pipeline (Experimental)

QEC includes an experimental sonification layer.

Run full demo:

python examples/qec_demo_full.py

Produces:

MIDI output (/tmp/qec_demo.mid)
simulation summary
optional LLM integration output
MIDI Parsing

Parse MIDI output:

python qec_mid_parser.py /tmp/qec_demo.mid

Mapping:

note 60 → syndrome 0
note 70 → syndrome 1

Purpose:

auditory inspection of syndrome dynamics
experimental debugging / visualization
📊 Legacy Benchmark Output

Example table:

physical_error | Steane | Surface | Reed-Muller | Fusion-QEC
...

Includes:

pseudo-threshold estimates
comparative decoder behavior
⚛️ Fusion-QEC / IonQ Notes (Historical Research)

Fusion-QEC references:

ion-trap architectures
CliNR protocol (2× logical error improvement target)
projected <10⁻¹² logical error rates

⚠️ Important:

These values are illustrative / research-aligned
Not device-calibrated
🤖 IRC Bot Integration (Experimental)

Run:

export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py

Purpose:

live simulation interaction
AI-assisted output interpretation
🔬 Modern System (Authoritative)

The current QEC system is NOT driven by the above demos.

It operates as:

metrics → attractor → strategy → evaluation → adaptation

Core modules:

src/qec/diagnostics/
src/qec/analysis/
src/qec/discovery/
src/qec/experiments/
🧱 Implementation Philosophy

QEC enforces:

deterministic execution
no hidden randomness
canonical JSON artifacts
strict layer separation

Legacy components may violate these principles —
they are preserved for context, not authority.

🔁 Reproducibility Notes

Modern QEC requires:

import numpy as np
rng = np.random.RandomState(seed)

All experiments must:

use explicit seeds
avoid global RNG
produce byte-identical outputs
🚫 Known Legacy Limitations

The following are not aligned with modern QEC architecture:

demo-style scripts (qec_steane.py)
MIDI pipeline (non-deterministic timing artifacts possible)
IRC bot integration
mixed simulation + analysis flows

These are:

historical artifacts, not core system components

🧠 Recommended Usage

Use this file to:

understand early system evolution
run quick sanity demos
explore experimental visualization tools

Do NOT use this file as:

architectural reference
implementation authority
source of truth for current system
📌 Summary

This file captures:

early QEC simulation workflows
experimental pipelines (MIDI, IRC)
historical benchmarking outputs

The modern system has moved toward:

deterministic, invariant-driven, spectral discovery

🧠 Final Note

If it’s not deterministic,
it’s not part of the system.

If it’s not layered correctly,
it’s not part of the architecture.

If it’s here —
it’s probably history.
