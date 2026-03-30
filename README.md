QSOLKCB / QEC
Deterministic Structural Analysis, Adaptive Control & Operator Console
for LDPC / QLDPC Tanner Graph Dynamics

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)

[![Type](https://img.shields.io/badge/type-deterministic%20analysis%20%2B%20adaptive%20control-blue)]()
[![Engine](https://img.shields.io/badge/engine-structure--driven-lightblue)]()
[![Determinism](https://img.shields.io/badge/determinism-bitwise%20reproducible-success)]()
[![Mode](https://img.shields.io/badge/mode-no%20stochastic%20search-critical)]()
[![Architecture](https://img.shields.io/badge/architecture-measure%20%E2%86%92%20control%20%E2%86%92%20adapt-purple)]()

[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

QEC is a deterministic research framework for:

belief propagation (BP) dynamics
Tanner graph structure
spectral instability
regime transitions
adaptive control of decoding strategies
operator-facing diagnostic control via Rust TUI

It functions as:

🧠 A deterministic analysis system
🌌 A phase-space reconstruction engine
⚙️ A closed-loop adaptive controller
🖥 A live Rust TUI operator cockpit (v106+)
🧠 What QEC Is (Current)

QEC is not just a simulator.

It is a:

Deterministic Structural + Adaptive + Operator System

The current closed-loop pipeline is:

metrics
→ collapse analysis
→ control flow
→ memory
→ adaptive control
→ regime jump
→ self-healing
→ history window
→ operator console

Everything is:

deterministic
explainable
reproducible
externally controlled
decoder-safe
UI-separated
🚀 Core Capabilities (Current)
1. Structural Diagnostics
BP trajectory analysis
attractor / basin detection
oscillation & metastability metrics
free-energy landscape analysis
2. Spectral Analysis
non-backtracking spectrum
eigenvector localization (IPR)
trapping-set detection
spectral instability scoring
3. Adaptive Control Stack (v105+)
collapse prediction
damping control
trend memory
adaptive response
regime jump detection
self-healing control
persistence windows
4. Operator Console (v106+)
live Rust TUI
real-time diagnostics
invariant monitor
action dispatch console
command history
session export / replay
multi-session browsing (v106.6+)
⚙️ Architecture
Tanner Graph
↓
Diagnostics
↓
Collapse / Control
↓
Adaptive Response
↓
Self-Healing
↓
History Window
↓
Rust TUI Operator Console
🔁 Determinism Guarantees

QEC enforces strict reproducibility:

no hidden randomness
deterministic ordering
canonical JSON outputs
stable ranking
explicit seeded RNG only
import numpy as np
np.random.RandomState(seed)

If it cannot be reproduced byte-for-byte, it is not a result.

🖥 Rust TUI Operator Console (v106+)

QEC now includes a Linutil-inspired Rust TUI control surface for:

live diagnostics
adaptive control workflows
session replay
operator-driven law-engine actions

The layout is optimized for fast, keyboard-first workflows:

Left   → navigation / mode selection
Center → live diagnostics / history / action console
Right  → invariant health / system status
Bottom → hotkeys / command legend

The interface is designed as a true operator cockpit for deterministic control and system inspection.

🚀 Build & Run
Build
cd tui
cargo build --release
Run
cargo run --release
Production-style startup
./target/release/qec-tui
⌨️ Key Controls
Navigation
↑ / ↓   move selection
Enter   switch active mode
Q       quit
Mode Shortcuts
D   diagnostics
C   control flow
M   memory
A   adaptive
R   regime jump
H   self-healing
W   history window
I   invariants
L   law engine
X   actions console
Action Console

When inside Actions mode:

D   run diagnostics
I   run invariants
L   run law engine
R   refresh all
Session & Replay
E   export session log
P   replay last session
S   scan saved sessions
V   view session diff
✨ Current TUI Features

The Rust console currently supports:

live Python-engine diagnostics
invariant status monitor
adaptive control state view
regime history timeline
action dispatch console
timestamped command history
session export / replay
multi-session browser
diff viewer

This makes QEC a persistent operator workstation, not just a passive dashboard.

🧱 Architecture Invariant

The TUI follows a strict architectural law:

ZERO LOGIC IN UI

Python = deterministic engine + control truth
Rust   = render + dispatch + operator state

This boundary is treated as a hard invariant.

All analysis, control, law, and adaptation logic remains in Python.

Rust is responsible only for:

subprocess dispatch
JSON parsing
UI state
terminal rendering
session persistence
🧠 Design Philosophy

Small is beautiful.
Determinism is architecture.
Structure before control.
Control before adaptation.
Adaptation before operation.
Operation before optimization.

If it cannot be reproduced byte-for-byte, it is not a result.

👤 Author

Trent Slade
QSOL-IMC
ORCID: https://orcid.org/0009-0002-4515-9237
