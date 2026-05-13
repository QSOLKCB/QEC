---

# 🔹 Operator Walkthrough — Rust TUI Control Surface

The Rust TUI is an operator-facing control surface for viewing deterministic system state, replay checkpoints, diagnostics, and phase structure.

## Install / launch

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
cd tui
cargo run --release
```

If the TUI binary is installed:

```bash
qec-tui
```

## Layout

```text
Left   → navigation
Center → workspace
Right  → system state
Bottom → hotkeys
```

## Panels

| Key | Panel | Purpose |
|---|---|---|
| `D` | Diagnostics | System state, invariants, convergence |
| `H` | History | Deterministic timeline and replay checkpoints |
| `P` | Phase | Attractor and phase-structure analysis |
| `A` | Actions | Deterministic control pathways |
| `R` | Replay | Reconstruction and hash-stability checks |
| `S` | Status | Global system integrity |
