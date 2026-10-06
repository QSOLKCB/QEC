---

# 🔹 Operator Walkthrough — Rust TUI Control Surface

The Rust TUI is an operator-facing control surface for viewing deterministic system state, replay checkpoints, diagnostics, and phase structure.

## Install / launch

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

The installer uses a matching uploaded Linux x86_64 binary when one exists.
Otherwise it downloads the latest release tag's source and builds the TUI with
`cargo build --locked --release`. QEC v173.0 has no uploaded TUI binary, so it
uses this source-build path. On Ubuntu, install the build prerequisites first:

```bash
sudo apt install curl python3 cargo build-essential
```

The default destination is `/usr/local/bin` (using `sudo` when necessary).
For a user-owned destination:

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | QEC_INSTALL_DIR="$HOME/.local/bin" sh
export PATH="$HOME/.local/bin:$PATH"
```

Launch from a terminal:

```bash
qec-tui
```

The installer does not install the Python engine or its dependencies. Follow
[`INSTALL.md`](INSTALL.md) and activate that virtual environment before launching
the TUI. Python selection is `QEC_PYTHON` (an explicit executable path), then
`VIRTUAL_ENV`'s interpreter, then `python3` or `python` on `PATH`. An explicitly
configured interpreter or active environment is authoritative: errors do not
silently switch to a different installation.

```bash
QEC_PYTHON="/path/to/QEC/.venv/bin/python" qec-tui --check-engine
```

`--check-engine` reports the four Python panel results without opening a terminal
UI and exits nonzero if any adapter fails or returns invalid panel data.

### Live adapter boundary

Diagnostics, History Window, Invariants, and Phase Dynamics request
`qec.cli.diagnostics`, `qec.cli.history`, `qec.cli.invariants`, and
`qec.cli.phase_diagnostics`. The Law action requests `qec.cli.law_engine`.
These five modules are **not currently shipped by this repository**. Activating
a virtual environment fixes interpreter discovery; it does not supply missing
CLI adapters. Their live engine integration remains unfinished.

Backend failures are shown with the Python error, and the status reports
`UNAVAILABLE`. Refresh propagates any failed action. Control Flow, Memory,
Adaptive, Regime Jump, Self-Healing, and the Law Engine display also have no live
adapter and are identified as unconnected.

For an explicit sample display with no Python dependency:

```bash
QEC_TUI_DEMO=1 qec-tui
```

Sample data and simulated actions are labelled **DEMO**, including exported
session logs. Demo PASS values are layout examples, not invariant verification.

To build the current checkout instead:

```bash
cd tui
cargo build --locked --release
./target/release/qec-tui --version
./target/release/qec-tui
```

Current source supports noninteractive `--help`, `--version`, and
`--check-engine`. Older releases,
including v173.0, open the TUI regardless of arguments; the installer therefore
does not launch the binary as an installation check. Current source derives
`--version` from QEC's `pyproject.toml` at build time, so updating the QEC release
version automatically updates the TUI's reported version.

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
