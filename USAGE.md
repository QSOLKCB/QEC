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
the TUI; its Python dispatch commands use `python` from your `PATH`.

To build the current checkout instead:

```bash
cd tui
cargo build --locked --release
./target/release/qec-tui --version
./target/release/qec-tui
```

Current source supports noninteractive `--help` and `--version`. Older releases,
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
