# IRC Bot Guide

## Overview

The fusion-qec-sim IRC bot is an AI-powered assistant that integrates quantum error correction simulations with IRC chat capabilities. It combines:

- **Dual Backend Support**: Choose between QuTiP or Qiskit quantum computing frameworks
- **Steane Code [[7,1,3]]**: Full implementation with depolarizing noise
- **MIDI Export**: Convert simulation data to musical representations
- **LLM Integration**: Conversational AI for code generation and explanation
- **IRC Protocol**: Real-time communication on IRC servers

## Quick Start

```bash
# Demo mode (no IRC connection, default QuTiP backend)
python run_bot.py --demo

# Connect to IRC server with QuTiP backend (default)
export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py

# Use Qiskit backend instead
export QEC_BACKEND=qiskit
python run_bot.py
```

## Backend Selection

The bot supports two quantum simulation backends:

- **QuTiP** (default): Quantum Toolbox in Python - stable, widely used
- **Qiskit**: IBM's quantum computing framework - industry standard

Switch backends at runtime using the `!backend` command or set via environment variable.

## Available Commands

- `!help` - Display available commands
- `!runsim [error_rate]` - Run Steane code simulation
- `!threshold` - Display pseudo-threshold
- `!backend [qutip|qiskit]` - Switch or display quantum backend
- `!ai <question>` - Ask AI about QEC
- `!note <note>` - Play MIDI note
- `!export [type]` - Export to MIDI

See full documentation in the repository for detailed usage.
