"""Controller layer — deterministic experiment orchestration.

Provides FSM-based control loops that orchestrate perturbation probes,
invariant analysis, and phase-aware decision logic.

Version: v80.0.0

Does not modify decoder internals.  Fully deterministic.
"""

from .qec_fsm import QECFSM
