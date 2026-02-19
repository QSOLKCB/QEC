"""
Quantum Error Correction IRC Bot Package

Components:
- qec_steane: Steane code QEC simulations with QuTiP
- qec_qldpc_codes: Protograph-based quantum LDPC codes (Komoto-Kasai 2025)
- midi_export: MIDI export for simulation data
- irc_bot: IRC bot implementation
- llm_integration: LLM-powered conversational AI
- integrated_bot: Full integration of all components
- info_mass_gravity: Information-theoretic and physics-inspired metrics
"""

__version__ = '1.0.0'

from .qec_qldpc_codes import (
    QuantumLDPCCode,
    JointSPDecoder,
    GF2e,
    create_code as create_qldpc_code,
    hashing_bound,
    hashing_bound_threshold,
    simulate_frame_error_rate as simulate_qldpc_fer,
    update_pauli_frame,
    syndrome,
    bp_decode,
    detect,
    infer,
    channel_llr,
)

# Optional heavy-dependency modules are loaded on first attribute access
# to avoid pulling in matplotlib/qutip/mido when only core decoder is needed.
_LAZY_IMPORTS = {
    'SteaneCode': '.qec_steane',
    'ThresholdSimulation': '.qec_steane',
    'SurfaceLattice': '.qec_steane',
    'MIDIConverter': '.midi_export',
    'IRCBot': '.irc_bot',
    'QECIRCBot': '.irc_bot',
    'LLMChatBot': '.llm_integration',
    'MockLLMProvider': '.llm_integration',
    'IntegratedQECBot': '.integrated_bot',
    'InfoMassGravity': '.info_mass_gravity',
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'SteaneCode',
    'ThresholdSimulation',
    'SurfaceLattice',
    'QuantumLDPCCode',
    'JointSPDecoder',
    'GF2e',
    'create_qldpc_code',
    'hashing_bound',
    'hashing_bound_threshold',
    'simulate_qldpc_fer',
    'MIDIConverter',
    'IRCBot',
    'QECIRCBot',
    'LLMChatBot',
    'MockLLMProvider',
    'IntegratedQECBot',
    'InfoMassGravity',
    'update_pauli_frame',
    'syndrome',
    'bp_decode',
    'detect',
    'infer',
    'channel_llr',
]
