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

from .qec_steane import SteaneCode, ThresholdSimulation, SurfaceLattice
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
from .midi_export import MIDIConverter
from .irc_bot import IRCBot, QECIRCBot
from .llm_integration import LLMChatBot, MockLLMProvider
from .integrated_bot import IntegratedQECBot
from .info_mass_gravity import InfoMassGravity

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
