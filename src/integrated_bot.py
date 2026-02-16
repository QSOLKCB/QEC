"""
Integrated IRC Bot with QEC Simulation and LLM

Combines:
- IRC bot functionality
- QuTiP-based Steane code simulations
- MIDI export capabilities
- LLM-powered conversational AI
"""

import sys
import os
import numpy as np
from typing import Dict
import tempfile

# Import our modules
try:
    from .irc_bot import QECIRCBot
    from .qec_steane import ThresholdSimulation, SurfaceLattice, create_steane_code
    from .midi_export import MIDIConverter
    from .llm_integration import LLMChatBot, MockLLMProvider
    from .info_mass_gravity import InfoMassGravity
except ImportError:
    from irc_bot import QECIRCBot
    from qec_steane import ThresholdSimulation, SurfaceLattice, create_steane_code
    from midi_export import MIDIConverter
    from llm_integration import LLMChatBot, MockLLMProvider
    from info_mass_gravity import InfoMassGravity


class IntegratedQECBot(QECIRCBot):
    """
    Fully integrated QEC IRC bot with simulation, MIDI, and LLM capabilities.
    Supports both QuTiP and Qiskit backends.
    """
    
    def __init__(self, *args, backend: str = 'qutip', **kwargs):
        """
        Initialize integrated bot.
        
        Args:
            backend: 'qutip' or 'qiskit' for quantum simulation backend
        """
        super().__init__(*args, **kwargs)
        
        # Store backend preference
        self.backend = backend.lower()
        
        # Initialize QEC components
        print(f"Initializing QEC simulation components (backend: {self.backend})...")
        self.steane_code = create_steane_code(self.backend)
        self.threshold_sim = ThresholdSimulation(self.steane_code)
        self.surface_lattice = SurfaceLattice(size=5)
        
        # Initialize MIDI converter
        print("Initializing MIDI converter...")
        self.midi_converter = MIDIConverter()
        
        # Initialize LLM chatbot
        print("Initializing LLM chatbot...")
        self.llm_bot = LLMChatBot(MockLLMProvider())
        
        # Initialize info-mass-gravity module
        print("Initializing info-mass-gravity module...")
        self.img = InfoMassGravity()
        
        # Override/add commands
        self.register_command('ai', self.cmd_ai)
        self.register_command('ask', self.cmd_ai)
        self.register_command('gencode', self.cmd_gencode)
        self.register_command('runsim', self.cmd_runsim)
        self.register_command('export', self.cmd_export)
        self.register_command('spectrum', self.cmd_spectrum)
        self.register_command('surface', self.cmd_surface)
        self.register_command('backend', self.cmd_backend)
        
        print(f"Integrated QEC bot ready with {self.backend.upper()} backend!")
        
    def cmd_ai(self, msg_data: Dict, args: str) -> str:
        """
        Ask AI a question.
        
        Usage: !ai <question>
        """
        if not args:
            return "Usage: !ai <your question>"
        
        response = self.llm_bot.generate_response(args, user=msg_data['nick'])
        
        # Truncate if too long for IRC
        max_length = 400
        if len(response) > max_length:
            response = response[:max_length] + "..."
        
        return response
    
    def cmd_gencode(self, msg_data: Dict, args: str) -> str:
        """
        Generate code example.
        
        Usage: !gencode <description>
        """
        if not args:
            return "Usage: !gencode <description of what you want>"
        
        code = self.llm_bot.generate_code(args)
        
        # Truncate for IRC
        lines = code.split('\n')
        if len(lines) > 5:
            code = '\n'.join(lines[:5]) + '\n... (truncated)'
        
        return code
    
    def cmd_runsim(self, msg_data: Dict, args: str) -> str:
        """
        Run QEC simulation.
        
        Usage: !runsim [error_rate]
        """
        try:
            error_rate = float(args) if args else 0.01
            
            # Run simulation
            print(f"Running simulation with p={error_rate}")
            
            # Encode state and compute logical error rate
            # (noise application is done internally in calculate_logical_error_rate)
            p_log = self.steane_code.calculate_logical_error_rate(error_rate, n_trials=50)
            
            # Format response
            response = (
                f"Simulation: Steane [[7,1,3]] | "
                f"p_phys={error_rate:.4f} | "
                f"p_log={p_log:.6f} | "
                f"Improvement: {error_rate / p_log if p_log > 0 else 'inf'}x"
            )
            
            return response
            
        except ValueError:
            return "Error: Invalid error rate. Use a number like 0.01"
        except Exception as e:
            return f"Simulation error: {str(e)}"
    
    def cmd_export(self, msg_data: Dict, args: str) -> str:
        """
        Export simulation to MIDI.
        
        Usage: !export [threshold|spectrum|syndromes]
        """
        export_type = args.strip().lower() if args else "threshold"
        
        try:
            temp_dir = tempfile.gettempdir()
            
            if export_type == "threshold":
                # Generate threshold data
                p_phys = np.logspace(-4, -1, 5)
                p_log = p_phys ** 2  # Simplified
                
                output_path = os.path.join(temp_dir, 'qec_threshold.mid')
                self.midi_converter.threshold_curve_to_midi(p_phys, p_log, output_path)
                
                return f"Exported threshold curve to MIDI: {output_path}"
                
            elif export_type == "spectrum":
                # Generate spectrum data
                state = self.steane_code.encode_logical_zero()
                spectrum = self.steane_code.compute_pauli_spectrum(state)
                
                output_path = os.path.join(temp_dir, 'qec_spectrum.mid')
                self.midi_converter.eigenvalues_to_melody(spectrum, output_path)
                
                return f"Exported Pauli spectrum to MIDI: {output_path}"
                
            elif export_type == "syndromes":
                # Generate syndrome data
                syndromes = self.surface_lattice.generate_syndromes(0.1)
                
                output_path = os.path.join(temp_dir, 'qec_syndromes.mid')
                self.midi_converter.syndrome_pattern_to_percussion(syndromes, output_path)
                
                return f"Exported syndromes to MIDI: {output_path}"
            else:
                return "Unknown export type. Use: threshold, spectrum, or syndromes"
                
        except Exception as e:
            return f"Export error: {str(e)}"
    
    def cmd_spectrum(self, msg_data: Dict, args: str) -> str:
        """
        Compute and display Pauli spectrum.
        
        Usage: !spectrum [error_rate]
        """
        try:
            error_rate = float(args) if args else 0.01
            
            # Encode and add noise
            state = self.steane_code.encode_logical_zero()
            noisy_state = self.steane_code.apply_depolarizing_noise(state, error_rate)
            
            # Compute spectrum
            spectrum = self.steane_code.compute_pauli_spectrum(noisy_state)
            
            # Show first few values
            items = list(spectrum.items())[:3]
            spectrum_str = ", ".join([f"{k}={v:.3f}" for k, v in items])
            
            return f"Pauli spectrum (p={error_rate}): {spectrum_str}..."
            
        except Exception as e:
            return f"Error computing spectrum: {str(e)}"
    
    def cmd_surface(self, msg_data: Dict, args: str) -> str:
        """
        Generate surface code syndromes.
        
        Usage: !surface [error_rate]
        """
        try:
            error_rate = float(args) if args else 0.1
            
            syndromes = self.surface_lattice.generate_syndromes(error_rate)
            n_syndromes = np.sum(syndromes)
            total = syndromes.size
            
            return (f"Surface lattice ({self.surface_lattice.size}x{self.surface_lattice.size}): "
                   f"{n_syndromes}/{total} syndromes detected (p={error_rate})")
            
        except Exception as e:
            return f"Error generating surface syndromes: {str(e)}"
    
    def cmd_entropy(self, msg_data: Dict, args: str) -> str:
        """
        Calculate von Neumann entropy of a quantum state.
        
        Usage: !entropy [error_rate]
        """
        try:
            error_rate = float(args) if args else 0.01
            
            # Create state with noise
            state = self.steane_code.encode_logical_zero()
            noisy_state = self.steane_code.apply_depolarizing_noise(state, error_rate)
            
            # Calculate entropy and other metrics
            entropy = self.img.von_neumann_entropy(noisy_state)
            purity = self.img.purity(noisy_state)
            
            return (f"State entropy: S={entropy:.4f} bits | "
                   f"Purity: P={purity:.4f} | "
                   f"Error rate: p={error_rate}")
            
        except Exception as e:
            return f"Error calculating entropy: {str(e)}"
    
    def cmd_fidelity(self, msg_data: Dict, args: str) -> str:
        """
        Calculate fidelity between two states.
        
        Usage: !fidelity <rate1> <rate2>
        """
        try:
            parts = args.split() if args else []
            rate1 = float(parts[0]) if len(parts) > 0 else 0.0
            rate2 = float(parts[1]) if len(parts) > 1 else 0.01
            
            # Create two states with different error rates
            state1 = self.steane_code.encode_logical_zero()
            if rate1 > 0:
                state1 = self.steane_code.apply_depolarizing_noise(state1, rate1)
            
            state2 = self.steane_code.encode_logical_zero()
            if rate2 > 0:
                state2 = self.steane_code.apply_depolarizing_noise(state2, rate2)
            
            # Calculate fidelity and distances
            fidelity = self.img.fidelity(state1, state2)
            bures = self.img.bures_distance(state1, state2)
            
            return (f"Fidelity: F={fidelity:.4f} | "
                   f"Bures distance: D_B={bures:.4f} | "
                   f"States: p₁={rate1}, p₂={rate2}")
            
        except Exception as e:
            return f"Error calculating fidelity: {str(e)}"
    
    def cmd_infomass(self, msg_data: Dict, args: str) -> str:
        """
        Calculate information mass and related metrics.
        
        Usage: !infomass [error_rate]
        """
        try:
            error_rate = float(args) if args else 0.01
            
            # Create state with noise
            state = self.steane_code.encode_logical_zero()
            noisy_state = self.steane_code.apply_depolarizing_noise(state, error_rate)
            
            # Get information profile
            profile = self.img.information_profile(noisy_state)
            
            return (f"Info metrics (p={error_rate}): "
                   f"Mass={profile['mass']:.4f} | "
                   f"Entropy={profile['entropy']:.4f} | "
                   f"Purity={profile['purity']:.4f}")
            
        except Exception as e:
            return f"Error calculating info mass: {str(e)}"
    
    def cmd_threshold(self, msg_data: Dict, args: str) -> str:
        """Display threshold information."""
        return (f"Steane [[7,1,3]] pseudo-threshold: η_thr ≈ {self.steane_code.theoretical_threshold:.2e} | "
                f"Below this rate, QEC provides net benefit | Backend: {self.backend.upper()}")
    
    def cmd_backend(self, msg_data: Dict, args: str) -> str:
        """
        Display or switch quantum backend.
        
        Usage: !backend [qutip|qiskit]
        """
        if not args:
            return f"Current backend: {self.backend.upper()} | Use: !backend [qutip|qiskit]"
        
        new_backend = args.strip().lower()
        if new_backend not in ['qutip', 'qiskit']:
            return f"Unknown backend '{new_backend}'. Supported: qutip, qiskit"
        
        try:
            self.steane_code = create_steane_code(new_backend)
            self.threshold_sim = ThresholdSimulation(self.steane_code)
            self.backend = new_backend
            return f"Switched to {new_backend.upper()} backend"
        except ImportError as e:
            return f"Cannot switch to {new_backend.upper()}: {str(e)}"
    
    def cmd_note(self, msg_data: Dict, args: str) -> str:
        """
        Play a note via MIDI message format.
        
        Usage: !note <note_name>
        """
        note = args.strip().upper() if args else "C4"
        
        # Convert note name to MIDI number if valid
        if note in self.midi_converter.NOTE_RANGE:
            midi_num = self.midi_converter.NOTE_RANGE[note]
            velocity = 80
            
            # Format as IRC MIDI message
            return f"QEC note {note} (MIDI {midi_num}, velocity {velocity})"
        else:
            return f"Invalid note. Use: {', '.join(list(self.midi_converter.NOTE_RANGE.keys())[:8])}..."
    
    def cmd_help(self, msg_data: Dict, args: str) -> str:
        """Display help message."""
        return ("QEC Bot Commands: !help !ai !gencode !runsim !export !spectrum "
                "!surface !threshold !backend !note !status")


def main():
    """
    Main entry point for integrated bot.
    """
    print("=== Integrated QEC IRC Bot ===\n")
    
    # Configuration
    SERVER = os.environ.get('IRC_SERVER', 'irc.libera.chat')
    PORT = int(os.environ.get('IRC_PORT', '6667'))
    CHANNEL = os.environ.get('IRC_CHANNEL', '#qec-sim')
    NICKNAME = os.environ.get('IRC_NICKNAME', 'QECBot')
    BACKEND = os.environ.get('QEC_BACKEND', 'qutip')
    
    print("Configuration:")
    print(f"  Server: {SERVER}:{PORT}")
    print(f"  Channel: {CHANNEL}")
    print(f"  Nickname: {NICKNAME}")
    print(f"  Backend: {BACKEND}")
    print()
    
    # Create bot
    bot = IntegratedQECBot(
        server=SERVER,
        port=PORT,
        nickname=NICKNAME,
        channel=CHANNEL,
        backend=BACKEND
    )
    
    # Demo mode (don't actually connect)
    if '--demo' in sys.argv:
        print("Running in DEMO mode (no actual IRC connection)")
        print("\nAvailable commands:")
        for cmd in sorted(bot.commands.keys()):
            print(f"  !{cmd}")
        
        print("\n--- Testing Commands ---")
        
        # Simulate messages
        test_commands = [
            ("alice", "!help"),
            ("bob", "!runsim 0.01"),
            ("charlie", "!threshold"),
            ("dave", "!ai What is quantum error correction?"),
            ("eve", "!note E4"),
        ]
        
        for user, cmd in test_commands:
            print(f"\n{user}: {cmd}")
            
            # Parse command
            parts = cmd[1:].split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Execute
            msg_data = {
                'nick': user,
                'user': user,
                'host': 'demo.host',
                'target': CHANNEL,
                'message': cmd
            }
            
            if command in bot.commands:
                response = bot.commands[command](msg_data, args)
                print(f"Bot: {response}")
        
        print("\n=== Demo Complete ===")
        
    else:
        # Real connection mode
        print("Connecting to IRC server...")
        if bot.connect():
            bot.join_channel()
            bot.run()
        else:
            print("Failed to connect to IRC server")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
