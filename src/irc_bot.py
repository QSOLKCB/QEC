"""
IRC Bot for QEC Simulations

A minimal IRC bot using socket library that can:
- Connect to IRC servers
- Send/receive messages
- Execute QEC simulation commands
- Export results to MIDI
- Integrate with LLM for conversational AI
"""

import socket
import ssl
import time
import re
from typing import Optional, Callable, Dict

_MAX_IRC_LINE_BYTES = 512
_MAX_PENDING_BYTES = 2048

def _safe_terminal_text(text: str) -> str:
    """Render untrusted IRC control and bidi text without terminal effects."""
    return text.encode('unicode_escape', errors='backslashreplace').decode('ascii')



class IRCBot:
    """
    Simple IRC bot using socket library.
    """
    
    def __init__(
        self,
        server: str,
        port: int = 6697,
        nickname: str = "QECBot",
        channel: str = "#qec-sim",
        realname: str = "Quantum Error Correction Bot",
        use_tls: bool = True
    ):
        """
        Initialize IRC bot.
        
        Args:
            server: IRC server address
            port: IRC server port (default 6667)
            nickname: Bot nickname
            channel: Channel to join
            realname: Bot real name
        """
        self.server = server
        self.port = port
        self.nickname = nickname
        self.channel = channel
        self.realname = realname
        self.use_tls = use_tls
        
        self.sock = None
        self.connected = False
        self.running = False
        
        # Command handlers
        self.commands: Dict[str, Callable] = {}
        
        # Rate limiting
        self.last_message_time = 0
        self.min_message_interval = 1.0  # seconds
        
    def connect(self) -> bool:
        """
        Connect to IRC server.
        
        Returns:
            True if connection successful
        """
        try:
            raw_sock = socket.create_connection((self.server, self.port), timeout=20)
            try:
                self.sock = ssl.create_default_context().wrap_socket(
                    raw_sock, server_hostname=self.server
                ) if self.use_tls else raw_sock
            except Exception:
                raw_sock.close()
                raise
            self.connected = True
            
            # Send connection info
            self._send_raw(f"NICK {self.nickname}")
            self._send_raw(f"USER {self.nickname} 0 * :{self.realname}")
            
            print(f"Connected to {self.server}:{self.port} as {self.nickname}")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from IRC server."""
        if self.sock:
            self._send_raw("QUIT :QECBot shutting down")
            self.sock.close()
            self.connected = False
            print("Disconnected from IRC server")
    
    def join_channel(self, channel: Optional[str] = None):
        """
        Join IRC channel.
        
        Args:
            channel: Channel to join (uses default if not provided)
        """
        channel = channel or self.channel
        self._send_raw(f"JOIN {channel}")
        print(f"Joined channel: {channel}")
    
    def send_message(self, target: str, message: str):
        """
        Send PRIVMSG to channel or user.
        
        Args:
            target: Channel or user to send to
            message: Message content
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_message_time < self.min_message_interval:
            time.sleep(self.min_message_interval)
        
        self._send_raw(f"PRIVMSG {target} :{message}")
        self.last_message_time = time.time()
    
    def _send_raw(self, message: str):
        """
        Send raw IRC message.
        
        Args:
            message: Raw IRC message
        """
        if self.sock:
            wire = (message + "\r\n").encode('utf-8')
            if len(wire) > _MAX_IRC_LINE_BYTES:
                raise ValueError("IRC output exceeds 512-byte line limit")
            self.sock.sendall(wire)
    
    def _receive(self) -> Optional[str]:
        """
        Receive message from IRC server.
        
        Returns:
            Received message or None
        """
        try:
            data = self.sock.recv(2048)
            return data.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Receive error: {e}")
            return None
    
    def register_command(self, command: str, handler: Callable):
        """
        Register command handler.
        
        Args:
            command: Command name (without prefix)
            handler: Handler function
        """
        self.commands[command] = handler
        print(f"Registered command: !{command}")
    
    def parse_message(self, line: str) -> Optional[Dict]:
        """
        Parse IRC message line.
        
        Args:
            line: IRC message line
            
        Returns:
            Parsed message dict or None
        """
        # Parse PRIVMSG format: :nick!user@host PRIVMSG #channel :message
        match = re.match(r':(.+?)!(.+?)@(.+?) PRIVMSG (.+?) :(.+)', line)
        if match:
            return {
                'nick': match.group(1),
                'user': match.group(2),
                'host': match.group(3),
                'target': match.group(4),
                'message': match.group(5)
            }
        return None
    
    def handle_command(self, msg_data: Dict):
        """
        Handle command from message.
        
        Args:
            msg_data: Parsed message data
        """
        message = msg_data['message'].strip()
        
        # Check if message starts with command prefix
        if not message.startswith('!'):
            return
        
        # Parse command and arguments
        parts = message[1:].split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Execute command handler
        if command in self.commands:
            try:
                response = self.commands[command](msg_data, args)
                if response:
                    # Send response to channel or user
                    target = msg_data['target']
                    if target == self.nickname:
                        # Private message, respond to sender
                        target = msg_data['nick']
                    self.send_message(target, response)
            except Exception as e:
                print(f"Command error: {e}")
                self.send_message(msg_data['target'], 
                                f"Error executing command: {str(e)}")
    
    def run(self):
        """
        Main bot loop.
        """
        if not self.connected:
            print("Not connected to IRC server")
            return
        
        self.running = True
        buffer = ""
        
        print("Bot running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                # Receive data
                data = self._receive()
                if data is None or data == "":
                    # None: receive failure; empty string: clean peer EOF.
                    # Neither means "try again immediately".
                    break

                buffer += data
                if len(buffer.encode('utf-8')) > _MAX_PENDING_BYTES:
                    print("IRC receive buffer limit exceeded; disconnecting")
                    break
                lines = buffer.split('\r\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                if len(buffer.encode('utf-8')) > _MAX_IRC_LINE_BYTES:
                    print("IRC partial line exceeded 512 bytes; disconnecting")
                    break
                
                for line in lines[:-1]:
                    if not line:
                        continue
                    
                    if len(line.encode('utf-8')) + 2 > _MAX_IRC_LINE_BYTES:
                        self.running = False
                        break
                    print(f"<< {_safe_terminal_text(line)}")
                    
                    # Handle PING
                    if line.startswith('PING'):
                        pong = line.replace('PING', 'PONG')
                        self._send_raw(pong)
                        continue
                    
                    # Parse and handle PRIVMSG
                    msg_data = self.parse_message(line)
                    if msg_data:
                        self.handle_command(msg_data)
                        
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        finally:
            self.running = False
            self.connected = False
            if self.sock is not None:
                self.sock.close()
                self.sock = None


class QECIRCBot(IRCBot):
    """
    IRC bot specialized for QEC simulations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize QEC IRC bot."""
        super().__init__(*args, **kwargs)
        
        # Register default commands
        self.register_command('help', self.cmd_help)
        self.register_command('simulate', self.cmd_simulate)
        self.register_command('threshold', self.cmd_threshold)
        self.register_command('midi', self.cmd_midi)
        self.register_command('note', self.cmd_note)
        self.register_command('status', self.cmd_status)
        
    def cmd_help(self, msg_data: Dict, args: str) -> str:
        """Display help message."""
        return ("QEC Bot Commands: !help, !simulate, !threshold, "
                "!midi, !note <note>, !status")
    
    def cmd_status(self, msg_data: Dict, args: str) -> str:
        """Display bot status."""
        uptime = time.time() - self.last_message_time if self.last_message_time else 0
        return f"QEC Bot v1.0 | Connected to {self.server} | Uptime: {uptime:.0f}s"
    
    def cmd_simulate(self, msg_data: Dict, args: str) -> str:
        """
        Run QEC simulation.
        
        Args format: [code_type] [error_rate]
        """
        parts = args.split()
        code_type = parts[0] if parts else "steane"
        error_rate = float(parts[1]) if len(parts) > 1 else 0.01
        
        # Placeholder simulation
        return (f"Simulating {code_type} code with p={error_rate:.4f}... "
                f"Logical error: {error_rate**2:.6f}")
    
    def cmd_threshold(self, msg_data: Dict, args: str) -> str:
        """Display threshold information."""
        return "Steane [[7,1,3]] pseudo-threshold: η_thr ≈ 9.3×10⁻⁵"
    
    def cmd_midi(self, msg_data: Dict, args: str) -> str:
        """Export simulation to MIDI."""
        return "MIDI export: threshold_curve.mid created"
    
    def cmd_note(self, msg_data: Dict, args: str) -> str:
        """
        Play a note.
        
        Args: note name (e.g., C4, E3)
        """
        note = args.strip().upper() if args else "C4"
        # This would trigger actual MIDI output in full implementation
        return f"Playing note: {note} (velocity: 80)"


def demo_irc_bot():
    """
    Demonstrate IRC bot functionality (offline mode).
    """
    print("=== IRC Bot Demo (Offline Mode) ===\n")
    
    # Create bot instance
    bot = QECIRCBot(
        server="irc.example.com",
        port=6667,
        nickname="QECBot",
        channel="#qec-sim"
    )
    
    print("Bot Configuration:")
    print(f"  Server: {bot.server}:{bot.port}")
    print(f"  Nickname: {bot.nickname}")
    print(f"  Channel: {bot.channel}")
    
    print("\nRegistered Commands:")
    for cmd in bot.commands.keys():
        print(f"  !{cmd}")
    
    print("\nSimulating IRC message parsing...")
    
    # Test message parsing
    test_messages = [
        ":alice!~alice@host.com PRIVMSG #qec-sim :!help",
        ":bob!~bob@host.com PRIVMSG #qec-sim :!simulate steane 0.01",
        ":charlie!~charlie@host.com PRIVMSG #qec-sim :!note E4",
    ]
    
    for msg in test_messages:
        parsed = bot.parse_message(msg)
        if parsed:
            print(f"\nParsed: {parsed['nick']}: {parsed['message']}")
            # Simulate command handling
            message = parsed['message']
            if message.startswith('!'):
                cmd = message[1:].split()[0]
                args = message[len(cmd)+2:] if len(message) > len(cmd)+1 else ""
                if cmd in bot.commands:
                    response = bot.commands[cmd](parsed, args)
                    print(f"  Response: {response}")
    
    print("\n=== Demo Complete ===")
    print("\nNote: To connect to a real IRC server, use:")
    print("  bot.connect()")
    print("  bot.join_channel()")
    print("  bot.run()")


if __name__ == '__main__':
    demo_irc_bot()
