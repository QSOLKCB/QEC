"""
Tests for IRC bot module.
"""

import pytest
from irc_bot import IRCBot, QECIRCBot


def test_irc_bot_initialization():
    """Test IRC bot initialization."""
    bot = IRCBot(
        server="irc.example.com",
        port=6667,
        nickname="TestBot",
        channel="#test"
    )
    
    assert bot.server == "irc.example.com"
    assert bot.port == 6667
    assert bot.nickname == "TestBot"
    assert bot.channel == "#test"
    assert not bot.connected


def test_message_parsing():
    """Test IRC message parsing."""
    bot = IRCBot(server="test.server", nickname="TestBot")
    
    # Test valid PRIVMSG
    line = ":alice!~alice@host.com PRIVMSG #channel :Hello, world!"
    parsed = bot.parse_message(line)
    
    assert parsed is not None
    assert parsed['nick'] == 'alice'
    assert parsed['target'] == '#channel'
    assert parsed['message'] == 'Hello, world!'
    
    # Test invalid message
    invalid_line = "INVALID MESSAGE FORMAT"
    parsed = bot.parse_message(invalid_line)
    assert parsed is None


def test_qec_bot_commands():
    """Test QEC bot command registration."""
    bot = QECIRCBot(
        server="test.server",
        nickname="QECBot",
        channel="#qec"
    )
    
    # Check default commands are registered
    assert 'help' in bot.commands
    assert 'simulate' in bot.commands
    assert 'threshold' in bot.commands
    assert 'midi' in bot.commands
    assert 'note' in bot.commands


def test_command_execution():
    """Test command execution."""
    bot = QECIRCBot(server="test.server", nickname="QECBot")
    
    msg_data = {
        'nick': 'alice',
        'user': 'alice',
        'host': 'host.com',
        'target': '#qec',
        'message': '!help'
    }
    
    # Execute help command
    response = bot.cmd_help(msg_data, "")
    assert response is not None
    assert 'help' in response.lower()
