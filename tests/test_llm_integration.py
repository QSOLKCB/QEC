"""
Tests for LLM integration module.
"""

import pytest
from llm_integration import RateLimiter, MockLLMProvider, LLMChatBot


def test_rate_limiter():
    """Test rate limiting functionality."""
    limiter = RateLimiter(calls_per_minute=3)
    
    # Should allow first 3 calls
    assert limiter.can_proceed()
    limiter.record_call()
    
    assert limiter.can_proceed()
    limiter.record_call()
    
    assert limiter.can_proceed()
    limiter.record_call()
    
    # Should block 4th call
    assert not limiter.can_proceed()


def test_mock_llm_provider():
    """Test mock LLM provider."""
    provider = MockLLMProvider()
    
    # Test code generation
    response = provider.generate("Show me example code")
    assert response is not None
    assert len(response) > 0
    
    # Test explanation
    response = provider.generate("What is quantum error correction?")
    assert 'Steane' in response or 'code' in response


def test_llm_chatbot():
    """Test LLM chatbot."""
    bot = LLMChatBot()
    
    # Test response generation
    response = bot.generate_response("What is the Steane code?", user="testuser")
    assert response is not None
    assert len(response) > 0
    
    # Check history
    assert len(bot.history) > 0


def test_content_moderation():
    """Test content moderation."""
    bot = LLMChatBot()
    
    # Test allowed message
    result = bot.moderate_message("Tell me about QEC")
    assert result['allowed'] is True
    
    # Test blocked message (if contains blocked words)
    result = bot.moderate_message("This is spam")
    assert result['allowed'] is False
    
    # Test too long message
    long_message = "a" * 1000
    result = bot.moderate_message(long_message)
    assert result['allowed'] is False
