"""
Tests for MIDI export module.
"""

import pytest
import numpy as np
from midi_export import MIDIConverter


def test_midi_converter_initialization():
    """Test MIDI converter initialization."""
    converter = MIDIConverter()
    assert converter.ticks_per_beat == 480
    assert converter.min_velocity == 8
    assert converter.max_velocity == 92


def test_error_rate_to_tempo():
    """Test error rate to tempo conversion."""
    converter = MIDIConverter()
    
    # Test standard mapping
    tempo = converter.error_rate_to_tempo(0.01)
    assert tempo == 120
    
    # Test bounds
    tempo = converter.error_rate_to_tempo(0.001)
    assert 60 <= tempo <= 240
    
    tempo = converter.error_rate_to_tempo(0.1)
    assert 60 <= tempo <= 240


def test_eigenvalue_to_velocity():
    """Test eigenvalue to velocity conversion."""
    converter = MIDIConverter()
    
    # Test extremes
    vel_min = converter.eigenvalue_to_velocity(-1.0)
    assert vel_min == 8
    
    vel_max = converter.eigenvalue_to_velocity(1.0)
    assert vel_max == 92
    
    # Test middle
    vel_mid = converter.eigenvalue_to_velocity(0.0)
    assert 8 <= vel_mid <= 92


def test_logical_error_to_arpeggio():
    """Test logical error to arpeggio conversion."""
    converter = MIDIConverter()
    
    # Test different error rates
    arpeggio_low = converter.logical_error_to_arpeggio(0.001)
    assert len(arpeggio_low) >= 3
    
    arpeggio_high = converter.logical_error_to_arpeggio(0.2)
    assert len(arpeggio_high) >= 3
    
    # Check structure
    for note, velocity, duration in arpeggio_low:
        assert 0 <= note <= 127
        assert 0 <= velocity <= 127
        assert duration > 0