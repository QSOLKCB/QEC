# SPDX-License-Identifier: MPL-2.0
"""Opt-in classical acoustic syndrome experiments; no physical protection claim."""
from .channel import AcousticSource, ChannelConfig, PCM, encode_symbols, read_wav, validate_source, wav_bytes

__all__ = ["AcousticSource", "ChannelConfig", "PCM", "encode_symbols", "read_wav", "validate_source", "wav_bytes"]
