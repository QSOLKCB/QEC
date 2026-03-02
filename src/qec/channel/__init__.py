"""
Pluggable channel models for LLR construction.

Provides:

- :class:`ChannelModel` — abstract base.
- :class:`OracleChannel` — backward-compatible oracle LLR (sign from error vector).
- :class:`BSCSyndromeChannel` — syndrome-only BSC (uniform LLR, no sign leakage).
- :class:`BSCSyndromeStructuredChannel` — syndrome-biased BSC (v3.8.0 probe).
"""

from .base import ChannelModel
from .oracle import OracleChannel
from .bsc_syndrome import BSCSyndromeChannel
from .bsc_syndrome_structured import BSCSyndromeStructuredChannel

__all__ = [
    "ChannelModel",
    "OracleChannel",
    "BSCSyndromeChannel",
    "BSCSyndromeStructuredChannel",
    "get_channel_model",
]

_CHANNEL_REGISTRY = {
    "oracle": OracleChannel,
    "bsc_syndrome": BSCSyndromeChannel,
    "bsc_syndrome_structured": BSCSyndromeStructuredChannel,
}


def get_channel_model(name: str, **kwargs) -> ChannelModel:
    """Look up a channel model by name and return an instance.

    Parameters
    ----------
    name : str
        Registered channel model name.
    **kwargs
        Passed to the channel model constructor (e.g.
        ``structured_kappa`` for ``bsc_syndrome_structured``).

    Raises :class:`ValueError` if *name* is not a registered channel.
    """
    cls = _CHANNEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown channel_model {name!r}. "
            f"Available: {sorted(_CHANNEL_REGISTRY)}"
        )
    return cls(**kwargs)
