"""
BenchmarkConfig dataclass with JSON I/O.

Configs are canonicalized on load: sweep lists are sorted and all
defaults are made explicit so that two configs producing the same
benchmark are byte-identical after serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from .schema import SCHEMA_VERSION


@dataclass
class RuntimeConfig:
    """Runtime measurement sub-configuration."""
    warmup: int = 5
    runs: int = 30
    measure_memory: bool = False


@dataclass
class DecoderSpec:
    """Specification for one decoder variant in a benchmark sweep."""
    adapter: str = "bp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceModelConfig:
    """Analytical gate-cost estimation sub-configuration (v3.0.1).

    Opt-in only.  When ``enabled`` is False (default) the resource
    model is a no-op and does not affect benchmark results.
    """
    enabled: bool = False
    model: str = "qubit_decomp_v1"
    native_model: str = "native_placeholder_v1"
    assumptions: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration.

    All sweep lists (``distances``, ``p_values``) are sorted on
    construction to guarantee deterministic sweep ordering.
    """
    schema_version: str = SCHEMA_VERSION
    seed: int = 12345
    distances: list[int] = field(default_factory=lambda: [3, 5, 7])
    p_values: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.003])
    trials: int = 200
    max_iters: int = 50
    decoders: list[DecoderSpec] = field(
        default_factory=lambda: [
            DecoderSpec(adapter="bp", params={
                "mode": "min_sum",
                "schedule": "flooding",
            })
        ]
    )
    runtime_mode: str = "off"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    collect_iter_hist: bool = False
    deterministic_metadata: bool = False
    # v3.0.1 optional fields — absent by default for backward compat.
    qudit: dict[str, Any] | None = None
    resource_model: ResourceModelConfig | None = None
    # v3.1.3 channel model — defaults to "oracle" for backward compat.
    channel_model: str = "oracle"

    def __post_init__(self) -> None:
        self.distances = sorted(self.distances)
        self.p_values = sorted(self.p_values)
        # Ensure decoders are DecoderSpec instances (handles dict from JSON).
        processed: list[DecoderSpec] = []
        for d in self.decoders:
            if isinstance(d, dict):
                processed.append(DecoderSpec(**d))
            else:
                processed.append(d)
        self.decoders = processed
        # Ensure runtime is RuntimeConfig.
        if isinstance(self.runtime, dict):
            self.runtime = RuntimeConfig(**self.runtime)
        # Ensure qudit is a plain dict (from JSON) or None.
        if isinstance(self.qudit, dict):
            # Validate via QuditSpec (lazy import to avoid circular dep).
            from ..qudit.spec import QuditSpec
            QuditSpec.from_dict(self.qudit)  # validates; discard result
        # Ensure resource_model is ResourceModelConfig or None.
        if isinstance(self.resource_model, dict):
            self.resource_model = ResourceModelConfig(**self.resource_model)
        # Validate channel_model.
        _VALID_CHANNEL_MODELS = {"oracle", "bsc_syndrome", "bsc_syndrome_structured"}
        if self.channel_model not in _VALID_CHANNEL_MODELS:
            raise ValueError(
                f"channel_model must be one of {sorted(_VALID_CHANNEL_MODELS)}, "
                f"got {self.channel_model!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        d = asdict(self)
        # Omit v3.0.1 optional fields when absent so that v3.0.0
        # configs round-trip identically.
        if d.get("qudit") is None:
            d.pop("qudit", None)
        if d.get("resource_model") is None:
            d.pop("resource_model", None)
        # Omit channel_model when default ("oracle") so that pre-v3.1.3
        # configs produce identical serialized output.
        # Technical debt: suppression of "channel_model": "oracle" preserves
        # v3.1.2 artifact hashes.  This implicit default may be made explicit
        # in v4.0.0.  This is intentional backward-compatibility behavior —
        # do NOT change serialization logic without a major version bump.
        if d.get("channel_model") == "oracle":
            d.pop("channel_model", None)
        return d

    def to_json(self) -> str:
        """Serialize to a deterministic JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Construct from a plain dict (e.g., parsed JSON)."""
        d = dict(data)
        if "decoders" in d:
            d["decoders"] = [
                DecoderSpec(**dec) if isinstance(dec, dict) else dec
                for dec in d["decoders"]
            ]
        if "runtime" in d:
            rt = d["runtime"]
            if isinstance(rt, dict):
                d["runtime"] = RuntimeConfig(**rt)
        if "resource_model" in d:
            rm = d["resource_model"]
            if isinstance(rm, dict):
                d["resource_model"] = ResourceModelConfig(**rm)
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkConfig":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str) -> "BenchmarkConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        """Save config to a JSON file (deterministic formatting)."""
        with open(path, "w") as f:
            f.write(self.to_json())
            f.write("\n")
