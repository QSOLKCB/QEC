"""
Belief-propagation decoder adapter for the benchmarking framework.

Wraps the existing :func:`~src.qec_qldpc_codes.bp_decode` entrypoint
behind the :class:`DecoderAdapter` interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DecoderAdapter
from ..runtime import measure_runtime

from qec.channel.geometry_post import apply_geometry_postprocessing

class BPAdapter(DecoderAdapter):
    """Adapter wrapping ``bp_decode`` for benchmark use.

    All decoder parameters are captured explicitly so that comparison
    results are reproducible.
    """

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}
        self._H: np.ndarray | None = None
        self._structural_config: Any | None = None

    @property
    def name(self) -> str:
        mode = self._params.get("mode", "sum_product")
        schedule = self._params.get("schedule", "flooding")
        pp = self._params.get("postprocess", "none")
        return f"bp_{mode}_{schedule}_{pp}"

    def initialize(self, *, config: dict[str, Any]) -> None:
        self._params = dict(config)
        # H is stored separately (not part of the identity block).
        self._H = self._params.pop("H", None)
        # structural_config is consumed separately; not passed to bp_decode.
        self._structural_config = self._params.pop("structural_config", None)

    def decode(
        self,
        *,
        syndrome: Any | None = None,
        llr: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from ...qec_qldpc_codes import bp_decode

        if self._H is None:
            raise RuntimeError("Adapter not initialized (no H matrix)")

        H_used = self._H
        s_used = syndrome

        if (self._structural_config is not None
                and self._structural_config.rpc.enabled):
            from ...qec.decoder.rpc import build_rpc_augmented_system

            s_for_rpc = (
                np.asarray(s_used, dtype=np.uint8)
                if s_used is not None
                else np.zeros(self._H.shape[0], dtype=np.uint8)
            )
            H_used, s_used = build_rpc_augmented_system(
                self._H, s_for_rpc, self._structural_config.rpc,
            )

        # ── Channel geometry interventions ──
        llr_used = llr
        if self._structural_config is not None:
            s_geom = (
                np.asarray(s_used, dtype=np.uint8)
                if s_used is not None
                else np.zeros(H_used.shape[0], dtype=np.uint8)
            )
            if self._structural_config.centered_field:
                from ...qec.channel.geometry import centered_syndrome_field
                llr_used = centered_syndrome_field(H_used, s_geom)
            elif self._structural_config.pseudo_prior:
                # When only pseudo_prior is enabled (no centered_field),
                # use standard syndrome field as base LLR.
                from ...qec.channel.geometry import syndrome_field
                llr_used = syndrome_field(H_used, s_geom)

            if self._structural_config.pseudo_prior:
                from ...qec.channel.geometry import pseudo_prior_bias, apply_pseudo_prior
                bias = pseudo_prior_bias(H_used, s_geom)
                llr_used = apply_pseudo_prior(
                    llr_used, bias,
                    self._structural_config.pseudo_prior_strength,
                )

            # ── Geometry field post-processing ──
            # Applied after centered_field and pseudo_prior, only when
            # geometry interventions produced an LLR vector.
            geometry_active = (
                self._structural_config.centered_field
                or self._structural_config.pseudo_prior
            )
            if geometry_active:
                if self._structural_config.normalize_geometry:
                    std = np.std(np.asarray(llr_used, dtype=np.float64))
                    llr_used = np.asarray(llr_used, dtype=np.float64) / (std + 1e-12)
                if self._structural_config.geometry_strength != 1.0:
                    llr_used = self._structural_config.geometry_strength * np.asarray(llr_used, dtype=np.float64)

        result = bp_decode(
            H_used,
            llr_used,
            syndrome_vec=s_used,
            **self._params,
        )
        correction, iters = result[0], result[1]

        e = kwargs.get("error_vector")
        success = False
        if e is not None:
            residual = np.asarray(e) ^ np.asarray(correction)
            success = not bool(np.any(residual))

        return {
            "success": bool(success),
            "correction": correction,
            "iters": int(iters),
            "meta": {},
        }

    def measure_runtime(self, *, workload: dict[str, Any]) -> dict[str, Any]:
        from ...qec_qldpc_codes import bp_decode

        if self._H is None:
            raise RuntimeError("Adapter not initialized (no H matrix)")

        llr = workload["llr"]
        syn = workload.get("syndrome")

        def _decode() -> None:
            bp_decode(self._H, llr, syndrome_vec=syn, **self._params)

        warmup = workload.get("warmup", 5)
        runs = workload.get("runs", 30)
        mem = workload.get("measure_memory", False)

        return measure_runtime(
            _decode, warmup=warmup, runs=runs, measure_memory=mem,
        )

    def serialize_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {"adapter": "bp"}
        # Sort params for deterministic output.
        identity["params"] = {
            k: _to_python(v) for k, v in sorted(self._params.items())
        }
        return identity


def _to_python(v: Any) -> Any:
    """Convert numpy types to plain Python for serialization."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _to_python(val) for k, val in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    return v
