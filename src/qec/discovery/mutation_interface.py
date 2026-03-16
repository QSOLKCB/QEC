"""Minimal mutation interface for spectral threshold search operators."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12


class SpectralMutationOperator:
    """Base class for deterministic spectral mutation operators."""

    def mutate(self, H: np.ndarray, spectrum_info: dict[str, Any], config: Any) -> tuple[np.ndarray, list[dict[str, Any]], str, dict[str, Any]]:
        raise NotImplementedError


class BeamMutation(SpectralMutationOperator):
    """Adapter for beam mutation operator."""

    def __init__(self, beam_mutator: Any) -> None:
        self._beam_mutator = beam_mutator

    def mutate(self, H: np.ndarray, spectrum_info: dict[str, Any], config: Any) -> tuple[np.ndarray, list[dict[str, Any]], str, dict[str, Any]]:
        _ = spectrum_info
        adaptive_steps = int(config.get("adaptive_steps", 1))
        h_beam, ops_beam = self._beam_mutator.mutate(H, steps=adaptive_steps)
        return np.asarray(h_beam, dtype=np.float64), ops_beam, "beam", {}


class NBEigenvectorFlowMutation(SpectralMutationOperator):
    """Adapter for NB flow mutation with single/multi mode support."""

    def __init__(self, flow_mutator: Any) -> None:
        self._flow_mutator = flow_mutator

    def mutate(self, H: np.ndarray, spectrum_info: dict[str, Any], config: Any) -> tuple[np.ndarray, list[dict[str, Any]], str, dict[str, Any]]:
        if bool(config.get("enable_multi_mode_nb_mutation", False)):
            eigenvectors = np.asarray(spectrum_info.get("eigenvectors", np.asarray([], dtype=np.float64)))
            mode_count = int(config.get("mode_count", 0))
            if mode_count <= 0 or eigenvectors.ndim != 2 or eigenvectors.shape[0] == 0:
                return np.asarray(H, dtype=np.float64), [], "nb_flow_multi_mode", {}

            memory = spectrum_info.get("mutation_memory")
            if bool(config.get("enable_spectral_mutation_memory", False)) and memory is not None:
                mode_weights = memory.compute_weights(mode_count)
            else:
                mode_weights = np.round(np.full(mode_count, 1.0 / float(mode_count), dtype=np.float64), _ROUND)

            multi_flow = self._flow_mutator.compute_multi_mode_flow(eigenvectors[:, :mode_count], mode_weights)
            selected_mode = int(np.argmax(mode_weights)) if mode_weights.size > 0 else 0
            h_flow, flow_info = self._flow_mutator.mutate_with_flow(H, multi_flow, mode_index=selected_mode)
            flow_info["mode_weights"] = [round(float(w), _ROUND) for w in mode_weights.tolist()]
            return np.asarray(h_flow, dtype=np.float64), [], "nb_flow_multi_mode", flow_info

        leading_vector = np.asarray(spectrum_info.get("leading_vector", np.asarray([], dtype=np.float64)), dtype=np.float64)
        h_flow, flow_info = self._flow_mutator.mutate(H, leading_vector, mode_index=0)
        return np.asarray(h_flow, dtype=np.float64), [], "nb_flow", flow_info
