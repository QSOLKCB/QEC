"""UFF-to-QEC Bridge Layer (v82.0.0).

Pure adapter that converts a UFF-generated velocity curve into
QEC-compatible features and runs the full QEC pipeline (FSM → replay
→ proof → multi-agent consensus).

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

import numpy as np

from qec.controller.execution_proof import create_execution_proof
from qec.controller.multi_agent_verifier import verify_multi_agent
from qec.controller.qec_fsm import QECFSM
from qec.controller.replay_engine import verify_run
from qec.controller.trajectory_observer import analyze_trajectory


# ---------------------------------------------------------------------------
# Step 1 — Default UFF velocity model (fallback if caller has none)
# ---------------------------------------------------------------------------

def _default_v_circ(R: np.ndarray, theta: List[float]) -> np.ndarray:
    """Minimal deterministic circular-velocity model.

    v(R) = V0 * (R / Rc) / (1 + (R / Rc)^beta)^(1/beta)

    Parameters
    ----------
    R : np.ndarray
        Radial positions.
    theta : list[float]
        [V0, Rc, beta]
    """
    V0, Rc, beta = float(theta[0]), float(theta[1]), float(theta[2])
    if Rc == 0.0:
        return np.zeros_like(R, dtype=np.float64)
    x = R / Rc
    denom = (1.0 + np.power(np.abs(x), beta)) ** (1.0 / beta)
    return V0 * x / denom


# ---------------------------------------------------------------------------
# Step 2 — Feature extraction (physics → QEC features)
# ---------------------------------------------------------------------------

def extract_features(R: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """Extract QEC-compatible features from a velocity curve.

    Parameters
    ----------
    R : np.ndarray
        Radial positions.
    v : np.ndarray
        Velocity values.

    Returns
    -------
    dict
        ``energy``, ``spread``, ``zcr``, ``centroid``.
    """
    energy = float(np.mean(v))
    spread = float(np.var(v))
    gradient = np.gradient(v, R)
    sign_changes = np.diff(np.sign(gradient))
    zcr = float(np.count_nonzero(sign_changes))
    v_abs = np.abs(v)
    v_sum = float(np.sum(v_abs))
    centroid = float(np.sum(R * v_abs) / v_sum) if v_sum > 0.0 else 0.0
    return {
        "energy": energy,
        "spread": spread,
        "zcr": zcr,
        "centroid": centroid,
    }


# ---------------------------------------------------------------------------
# Step 3 — Build QEC sample input from features
# ---------------------------------------------------------------------------

def build_sample(features: Dict[str, float]) -> Dict[str, Any]:
    """Convert extracted features into a QEC FSM input dict.

    Maps feature values onto the sonic-analysis keys expected by the
    perturbation probe and FSM pipeline.

    Parameters
    ----------
    features : dict
        Output of ``extract_features``.

    Returns
    -------
    dict
        FSM-compatible input dict.
    """
    return {
        "rms_energy": features["energy"],
        "spectral_centroid_hz": features["centroid"],
        "spectral_spread_hz": features["spread"],
        "zero_crossing_rate": features["zcr"],
        "fft_top_peaks": [
            {"frequency_hz": 100.0, "magnitude": 0.5},
            {"frequency_hz": 200.0, "magnitude": 0.3},
        ],
    }


# ---------------------------------------------------------------------------
# Step 4 — Run full QEC pipeline
# ---------------------------------------------------------------------------

# Demo Ed25519 key pair (same as run_execution_proof.py)
_DEMO_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MC4CAQAwBQYDK2VwBCIEIO4Nngc2zhyTpxaDALLMVmUQ6OOjMk0eOgLjGnLLY2nN\n"
    b"-----END PRIVATE KEY-----\n"
)

_DEMO_PUBLIC_KEY_PEM = (
    b"-----BEGIN PUBLIC KEY-----\n"
    b"MCowBQYDK2VwAyEAk5fIB0Cvc5fb2v0wizvCJjQFro2sald9OS1eyUO1soM=\n"
    b"-----END PUBLIC KEY-----\n"
)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "stability_threshold": 0.5,
    "boundary_crossing_threshold": 2,
    "max_reject_cycles": 3,
    "epsilon": 1e-3,
    "n_perturbations": 9,
    "drift_threshold": 1e-4,
}


def run_uff_experiment(
    theta: List[float],
    *,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    R: np.ndarray | None = None,
    config: Dict[str, Any] | None = None,
    signer_id: str = "uff-bridge",
    private_key_pem: bytes = _DEMO_PRIVATE_KEY_PEM,
    public_key_pem: bytes = _DEMO_PUBLIC_KEY_PEM,
) -> Dict[str, Any]:
    """Run a full UFF→QEC experiment pipeline.

    Parameters
    ----------
    theta : list[float]
        UFF model parameters [V0, Rc, beta].
    v_circ_fn : callable, optional
        Velocity curve generator.  Defaults to built-in model.
    R : np.ndarray, optional
        Radial grid.  Defaults to ``np.linspace(0.1, 20, 100)``.
    config : dict, optional
        FSM configuration overrides.
    signer_id : str
        Signer identity for execution proof.
    private_key_pem / public_key_pem : bytes
        PEM-encoded Ed25519 key pair for proof signing.

    Returns
    -------
    dict
        Unified experiment artifact with keys: ``theta``, ``features``,
        ``probe``, ``invariants``, ``trajectory``, ``verification``,
        ``proof``, ``consensus``.
    """
    theta = list(theta)
    if R is None:
        R = np.linspace(0.1, 20.0, 100)
    if v_circ_fn is None:
        v_circ_fn = _default_v_circ
    merged_config = dict(_DEFAULT_CONFIG)
    if config is not None:
        merged_config.update(config)

    # --- Generate velocity curve ---
    v = v_circ_fn(R, theta)

    # --- Extract features ---
    features = extract_features(R, v)

    # --- Build QEC sample ---
    sample_input = build_sample(features)

    # --- Run FSM ---
    fsm = QECFSM(config=dict(merged_config))
    fsm_result = fsm.run(sample_input, max_steps=20)
    history = fsm_result["history"]

    # --- Trajectory analysis ---
    trajectory = analyze_trajectory(history)

    # --- Replay verification ---
    verification = verify_run(
        sample_input, history, merged_config, max_steps=20,
    )

    # --- Execution proof ---
    proof = create_execution_proof(
        verify_result=verification,
        signer_id=signer_id,
        private_key_pem=private_key_pem,
        public_key_pem=public_key_pem,
        metadata={"theta": theta},
    )

    # --- Multi-agent consensus ---
    consensus = verify_multi_agent(
        initial_input=sample_input,
        history=history,
        config=merged_config,
        proof=proof,
    )

    return {
        "theta": theta,
        "features": features,
        "probe": copy.deepcopy(fsm_result),
        "invariants": {
            "history": history,
            "final_state": fsm_result["final_state"],
        },
        "trajectory": trajectory,
        "verification": verification,
        "proof": proof,
        "consensus": consensus,
    }
