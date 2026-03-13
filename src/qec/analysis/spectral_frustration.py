"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    trap_threshold: float = 0.15
    r: float = 1.5
    precision: int = 12
    normalize_frustration: bool = False


@dataclass(frozen=True, eq=False)
class SpectralFrustrationResult:
    frustration_score: float
    negative_modes: int
    max_ipr: float
    transport_imbalance: float
    trap_modes: tuple[np.ndarray, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralFrustrationResult):
            return False
        if (
            self.frustration_score != other.frustration_score
            or self.negative_modes != other.negative_modes
            or self.max_ipr != other.max_ipr
            or self.transport_imbalance != other.transport_imbalance
            or len(self.trap_modes) != len(other.trap_modes)
        ):
            return False
        return all(np.array_equal(a, b) for a, b in zip(self.trap_modes, other.trap_modes))

def _to_dense_float64(A: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    if isinstance(A, np.ndarray):
        return np.asarray(A, dtype=np.float64)
    return np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64)


def build_bethe_hessian(A: np.ndarray | scipy.sparse.spmatrix, r: float) -> np.ndarray:
    """Construct Bethe-Hessian matrix ``H(r) = (r^2-1)I - rA + D``."""
    A_arr = _to_dense_float64(A)
    r_f = float(r)
    I = np.eye(A_arr.shape[0], dtype=np.float64)
    D = np.diag(np.sum(A_arr, axis=1, dtype=np.float64)).astype(np.float64, copy=False)
    H = I.copy()
    H *= (r_f * r_f - 1.0)
    H -= r_f * A_arr
    H += D
    return H


def apply_swap(A: np.ndarray | scipy.sparse.spmatrix, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
    """Apply deterministic 2-edge swap on a symmetric adjacency matrix copy."""
    A_trial = _to_dense_float64(A).copy()
    ci_i, vi_i, cj_i, vj_i = int(ci), int(vi), int(cj), int(vj)

    A_trial[ci_i, vi_i] = 0.0
    A_trial[vi_i, ci_i] = 0.0
    A_trial[cj_i, vj_i] = 0.0
    A_trial[vj_i, cj_i] = 0.0

    A_trial[ci_i, vj_i] = 1.0
    A_trial[vj_i, ci_i] = 1.0
    A_trial[cj_i, vi_i] = 1.0
    A_trial[vi_i, cj_i] = 1.0
    return A_trial


def count_negative_modes(H: np.ndarray | scipy.sparse.spmatrix) -> int:
    """Count negative modes using deterministic Sylvester inertia (LDL)."""
    H_arr = _to_dense_float64(H)
    _, D, _ = scipy.linalg.ldl(H_arr, lower=True, hermitian=True)
    return int(np.sum(np.diag(D) < 0.0))


def spectral_frustration_count(
    A: np.ndarray | scipy.sparse.spmatrix,
    r: float,
    candidate_swaps: list[tuple[int, int, int, int]] | None = None,
    flow_field: object | None = None,
) -> dict[str, object]:
    """Evaluate baseline and candidate frustration as negative-mode counts."""
    _ = flow_field
    builder = BetheHessianBuilder(A, r)
    baseline = count_negative_modes(builder.build())

    trials: list[dict[str, object]] = []
    for ci, vi, cj, vj in sorted(candidate_swaps or []):
        H_trial = builder.build_after_swap(ci, vi, cj, vj)
        neg_modes = count_negative_modes(H_trial)
        trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(neg_modes)})

    return {
        "baseline_negative_modes": int(baseline),
        "candidate_negative_modes": trials,
    }


class SpectralFrustrationAnalyzer:
    def __init__(self, config: SpectralFrustrationConfig | None = None, *, precision: int | None = None) -> None:
        if config is None:
            p = 12 if precision is None else int(precision)
            self.config = SpectralFrustrationConfig(precision=p)
        else:
            p = config.precision if precision is None else int(precision)
            self.config = SpectralFrustrationConfig(
                trap_threshold=float(config.trap_threshold),
                r=float(config.r),
                precision=p,
                normalize_frustration=bool(config.normalize_frustration),
            )

    @staticmethod
    def _tanner_adjacency(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        H_arr = _to_dense_float64(H)
        m, n = H_arr.shape
        A = np.zeros((m + n, m + n), dtype=np.float64)
        A[:m, m:] = H_arr
        A[m:, :m] = H_arr.T
        return A

    @staticmethod
    def _canonical_phase(v: np.ndarray) -> np.ndarray:
        if v.size == 0:
            return v
        idx = int(np.argmax(np.abs(v)))
        if v[idx] < 0.0:
            return -v
        return v

    def extract_trap_modes(self, H: np.ndarray | scipy.sparse.spmatrix) -> tuple[np.ndarray, ...]:
        H_arr = _to_dense_float64(H)
        m, n = H_arr.shape
        var_deg = np.sum(H_arr, axis=0, dtype=np.float64)
        chk_deg = np.sum(H_arr, axis=1, dtype=np.float64)

        mode = np.zeros(m + n, dtype=np.float64)
        for vi in range(n):
            d = float(var_deg[vi])
            mode[vi] = 1.0 / d if d > 0.0 else 0.0
        for ci in range(m):
            d = float(chk_deg[ci])
            mode[n + ci] = 1.0 / d if d > 0.0 else 0.0

        norm = float(np.linalg.norm(mode))
        if norm > 0.0:
            mode /= norm
        mode = self._canonical_phase(mode)
        return (mode,)

    def detect_trap_nodes(self, H: np.ndarray | scipy.sparse.spmatrix) -> list[int]:
        H_arr = _to_dense_float64(H)
        m, n = H_arr.shape
        modes = self.extract_trap_modes(H_arr)
        if not modes:
            return []
        amp = np.max(np.abs(np.vstack(modes)), axis=0)

        A = self._tanner_adjacency(H_arr)
        visited = np.zeros(A.shape[0], dtype=bool)
        components: list[list[int]] = []
        for start in range(A.shape[0]):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp: list[int] = []
            while stack:
                u = stack.pop()
                comp.append(int(u))
                nbrs = np.nonzero(A[u])[0]
                for v in sorted(int(x) for x in nbrs.tolist()):
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(sorted(comp))

        ranked = sorted(
            components,
            key=lambda c: (-float(np.mean(amp[c])) if c else 0.0, c[0] if c else -1),
        )
        target = ranked[0] if ranked else []
        target_amp = amp[target] if target else np.zeros(0, dtype=np.float64)
        thresh = float(self.config.trap_threshold) * float(np.max(target_amp) if target_amp.size else 0.0)
        nodes = [int(i) for i in target if float(amp[i]) >= thresh]
        return sorted(nodes)

    def compute_frustration(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        *,
        flow_field: object | None = None,
    ) -> SpectralFrustrationResult:
        _ = flow_field
        A = self._tanner_adjacency(H)
        B = build_bethe_hessian(A, self.config.r)
        neg = count_negative_modes(B)
        modes = self.extract_trap_modes(H)

        max_ipr = 0.0
        for mode in modes:
            p = np.abs(mode)
            denom = float(np.sum(p))
            if denom > 0.0:
                q = p / denom
                max_ipr = max(max_ipr, float(np.sum(q * q)))

        nodes = self.detect_trap_nodes(H)
        transport_imbalance = float(len(nodes)) / float(A.shape[0] if A.shape[0] else 1)
        cycle_proxy = float(np.sum(_to_dense_float64(H), dtype=np.float64)) / float(_to_dense_float64(H).shape[0] + _to_dense_float64(H).shape[1])
        frustration = float(neg) + max_ipr + transport_imbalance + cycle_proxy
        if self.config.normalize_frustration and A.shape[0] > 0:
            frustration /= float(A.shape[0])

        p = self.config.precision
        return SpectralFrustrationResult(
            frustration_score=round(float(frustration), p),
            negative_modes=int(neg),
            max_ipr=round(float(max_ipr), p),
            transport_imbalance=round(float(transport_imbalance), p),
            trap_modes=tuple(np.asarray(m, dtype=np.float64).copy() for m in modes),
        )

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        *,
        r: float,
        swaps: list[tuple[int, int, int, int]],
        use_cache: bool = True,
        flow_field: object | None = None,
    ) -> dict[str, object]:
        _ = use_cache
        return spectral_frustration_count(A, r=r, candidate_swaps=swaps, flow_field=flow_field)
