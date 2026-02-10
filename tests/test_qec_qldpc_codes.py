"""
Tests for protograph-based quantum LDPC codes (Komoto-Kasai 2025).
"""

import pytest
import numpy as np

# Import directly from the module to avoid pulling in qutip via __init__
from src.qec_qldpc_codes import (
    GF2e,
    ProtographPair,
    build_protograph_pair,
    QuantumLDPCCode,
    JointSPDecoder,
    ConstructionInvariantError,
    depolarizing_channel,
    hashing_bound,
    hashing_bound_threshold,
    simulate_frame_error_rate,
    create_code,
    PREDEFINED_CODES,
    _verify_orthogonality_gf,
)


# ───────────────────────────────────────────────────────────────────
# GF(2^e) Arithmetic
# ───────────────────────────────────────────────────────────────────

class TestGF2e:
    def test_gf4_basic(self):
        gf = GF2e(2)
        assert gf.q == 4
        assert gf.e == 2
        assert gf.add(0, 0) == 0
        assert gf.mul(1, 1) == 1
        assert gf.mul(1, 3) == 3

    def test_gf8_basic(self):
        gf = GF2e(3)
        assert gf.q == 8
        assert gf.e == 3

    def test_gf16_basic(self):
        gf = GF2e(4)
        assert gf.q == 16
        assert gf.e == 4

    def test_unsupported_degree(self):
        with pytest.raises(ValueError, match="not supported"):
            GF2e(5)

    def test_mul_commutative(self):
        gf = GF2e(3)
        for a in range(gf.q):
            for b in range(gf.q):
                assert gf.mul(a, b) == gf.mul(b, a)

    def test_mul_by_zero(self):
        gf = GF2e(3)
        for a in range(gf.q):
            assert gf.mul(a, 0) == 0
            assert gf.mul(0, a) == 0

    def test_mul_by_one(self):
        gf = GF2e(3)
        for a in range(gf.q):
            assert gf.mul(a, 1) == a

    def test_inverse(self):
        gf = GF2e(3)
        for a in range(1, gf.q):
            inv_a = gf.inv(a)
            assert gf.mul(a, inv_a) == 1

    def test_inverse_of_zero_raises(self):
        gf = GF2e(3)
        with pytest.raises(ZeroDivisionError):
            gf.inv(0)

    def test_add_self_is_zero(self):
        """In GF(2^e), a + a = 0 (characteristic 2)."""
        gf = GF2e(3)
        for a in range(gf.q):
            assert gf.add(a, a) == 0

    def test_companion_matrix_zero(self):
        gf = GF2e(3)
        C = gf.companion_matrix(0)
        assert np.all(C == 0)

    def test_companion_matrix_identity(self):
        gf = GF2e(3)
        C = gf.companion_matrix(1)
        assert np.array_equal(C, np.eye(3, dtype=np.uint8))

    def test_companion_preserves_multiplication(self):
        """C(a*b) = C(a) @ C(b) mod 2."""
        gf = GF2e(3)
        for a in range(1, gf.q):
            for b in range(1, gf.q):
                Ca = gf.companion_matrix(a)
                Cb = gf.companion_matrix(b)
                Cab = gf.companion_matrix(gf.mul(a, b))
                product = (Ca.astype(int) @ Cb.astype(int)) % 2
                assert np.array_equal(product, Cab), (
                    f"C({a})*C({b}) != C({gf.mul(a,b)})"
                )

    def test_companion_preserves_addition(self):
        """C(a+b) = C(a) + C(b) mod 2."""
        gf = GF2e(3)
        for a in range(gf.q):
            for b in range(gf.q):
                Ca = gf.companion_matrix(a)
                Cb = gf.companion_matrix(b)
                Cab = gf.companion_matrix(gf.add(a, b))
                summ = (Ca.astype(int) + Cb.astype(int)) % 2
                assert np.array_equal(summ, Cab)

    def test_nonzero_elements(self):
        gf = GF2e(3)
        nz = gf.nonzero_elements()
        assert len(nz) == 7
        assert 0 not in nz
        assert all(1 <= x <= 7 for x in nz)


# ───────────────────────────────────────────────────────────────────
# Protograph Pair
# ───────────────────────────────────────────────────────────────────

class TestProtographPair:
    def test_rate_computation(self):
        gf = GF2e(3)
        proto = build_protograph_pair(J=1, L=4, gf=gf, seed=42)
        assert abs(proto.rate - 0.50) < 1e-10

    def test_rate_075(self):
        gf = GF2e(3)
        proto = build_protograph_pair(J=1, L=8, gf=gf, seed=42)
        assert abs(proto.rate - 0.75) < 1e-10

    def test_orthogonality_single_row(self):
        """B_X . B_Z^T = 0 over GF(2^e) for J=1."""
        gf = GF2e(3)
        proto = build_protograph_pair(J=1, L=4, gf=gf, seed=42)
        assert _verify_orthogonality_gf(proto.B_X, proto.B_Z, gf)

    def test_orthogonality_multi_row(self):
        """B_X . B_Z^T = 0 over GF(2^e) for J=2."""
        gf = GF2e(3)
        proto = build_protograph_pair(J=2, L=10, gf=gf, seed=42)
        assert _verify_orthogonality_gf(proto.B_X, proto.B_Z, gf)

    def test_shapes(self):
        gf = GF2e(3)
        proto = build_protograph_pair(J=2, L=10, gf=gf, seed=42)
        assert proto.B_X.shape == (2, 10)
        assert proto.B_Z.shape == (2, 10)

    def test_self_orthogonal(self):
        """B_X == B_Z (self-orthogonal construction)."""
        gf = GF2e(3)
        proto = build_protograph_pair(J=1, L=4, gf=gf, seed=42)
        assert np.array_equal(proto.B_X, proto.B_Z)

    @pytest.mark.parametrize("J,L", [(1, 4), (1, 8), (2, 10), (2, 6), (3, 12)])
    def test_orthogonality_various_sizes(self, J, L):
        """Orthogonality holds across multiple (J, L) combinations."""
        gf = GF2e(3)
        proto = build_protograph_pair(J=J, L=L, gf=gf, seed=42)
        assert _verify_orthogonality_gf(proto.B_X, proto.B_Z, gf)


# ───────────────────────────────────────────────────────────────────
# Quantum LDPC Code — construction invariants
# ───────────────────────────────────────────────────────────────────

class TestQuantumLDPCCode:
    @pytest.fixture
    def small_code(self):
        return create_code('rate_0.50', lifting_size=8, seed=42)

    def test_create_rate_050(self, small_code):
        code = small_code
        assert code.n > 0
        assert code.k >= 0
        assert code.n == code.field.e * 8 * 4  # e * P * L = 3*8*4 = 96

    def test_create_rate_075(self):
        code = create_code('rate_0.75', lifting_size=8, seed=42)
        assert code.n == 3 * 8 * 8  # e*P*L = 192
        assert code.rate > 0.0

    def test_css_orthogonality(self, small_code):
        """H_X @ H_Z^T = 0 mod 2 — the non-negotiable invariant."""
        assert small_code.verify_css_orthogonality()

    def test_css_orthogonality_rate075(self):
        code = create_code('rate_0.75', lifting_size=8, seed=42)
        assert code.verify_css_orthogonality()

    def test_css_orthogonality_rate060(self):
        code = create_code('rate_0.60', lifting_size=8, seed=42)
        assert code.verify_css_orthogonality()

    def test_parity_check_shapes(self, small_code):
        code = small_code
        assert code.H_X.shape == (1 * 3 * 8, 4 * 3 * 8)
        assert code.H_Z.shape == (1 * 3 * 8, 4 * 3 * 8)

    def test_no_all_zero_rows(self, small_code):
        """Every check row should participate in at least one variable."""
        code = small_code
        assert np.all(code.H_X.sum(axis=1) > 0)
        assert np.all(code.H_Z.sum(axis=1) > 0)

    def test_no_all_zero_columns(self, small_code):
        """Every variable should participate in at least one check."""
        code = small_code
        assert np.all(code.H_X.sum(axis=0) > 0)
        assert np.all(code.H_Z.sum(axis=0) > 0)

    def test_syndrome_zero_error(self, small_code):
        """Zero error vector should give zero syndrome."""
        code = small_code
        zero_err = np.zeros(code.n, dtype=np.uint8)
        assert np.all(code.syndrome_X(zero_err) == 0)
        assert np.all(code.syndrome_Z(zero_err) == 0)

    def test_syndrome_nonzero(self, small_code):
        """A single-qubit error should (typically) produce a nonzero syndrome."""
        code = small_code
        err = np.zeros(code.n, dtype=np.uint8)
        err[0] = 1
        sx = code.syndrome_X(err)
        sz = code.syndrome_Z(err)
        assert np.any(sx != 0) or np.any(sz != 0)

    def test_determinism(self):
        """Same seed → identical matrices."""
        c1 = create_code('rate_0.50', lifting_size=8, seed=99)
        c2 = create_code('rate_0.50', lifting_size=8, seed=99)
        assert np.array_equal(c1.H_X, c2.H_X)
        assert np.array_equal(c1.H_Z, c2.H_Z)

    def test_unknown_code_name(self):
        with pytest.raises(ValueError, match="Unknown code"):
            create_code('rate_0.99')

    def test_repr(self, small_code):
        r = repr(small_code)
        assert 'QuantumLDPCCode' in r
        assert 'rate=' in r


# ───────────────────────────────────────────────────────────────────
# Joint Sum-Product Decoder
# ───────────────────────────────────────────────────────────────────

class TestJointSPDecoder:
    @pytest.fixture
    def code_and_decoder(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        dec = JointSPDecoder(code, max_iter=30)
        return code, dec

    def test_decode_zero_syndrome(self, code_and_decoder):
        """Zero syndrome → zero correction."""
        code, dec = code_and_decoder
        sx = np.zeros(code.m_X, dtype=np.uint8)
        sz = np.zeros(code.m_Z, dtype=np.uint8)
        x_hat, z_hat, converged = dec.decode(sx, sz, p_phys=0.01)
        assert converged
        assert np.all(x_hat == 0)
        assert np.all(z_hat == 0)

    def test_decode_returns_correct_shapes(self, code_and_decoder):
        code, dec = code_and_decoder
        sx = np.zeros(code.m_X, dtype=np.uint8)
        sz = np.zeros(code.m_Z, dtype=np.uint8)
        x_hat, z_hat, _ = dec.decode(sx, sz, p_phys=0.01)
        assert x_hat.shape == (code.n,)
        assert z_hat.shape == (code.n,)

    def test_decode_low_noise(self, code_and_decoder):
        """At very low noise, decoder should usually succeed."""
        code, dec = code_and_decoder
        rng = np.random.default_rng(123)
        successes = 0
        trials = 20
        for _ in range(trials):
            x_err, z_err = depolarizing_channel(code.n, 0.005, rng)
            sx = code.syndrome_X(z_err)
            sz = code.syndrome_Z(x_err)
            x_hat, z_hat, converged = dec.decode(sx, sz, p_phys=0.005)
            if converged:
                successes += 1
        assert successes > 0


# ───────────────────────────────────────────────────────────────────
# Depolarizing Channel
# ───────────────────────────────────────────────────────────────────

class TestDepolarizingChannel:
    def test_zero_noise(self):
        x, z = depolarizing_channel(100, 0.0, rng=np.random.default_rng(0))
        assert np.all(x == 0)
        assert np.all(z == 0)

    def test_shapes(self):
        x, z = depolarizing_channel(50, 0.1, rng=np.random.default_rng(0))
        assert x.shape == (50,)
        assert z.shape == (50,)

    def test_error_rate_statistics(self):
        """Error rate should be roughly consistent with p."""
        rng = np.random.default_rng(42)
        n = 100000
        p = 0.12
        x, z = depolarizing_channel(n, p, rng)
        expected_x_rate = 2 * p / 3
        actual_x_rate = x.sum() / n
        assert abs(actual_x_rate - expected_x_rate) < 0.01

    def test_dtype(self):
        x, z = depolarizing_channel(10, 0.1, rng=np.random.default_rng(0))
        assert x.dtype == np.uint8
        assert z.dtype == np.uint8


# ───────────────────────────────────────────────────────────────────
# Hashing Bound — test INVARIANTS and TRENDS, not fantasy thresholds
# ───────────────────────────────────────────────────────────────────

class TestHashingBound:
    def test_zero_noise(self):
        assert hashing_bound(0.0) == 1.0

    def test_max_noise(self):
        assert hashing_bound(0.75) == 0.0

    def test_non_negative(self):
        for p in np.linspace(0, 0.75, 50):
            assert hashing_bound(p) >= 0.0

    def test_monotone_decreasing(self):
        ps = np.linspace(0.001, 0.74, 100)
        bounds = [hashing_bound(p) for p in ps]
        for i in range(len(bounds) - 1):
            assert bounds[i] >= bounds[i + 1] - 1e-10

    def test_crosses_zero_near_0189(self):
        """The hashing bound reaches 0 near p ~ 0.1893."""
        assert hashing_bound(0.18) > 0.0
        assert hashing_bound(0.20) < 0.05

    def test_threshold_bounded_by_hashing(self):
        """Threshold for any rate must be <= hashing bound zero-crossing."""
        p_zero = hashing_bound_threshold(0.0)  # ~0.189
        for rate in [0.50, 0.60, 0.75]:
            p_th = hashing_bound_threshold(rate)
            assert 0.0 < p_th <= p_zero + 1e-6

    def test_threshold_decreases_with_rate(self):
        """Higher rate → lower tolerable noise (monotone)."""
        p_50 = hashing_bound_threshold(0.50)
        p_60 = hashing_bound_threshold(0.60)
        p_75 = hashing_bound_threshold(0.75)
        assert p_50 > p_60 > p_75 > 0.0

    def test_threshold_rate_zero(self):
        p_th = hashing_bound_threshold(0.0)
        assert p_th > 0.18

    def test_threshold_rate_one(self):
        p_th = hashing_bound_threshold(1.0)
        assert p_th == 0.0

    def test_hashing_bound_at_threshold_equals_rate(self):
        """hashing_bound(threshold(R)) ~ R (by definition of bisection)."""
        for rate in [0.25, 0.50, 0.75]:
            p_th = hashing_bound_threshold(rate)
            assert abs(hashing_bound(p_th) - rate) < 1e-6


# ───────────────────────────────────────────────────────────────────
# Simulation
# ───────────────────────────────────────────────────────────────────

class TestSimulation:
    def test_simulation_returns_dict(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        dec = JointSPDecoder(code, max_iter=10)
        res = simulate_frame_error_rate(code, dec, 0.01, n_frames=5, seed=42)
        assert 'fer' in res
        assert 'fer_x' in res
        assert 'fer_z' in res
        assert 'n_physical' in res
        assert 'n_logical' in res
        assert 'rate' in res
        assert res['n_physical'] == code.n

    def test_simulation_fer_bounds(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        dec = JointSPDecoder(code, max_iter=10)
        res = simulate_frame_error_rate(code, dec, 0.01, n_frames=10, seed=42)
        assert 0.0 <= res['fer'] <= 1.0
        assert 0.0 <= res['fer_x'] <= 1.0
        assert 0.0 <= res['fer_z'] <= 1.0

    def test_fer_decreases_with_noise(self):
        """Basic competence: FER should not increase when noise decreases."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        dec = JointSPDecoder(code, max_iter=20)
        r_high = simulate_frame_error_rate(code, dec, 0.10, n_frames=50, seed=42)
        r_low = simulate_frame_error_rate(code, dec, 0.005, n_frames=50, seed=42)
        # Low noise FER should be <= high noise FER (allow small stat noise)
        assert r_low['fer'] <= r_high['fer'] + 0.15


# ───────────────────────────────────────────────────────────────────
# Predefined Codes — structural invariants on every configuration
# ───────────────────────────────────────────────────────────────────

class TestPredefinedCodes:
    @pytest.mark.parametrize("name", list(PREDEFINED_CODES.keys()))
    def test_create_all_predefined(self, name):
        """Construction must succeed and CSS orthogonality must hold."""
        code = create_code(name, lifting_size=8, seed=42)
        assert code.n > 0
        assert code.k >= 0
        # The constructor itself asserts orthogonality, but be explicit:
        assert code.verify_css_orthogonality()

    @pytest.mark.parametrize("name", list(PREDEFINED_CODES.keys()))
    def test_predefined_has_description(self, name):
        assert 'description' in PREDEFINED_CODES[name]
        assert len(PREDEFINED_CODES[name]['description']) > 10
