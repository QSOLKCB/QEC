import numpy as np

from qec.analysis.eigenmode_mutation import (
    adjacency_list_from_H,
    build_bethe_hessian,
    extract_unstable_modes,
)


def test_bethe_hessian_and_unstable_modes_are_deterministic() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    B1, r1 = build_bethe_hessian(H)
    B2, r2 = build_bethe_hessian(H)

    assert np.isclose(r1, r2)
    assert np.allclose(B1.toarray(), B2.toarray())

    modes1 = extract_unstable_modes(B1, num_modes=4)
    modes2 = extract_unstable_modes(B1, num_modes=4)

    assert len(modes1) == len(modes2)
    for a, b in zip(modes1, modes2):
        assert np.isclose(a["eigenvalue"], b["eigenvalue"])
        assert np.isclose(a["severity"], b["severity"])
        assert a["support_nodes"] == b["support_nodes"]


def test_adjacency_list_is_sorted_and_offset() -> None:
    H = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    adj = adjacency_list_from_H(H)
    assert adj == [(0, 2), (1, 2), (1, 3)]


def test_unstable_modes_include_sorted_eigen_rank() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    B, _ = build_bethe_hessian(H)
    modes = extract_unstable_modes(B, num_modes=4)
    ranks = [int(mode["eigen_rank"]) for mode in modes]
    assert ranks == sorted(ranks)
