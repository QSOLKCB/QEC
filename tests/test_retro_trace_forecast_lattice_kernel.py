from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_forecast_lattice_kernel import (
    LatticeCoordinate,
    RetroTraceForecastLatticeReceipt,
    RetroTraceLatticeCell,
    RetroTraceLatticeForecastStep,
    forecast_retro_trace_lattice,
)
from qec.analysis.retro_trace_intake_bridge import build_retro_trace


def _target_receipt():
    return build_retro_target(
        target_id="lattice-target",
        isa_family="z80",
        word_size=8,
        address_width=16,
        ram_budget=64 * 1024,
        rom_budget=32 * 1024,
        cycle_budget=3_500_000,
        display_budget={"width": 256, "height": 192, "colors": 16},
        audio_budget={"channels": 3, "sample_rate": 44_100},
        input_budget={"buttons": 2, "axes": 0},
        fpu_policy="none",
        provenance="hardware",
    )


def _stable_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)),
        memory_trace=({"address": 0x4000, "op": "read", "value": 0xAB},),
        timing_trace=tuple({"cycle": 120 + 40 * idx} for idx in range(8)),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=tuple({"port": 1, "button": "A", "state": idx % 2} for idx in range(8)),
        metadata={"emulator": "retroarch", "rom_hash": "lattice-a", "version": "1.0.0"},
    )


def _sparse_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(),
        memory_trace=tuple(),
        timing_trace=tuple(),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "empty", "version": "1.0.0"},
    )


def test_deterministic_replay_identical_hash_and_bytes() -> None:
    trace = _stable_trace()
    left = forecast_retro_trace_lattice(trace, horizon=12, lattice_mode="sierpinski_3")
    right = forecast_retro_trace_lattice(trace, horizon=12, lattice_mode="sierpinski_3")
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.to_canonical_bytes() == right.to_canonical_bytes()
    assert left.stable_hash() == right.stable_hash()


@pytest.mark.parametrize("mode", ["sierpinski_3", "neutral_atom_5", "rubik_8"])
def test_coordinate_bounds_per_mode(mode: str) -> None:
    max_bound = {"sierpinski_3": 2, "neutral_atom_5": 4, "rubik_8": 7}[mode]
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=10, lattice_mode=mode)
    for step in receipt.series.steps:
        for cell in step.occupied_cells:
            assert 0 <= cell.coordinate.x <= max_bound
            assert 0 <= cell.coordinate.y <= max_bound
            assert 0 <= cell.coordinate.z <= max_bound


def test_invalid_lattice_mode_rejected() -> None:
    with pytest.raises(ValueError, match="lattice_mode must be one of"):
        forecast_retro_trace_lattice(_stable_trace(), horizon=6, lattice_mode="hex_9")


def test_monotonic_step_progression() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=24, lattice_mode="rubik_8")
    counts = [step.projected_occupancy_count for step in receipt.series.steps]
    assert counts == sorted(counts)


@pytest.mark.parametrize("mode", ["sierpinski_3", "neutral_atom_5", "rubik_8"])
def test_classification_and_bounds(mode: str) -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=18, lattice_mode=mode)
    assert receipt.summary.collapse_risk_classification in {"STABLE", "DRIFT", "UNSTABLE"}
    assert 0.0 <= receipt.summary.occupancy_dispersion <= 1.0
    assert 0.0 <= receipt.summary.locality_risk <= 1.0
    assert 0.0 <= receipt.summary.overall_stability_forecast <= 1.0


def test_sparse_trace_edge_case() -> None:
    receipt = forecast_retro_trace_lattice(_sparse_trace(), horizon=8, lattice_mode="neutral_atom_5")
    assert len(receipt.series.steps) == 8
    assert all(step.projected_occupancy_count >= 1 for step in receipt.series.steps)


def test_equivalent_trace_inputs_produce_identical_receipts() -> None:
    baseline = _stable_trace()
    equivalent = build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(reversed(tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)))),
        memory_trace=({"value": 0xAB, "op": "read", "address": 0x4000},),
        timing_trace=tuple(reversed(tuple({"cycle": 120 + 40 * idx} for idx in range(8)))),
        display_trace=({"event": "start", "scanline": 0},),
        audio_trace=({"pattern": "pulse", "channel": 1},),
        input_trace=tuple(reversed(tuple({"state": idx % 2, "button": "A", "port": 1} for idx in range(8)))),
        metadata={"version": "1.0.0", "rom_hash": "lattice-a", "emulator": "retroarch"},
    )
    left = forecast_retro_trace_lattice(baseline, horizon=14, lattice_mode="neutral_atom_5")
    right = forecast_retro_trace_lattice(equivalent, horizon=14, lattice_mode="neutral_atom_5")
    assert left.to_canonical_bytes() == right.to_canonical_bytes()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_coordinate_rejected() -> None:
    coordinate = LatticeCoordinate(x=0, y=0, z=0)
    cell_payload = {
        "lattice_mode": "sierpinski_3",
        "coordinate": coordinate.to_dict(),
        "occupancy_share": 1.0,
        "locality_pressure": 0.0,
    }
    cell = RetroTraceLatticeCell(
        lattice_mode="sierpinski_3",
        coordinate=coordinate,
        occupancy_share=1.0,
        locality_pressure=0.0,
        _stable_hash=sha256_hex(cell_payload),
    )
    with pytest.raises(ValueError, match="duplicate occupied coordinates within step"):
        RetroTraceLatticeForecastStep(
            step_index=1,
            projected_occupancy_count=2,
            projected_density_score=0.2,
            projected_locality_pressure=0.1,
            projected_dispersion_score=0.1,
            stability_score=0.8,
            occupied_cells=(cell, cell),
            _stable_hash="a" * 64,
        )


def test_non_canonical_cell_order_rejected() -> None:
    c0 = LatticeCoordinate(x=0, y=0, z=1)
    c1 = LatticeCoordinate(x=0, y=0, z=0)
    p0 = {"lattice_mode": "sierpinski_3", "coordinate": c0.to_dict(), "occupancy_share": 0.5, "locality_pressure": 0.0}
    p1 = {"lattice_mode": "sierpinski_3", "coordinate": c1.to_dict(), "occupancy_share": 0.5, "locality_pressure": 0.0}
    cell0 = RetroTraceLatticeCell("sierpinski_3", c0, 0.5, 0.0, sha256_hex(p0))
    cell1 = RetroTraceLatticeCell("sierpinski_3", c1, 0.5, 0.0, sha256_hex(p1))
    with pytest.raises(ValueError, match="occupied_cells must use canonical ordering"):
        RetroTraceLatticeForecastStep(
            step_index=1,
            projected_occupancy_count=2,
            projected_density_score=0.2,
            projected_locality_pressure=0.0,
            projected_dispersion_score=0.1,
            stability_score=0.9,
            occupied_cells=(cell0, cell1),
            _stable_hash="a" * 64,
        )


@pytest.mark.parametrize("invalid_horizon", [True, False])
def test_rejects_bool_horizon(invalid_horizon: bool) -> None:
    with pytest.raises(ValueError, match=r"horizon must be int in \[1,256\]"):
        forecast_retro_trace_lattice(_stable_trace(), horizon=invalid_horizon, lattice_mode="rubik_8")


@pytest.mark.parametrize("field,value", [("x", True), ("y", False), ("z", True)])
def test_coordinate_rejects_bool_fields(field: str, value: bool) -> None:
    kwargs = {"x": 0, "y": 0, "z": 0}
    kwargs[field] = value
    with pytest.raises(ValueError, match=f"{field} must be int"):
        LatticeCoordinate(**kwargs)


@pytest.mark.parametrize("invalid_step", [True, False])
def test_step_rejects_bool_step_index(invalid_step: bool) -> None:
    with pytest.raises(ValueError, match="step_index must be positive int"):
        RetroTraceLatticeForecastStep(
            step_index=invalid_step,
            projected_occupancy_count=0,
            projected_density_score=0.0,
            projected_locality_pressure=0.0,
            projected_dispersion_score=0.0,
            stability_score=0.0,
            occupied_cells=tuple(),
            _stable_hash="a" * 64,
        )


def test_receipt_reconstruction_rejects_summary_mismatch() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=6, lattice_mode="sierpinski_3")
    with pytest.raises(ValueError, match="summary values mismatch recomputed canonical series values"):
        RetroTraceForecastLatticeReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            series=receipt.series,
            summary=receipt.summary.__class__(
                lattice_mode=receipt.summary.lattice_mode,
                dominant_region=receipt.summary.dominant_region,
                occupancy_dispersion=receipt.summary.occupancy_dispersion,
                locality_risk=receipt.summary.locality_risk,
                overall_stability_forecast=0.0,
                collapse_risk_classification="UNSTABLE",
                _stable_hash=sha256_hex(
                    {
                        "lattice_mode": receipt.summary.lattice_mode,
                        "dominant_region": receipt.summary.dominant_region,
                        "occupancy_dispersion": receipt.summary.occupancy_dispersion,
                        "locality_risk": receipt.summary.locality_risk,
                        "overall_stability_forecast": 0.0,
                        "collapse_risk_classification": "UNSTABLE",
                    }
                ),
            ),
            _stable_hash=receipt.stable_hash(),
        )


def test_snapshot_fixture_for_deterministic_forecast() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=4, lattice_mode="sierpinski_3")
    expected_json = (
        '{"retro_trace_hash":"74ba968ac06131fceb51bc2b41bc02d51c8cec5e918adf133e7fe04d98ec9739","series":{"horizon":4,"lattice_mode":"sierpinski_3","stable_hash":"c610a4cd2aecf63428700866bfa06c7eacb56c52ae6823602b0c974788b415f4","steps":[{"occupied_cells":[{"coordinate":{"x":0,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"b7832a85830839e92b30bf6fee05fc8bb0ee88b2a3c16956d5a3b2da68b19c66"},{"coordinate":{"x":0,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"37ea9fb3efb5f6d01a6ce6810609760189663a4ade8dd174053aa82c61e888b5"},{"coordinate":{"x":0,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.071428571429,"stable_hash":"824a20e96e46de0c2f32c70ffbc0702c577ab1cb4b54df1119b1432b47270241"},{"coordinate":{"x":0,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.071428571429,"stable_hash":"bc6b5a1bb956345ab41cc896537c0ceffdba872350141da8b2a0946e51a846bd"},{"coordinate":{"x":0,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"0cdeacfd3f23ceb18fdfed362f0534ffb78680ca1ca3386ab175497fbce1ec4e"},{"coordinate":{"x":0,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.035714285714,"stable_hash":"071beb7feca1e62a0fcfa93f7e9a24b3eccddc0170c39b803d6fb8ccaff77a82"},{"coordinate":{"x":1,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.071428571429,"stable_hash":"4d67e5c18087cb34624d356b3712c5151e9d4444d5a9782b0838059125ebfe4a"},{"coordinate":{"x":1,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"c341ddedd58fe5ece21c3227f71f3608de746b31a04c6074a84b93e3253a8eac"},{"coordinate":{"x":1,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.035714285714,"stable_hash":"7d58e3af4c9df4ffb42a5e0ccb885796a84f77b507842625b07fbe28d403c7b1"},{"coordinate":{"x":1,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.035714285714,"stable_hash":"c90f05afdc3b87b506fa6ca23425e9b3076384c80178addf404bba8e2ee78a6a"},{"coordinate":{"x":1,"y":2,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.035714285714,"stable_hash":"a6e0f5b35975bfe8ecec555d2ec637681d6942c7329974e5006d28aacc5f231d"},{"coordinate":{"x":1,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.035714285714,"stable_hash":"08af403d7050212e8c7a0b930f93af3b8766474b5d00c24f7ab7856c97838152"},{"coordinate":{"x":1,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.035714285714,"stable_hash":"e4d9f1fe348dd186807322e5662845ec4b998d1a620d65bec5afc6ccefa6f59a"},{"coordinate":{"x":2,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.035714285714,"stable_hash":"0b1241d91e32faf2f3a77a1c7dc12fed2e94ebbc20cd0a5e1dfbb666ce529d12"},{"coordinate":{"x":2,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.035714285714,"stable_hash":"c42d0bd6e231521ba0c5397a890ce187bf570d92edddb1afb2ce5c3957aeb945"},{"coordinate":{"x":2,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.035714285714,"stable_hash":"5454ddc034c2e2a51c1c57ddfdb3b53204f16d89b7ea1c06210f54c4de975a57"},{"coordinate":{"x":2,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.035714285714,"stable_hash":"81f8e2dc797cfdb6ad2db4893bcad127ca0d27755a84b9db8b34c38bbfbeb10a"},{"coordinate":{"x":2,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"8534f39985fea7ff061182b9e3027fbd90f2a7237056a4c2a4e67ebd7725c0f5"},{"coordinate":{"x":2,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.071428571429,"stable_hash":"8acd077f60204b5402fd2ddc5c44ea8e13b97b13ab301dce0bfe383680711e17"}],"projected_density_score":0.703703703704,"projected_dispersion_score":0.428849902534,"projected_locality_pressure":0.559523809524,"projected_occupancy_count":19,"stability_score":0.408263714842,"stable_hash":"65a0b39d371ce795836b577567b9ddd43a351e47b6b016f2458c58daeda67995","step_index":1},{"occupied_cells":[{"coordinate":{"x":0,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.068965517241,"stable_hash":"65f5c2b24e71fec51fb8a7926b7bfb97eadf08f398abd2c70e9ee009becea9c1"},{"coordinate":{"x":0,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.068965517241,"stable_hash":"0b648f7eee49e31833f8f0e54271b37192bde7e01c5ce3b615c0624ff379e11b"},{"coordinate":{"x":0,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.068965517241,"stable_hash":"03b9b108e0d181a877685a5d3497010a5e84ffdcf0894720ccd17b6d62eb368d"},{"coordinate":{"x":0,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.068965517241,"stable_hash":"1f7e29233bde955a4f595d8e934d3f3d0537cbaf8431e817b97dbedf594db3e1"},{"coordinate":{"x":0,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.068965517241,"stable_hash":"4f3c81e777d5b9cae9230a4ddd80fbd93aeda6746637997400a322a1ebefc1f8"},{"coordinate":{"x":0,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.034482758621,"stable_hash":"6938c496f8ac602dd6146526506b01bad1917c4bbc9635626fed7ce942959241"},{"coordinate":{"x":1,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.068965517241,"stable_hash":"5a2f967269b0bad815b2cb8f4ae43c7a7c21060ace1bbd643d24724f2d5e6372"},{"coordinate":{"x":1,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.068965517241,"stable_hash":"f0c5e401d7a25898fb8f7a8133e9d8276bbe9e5495c7466f96c7d6da0d222d0f"},{"coordinate":{"x":1,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.034482758621,"stable_hash":"0c779ea2f9aeca244d1ba194e9685c74f946cf56484107f89acdc75e2d42f537"},{"coordinate":{"x":1,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.034482758621,"stable_hash":"cec9742a585a823eb31ea21390628ef0f319a17b86e7efa0cf21dd2fc86199b2"},{"coordinate":{"x":1,"y":2,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.034482758621,"stable_hash":"04a3ec2954759c6a34d9f21af2ea46073872c2fec1119073cdb1a97b46955fe1"},{"coordinate":{"x":1,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.034482758621,"stable_hash":"e93bb23328b5106f08c4dede7b8b6fbcde76b631ef1ec3a0b3ed9c713e90c348"},{"coordinate":{"x":1,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.034482758621,"stable_hash":"b8f03d406c158b6fc3f8b5ca16d57fb5385a9bb4da5eef728b3743d2de0943c8"},{"coordinate":{"x":2,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.034482758621,"stable_hash":"dc13281f5c59e2d8a7bbb5989edeb3773beca0c6cb44e3603ad4129343cd0518"},{"coordinate":{"x":2,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.034482758621,"stable_hash":"915a113e9e433203b46c7b919ed75dea37b1578a2ea8d27f81a1da3c38af5656"},{"coordinate":{"x":2,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.034482758621,"stable_hash":"b7bc2e681b689239ebc977ccfe87ff93a833d2359a65b931239e70a208bad946"},{"coordinate":{"x":2,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.034482758621,"stable_hash":"269e2360da967aa2ccab0337ab2e165f78d3d64a846c731712ecd8665fd78a70"},{"coordinate":{"x":2,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.103448275862,"stable_hash":"a0c0c07fcba0081f989a578acba6373052aa65035b53be962500e06c8718ce42"},{"coordinate":{"x":2,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.068965517241,"stable_hash":"842504b9671628d9e2e23160aed4cb429f52bcc731221894cf6baf29f8ca3a8f"}],"projected_density_score":0.703703703704,"projected_dispersion_score":0.428849902534,"projected_locality_pressure":0.557471264368,"projected_occupancy_count":19,"stability_score":0.408879478389,"stable_hash":"550712e251cbf52ea953bc76e61ea5b8b630feb263dca7e047d3afeb1fde4e1b","step_index":2},{"occupied_cells":[{"coordinate":{"x":0,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.066666666667,"stable_hash":"7664e8c281b8fba2e78086cc4ea70f01c068b5a0055680bf2dcbebf71f80c559"},{"coordinate":{"x":0,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.066666666667,"stable_hash":"e9ad1c80dbca0d5c45442f3068ba5e2996f5538e12746eca90f82ddf80b680a2"},{"coordinate":{"x":0,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.066666666667,"stable_hash":"0aab01a41442d1ffe719cc7b240069bd93dbc8c0a9da3c1811d64266d602be2f"},{"coordinate":{"x":0,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.066666666667,"stable_hash":"3ff1f222da88ef0e5e540a4805de1619a4edcf7bca6701c5467d1bc23bc56f67"},{"coordinate":{"x":0,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.066666666667,"stable_hash":"c4366c04ddf93b2cbd56ac40fa7ebe56444415ee8d05e7c8cd26a0a7eee7a936"},{"coordinate":{"x":0,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.033333333333,"stable_hash":"910c91b860890aaf84dadff086fd548b037309ad0c2590c63952e38af1ad195f"},{"coordinate":{"x":1,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.333333333333,"occupancy_share":0.033333333333,"stable_hash":"2a003685e38dfa97dd4b815805bb8e9878ef0ebb1c0c2c609ce1fce2933c737a"},{"coordinate":{"x":1,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.066666666667,"stable_hash":"5b35513f3181e1d8d73b09b6fc0e7f4edeaffbc1f2539a0783748a1a26cdf921"},{"coordinate":{"x":1,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.066666666667,"stable_hash":"e9900a13ed5affcbcde36b47be679b0091db20ae4bb958c0c9e36025ba2be619"},{"coordinate":{"x":1,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.033333333333,"stable_hash":"18e5adfd56af3ed45a78a907b3123a33fed7ac4b83636490b1e8d0a936b8b413"},{"coordinate":{"x":1,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.033333333333,"stable_hash":"c6920cc205bf1fbba3b43f0cda0524e89f2246ed903eb3753dfb7c60fce40ab6"},{"coordinate":{"x":1,"y":2,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.033333333333,"stable_hash":"e97e7ff21a7c15f19aabf5848c7a291bb6de6abba26599f948fed61b3b815387"},{"coordinate":{"x":1,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.033333333333,"stable_hash":"e0fd80584d95e4dc0c318bfdb83ca3960a4015474f1222a79399b2777463c853"},{"coordinate":{"x":1,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.033333333333,"stable_hash":"4381ab178dae7628cf9f3032bc09e9d9ea469b128701dab9530527b7e71b7800"},{"coordinate":{"x":2,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.333333333333,"occupancy_share":0.033333333333,"stable_hash":"184378f83f7ba68e90a263bc802a42d2d350d0408dd96ae1dd78a63dfe490e07"},{"coordinate":{"x":2,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.033333333333,"stable_hash":"c1ca3639ab123b6a064f69123b40c536a633d9adec577c3676e359471f0676a2"},{"coordinate":{"x":2,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.033333333333,"stable_hash":"25beb4eff1c49a95883f0a67c040b85833c2c037fa8daa6fb40db6c7c6eb1633"},{"coordinate":{"x":2,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.033333333333,"stable_hash":"bfa1ad8fcfa314f017143bcfd76386008a0127a7c7ce15d0a77ac5f8b04ad87c"},{"coordinate":{"x":2,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.1,"stable_hash":"8a9c7fae5a111ca8fab4fff5c812f32b1c71a67d911196771370df6e3e411a27"},{"coordinate":{"x":2,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.066666666667,"stable_hash":"408a6ea06f3e34b03bd381bf3782b28b03eff949d64d17ab491d5fd48f13047c"}],"projected_density_score":0.740740740741,"projected_dispersion_score":0.435964912281,"projected_locality_pressure":0.566666666666,"projected_occupancy_count":20,"stability_score":0.387675438596,"stable_hash":"c3ef629f69b0b2b590d2ab08f3afdc833a533ffcd3ecb4ad434325aadbd3e921","step_index":3},{"occupied_cells":[{"coordinate":{"x":0,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"df089ce4b31e8c908eafe11c7afc9ecf0c879c397a66969c2e5da288d7da3470"},{"coordinate":{"x":0,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"72dc251cdedaf8a6fac07f0e7db51b03c0ed3c6c93b9f95bd8c6f851f8e792ae"},{"coordinate":{"x":0,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.064516129032,"stable_hash":"7328630e8dccbcc6f30e0372a3ae8811c7184176c38d6047e39e5bce4d27943c"},{"coordinate":{"x":0,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.064516129032,"stable_hash":"477c471cacb52903242efefc856f069ff3c4819d1c64f0abc416ee1ae7eeea63"},{"coordinate":{"x":0,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"d38f95fe7f0c70e87a5a66391e43b2f0f3b369f45440ce0175d1443e3f91d151"},{"coordinate":{"x":0,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"b93d387b9747e5936e4bd5ec4727bf415eb73b75f21795995165a38340c5d653"},{"coordinate":{"x":1,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.333333333333,"occupancy_share":0.032258064516,"stable_hash":"848d60a260009bee38aeabffc2950d03fb5805f04e354f1b3208c1e31c1d9d2e"},{"coordinate":{"x":1,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.064516129032,"stable_hash":"f469962205565a879611db867a19e5789430c8cd6ef6b1c25d4a9456ed862cf9"},{"coordinate":{"x":1,"y":0,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"1b43bac09cd53a824547a9df96abed3c0a2b38bb6fec07d2e38f65999d144f31"},{"coordinate":{"x":1,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.032258064516,"stable_hash":"a983e6bfdbec613c266498986d3f57170e21c026562a101b69bf2402fc990b8e"},{"coordinate":{"x":1,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.032258064516,"stable_hash":"bfc41f8cd55a18f9241db2d75f03a716f56f2112035d52beb4485f63f3dc5cbc"},{"coordinate":{"x":1,"y":2,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.166666666667,"occupancy_share":0.032258064516,"stable_hash":"8632dfcf2ab70de2ac104424cf2d381239d4c3486408979659c50bd1a06ab27d"},{"coordinate":{"x":1,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.833333333333,"occupancy_share":0.032258064516,"stable_hash":"a68c7b61fdc7f0a853a58f9b5f5b9c463c5e205a216080b292c82e2c1bdbdb9f"},{"coordinate":{"x":1,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.032258064516,"stable_hash":"3cc711c847720e23203bc0df635b25661bfdd946f530c7f099bed533b744a80d"},{"coordinate":{"x":2,"y":0,"z":0},"lattice_mode":"sierpinski_3","locality_pressure":0.333333333333,"occupancy_share":0.032258064516,"stable_hash":"26af6caff836746bc915e33ef29e32b69fea8d8c3ec4f1e01ebe79f936cfb565"},{"coordinate":{"x":2,"y":0,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.032258064516,"stable_hash":"3d4586e2585e804ad502641e4a6d7e5c6b2135055546c9bb89a74bc535016cf7"},{"coordinate":{"x":2,"y":1,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.666666666667,"occupancy_share":0.032258064516,"stable_hash":"ac9a9bdd36bfceb65c63019ded5352da6385dd18b8954e753c555ec20b635434"},{"coordinate":{"x":2,"y":1,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.032258064516,"stable_hash":"bfbf733ac05028d43ac417acbc37f26c98e2e8e5f063efc7f3c46604ee54ec59"},{"coordinate":{"x":2,"y":2,"z":1},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.096774193548,"stable_hash":"0fb22ebedd8fff78f0470f084d56bfa25cbf46f78bad12740e46521f693c7492"},{"coordinate":{"x":2,"y":2,"z":2},"lattice_mode":"sierpinski_3","locality_pressure":0.5,"occupancy_share":0.064516129032,"stable_hash":"09a17582d20c2e7732ed7d72a2452ed2742240a2a668354a55c0d5bc8b70d26c"}],"projected_density_score":0.740740740741,"projected_dispersion_score":0.435964912281,"projected_locality_pressure":0.56451612903,"projected_occupancy_count":20,"stability_score":0.388320599887,"stable_hash":"a69779636cfbde07acc135dbd9abee6c4e5c242546a864bfac56f20232489058","step_index":4}]},"stable_hash":"21e1e2a66b8f2545a63be7451ab52b8bb2dcba12090360f77308095bd0e117ae","summary":{"collapse_risk_classification":"UNSTABLE","dominant_region":"LLL","lattice_mode":"sierpinski_3","locality_risk":0.562044467397,"occupancy_dispersion":0.432407407407,"overall_stability_forecast":0.398284807929,"stable_hash":"baeba4c337f5e76c2de41697df19fbea796ad4d35bc42ab4fdc7be7f99b7d70e"}}'
    )
    expected_hash = "21e1e2a66b8f2545a63be7451ab52b8bb2dcba12090360f77308095bd0e117ae"
    assert receipt.to_canonical_json() == expected_json
    assert receipt.stable_hash() == expected_hash


def test_replay_certification_many_rebuilds() -> None:
    trace = _stable_trace()
    receipts = [forecast_retro_trace_lattice(trace, horizon=10, lattice_mode="neutral_atom_5") for _ in range(75)]
    assert len({item.stable_hash() for item in receipts}) == 1
    assert len({item.to_canonical_json() for item in receipts}) == 1
