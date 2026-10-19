# SPDX-License-Identifier: MPL-2.0
"""Classical replay conformance; toy callbacks are not decoder benchmarks."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import itertools
import json

import pytest

from qec.experiments.syndrome_delivery import (
    CLAIM_BOUNDARY,
    MAX_FRAMES,
    MAX_PARTITIONS,
    MAX_TICKS,
    MAX_WIDTH,
    DeliveryPlan,
    SyndromeTrace,
    run_syndrome_delivery,
    validate_syndrome_delivery_receipt,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256


def toy_correction(frame):
    """Deterministic exact-binary contract fixture, not a QEC decoder."""
    return (sum(frame) % 2, frame[0])


def example():
    trace = SyndromeTrace(4, ((1, 0, 1, 0), (0, 1, 0, 0), (1, 1, 1, 0)))
    plan = DeliveryPlan((2, 2), (3, 1), ((3, 0), (-3, -1), (-2, 0)),
                        frame_period_ticks=2, deadline_ticks=6, decode_ticks=1)
    return trace, plan


def run(trace=None, plan=None, decoder=toy_correction, **kwargs):
    trace, plan = (trace, plan) if trace is not None else example()
    return run_syndrome_delivery(trace, plan, decoder, correction_width=kwargs.pop("correction_width", 2), **kwargs)


def test_reordered_fragments_reassemble_before_same_unchanged_callback():
    trace, plan = example()
    calls = []

    def observed(frame):
        assert type(frame) is tuple
        calls.append(frame)
        return toy_correction(frame)

    receipt = run(trace, plan, observed)
    assert calls == list(trace.frames) * 2
    assert receipt["chronological_reassembled_frames"] == [list(frame) for frame in trace.frames]
    assert receipt["baseline_corrections"] == [[0, 1], [1, 0], [1, 1]]
    assert receipt["baseline_corrections"] == receipt["replay_corrections"]
    assert receipt["inputs_match"] is True
    assert receipt["outputs_match"] is True
    assert receipt["comparison_outcome"] == "match"
    assert receipt["compared_frame_count"] == 3
    assert [(row["arrival_tick"], row["frame_index"], row["partition_index"])
            for row in receipt["arrivals"]] == [(1, 0, 1), (2, 1, 0), (2, 1, 1),
                                                (5, 2, 0), (5, 2, 1), (6, 0, 0)]
    assert [row["all_fragments_arrived_tick"] for row in receipt["timings"]] == [6, 2, 5]
    assert [row["chronological_reassembly_tick"] for row in receipt["timings"]] == [6, 6, 6]
    assert [row["replay_finish_tick"] for row in receipt["timings"]] == [7, 8, 9]
    assert receipt["baseline_deadline_miss_count"] == 0
    assert receipt["replay_deadline_miss_count"] == 1


def test_arrival_ties_use_frame_then_partition_index():
    trace = SyndromeTrace(3, ((1, 0, 1), (0, 1, 0)))
    plan = DeliveryPlan((1, 1, 1), (2, 2, 2), ((0, 0, 0), (-1, -1, -1)))
    receipt = run(trace, plan)
    assert [(row["frame_index"], row["partition_index"]) for row in receipt["arrivals"]] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert all(row["arrival_tick"] == 2 for row in receipt["arrivals"])


@pytest.mark.parametrize("deadline, missed", [(0, 1), (4, 1), (5, 0), (6, 0)])
def test_exact_deadline_boundary(deadline, missed):
    trace = SyndromeTrace(1, ((1,),))
    plan = DeliveryPlan((1,), (3,), deadline_ticks=deadline, decode_ticks=2)
    receipt = run(trace, plan)
    assert receipt["timings"][0]["replay_finish_tick"] == 5
    assert receipt["replay_deadline_miss_count"] == missed


def test_serial_baseline_queue_and_replay_queue_are_explicit_model_costs():
    trace = SyndromeTrace(1, ((0,), (1,), (0,)))
    plan = DeliveryPlan((1,), (0,), frame_period_ticks=1, deadline_ticks=3, decode_ticks=2)
    receipt = run(trace, plan)
    assert [row["baseline_finish_tick"] for row in receipt["timings"]] == [2, 4, 6]
    assert [row["replay_finish_tick"] for row in receipt["timings"]] == [2, 4, 6]
    assert receipt["baseline_deadline_miss_count"] == receipt["replay_deadline_miss_count"] == 1


def test_zero_delay_zero_cost_is_on_time():
    receipt = run(SyndromeTrace(2, ((0, 1), (1, 0))), DeliveryPlan((1, 1), (0, 0)))
    assert receipt["baseline_deadline_miss_count"] == receipt["replay_deadline_miss_count"] == 0


def test_stateful_callback_mismatch_is_computed_from_actual_outputs():
    trace, plan = example()
    calls = 0

    def stateful(frame):
        nonlocal calls
        calls += 1
        return (int(calls > len(trace.frames)), 0)

    receipt = run(trace, plan, stateful)
    assert calls == 6
    assert receipt["inputs_match"] is True
    assert receipt["outputs_match"] is False
    assert receipt["output_mismatch_indices"] == [0, 1, 2]
    assert receipt["comparison_outcome"] == "mismatch"
    assert receipt["baseline_corrections"] == [[0, 0]] * 3
    assert receipt["replay_corrections"] == [[1, 0]] * 3


def test_empty_trace_has_no_decode_or_empirical_evidence():
    def unused(frame):
        raise AssertionError("must not run")

    receipt = run(SyndromeTrace(2, ()), DeliveryPlan((1, 1), (0, 2)), unused)
    assert receipt["comparison_outcome"] == "empty_trace"
    assert receipt["compared_frame_count"] == 0
    assert receipt["outputs_match"] is True  # Explicitly vacuous.
    for key in ("arrivals", "timings", "baseline_corrections", "replay_corrections",
                "chronological_reassembled_frames", "output_mismatch_indices"):
        assert receipt[key] == []


def test_canonical_hashes_determinism_and_json_roundtrip():
    trace, plan = example()
    first = run(trace, plan)
    assert canonical_bytes(first) == canonical_bytes(run(trace, plan))
    assert first["input_sha256"] == canonical_sha256(first["input"])
    assert first["sha256"] == canonical_sha256({key: value for key, value in first.items() if key != "sha256"})
    validate_syndrome_delivery_receipt(json.loads(json.dumps(first)), trace, plan, toy_correction,
                                       correction_width=2)


def test_equivalent_container_inputs_and_explicit_zero_jitter_are_canonical():
    trace = SyndromeTrace(2, [[1, 0], [0, 1]])
    plan = DeliveryPlan([1, 1], [0, 0])
    explicit = DeliveryPlan((1, 1), (0, 0), ((0, 0), (0, 0)))
    assert canonical_bytes(run(trace, plan)) == canonical_bytes(run(trace, explicit))


def test_inputs_are_immutable_snapshots_and_outputs_do_not_alias_them():
    frames = [[1, 0]]
    widths, delays, jitter = [1, 1], [2, 0], [[0, 0]]
    trace = SyndromeTrace(2, frames)
    plan = DeliveryPlan(widths, delays, jitter)
    frames[0][0], widths[0], delays[0], jitter[0][0] = 0, 9, 4, 8
    receipt = run(trace, plan)
    assert trace.frames == ((1, 0),)
    assert plan.partition_widths == (1, 1)
    assert plan.base_delay_ticks == (2, 0)
    assert plan.jitter_ticks == ((0, 0),)
    receipt["input"]["frames"][0][0] = 0
    assert trace.frames == ((1, 0),)
    with pytest.raises(FrozenInstanceError):
        trace.frames = ()
    with pytest.raises(FrozenInstanceError):
        plan.decode_ticks = 2


@pytest.mark.parametrize("width, frames", [
    (True, ((0,),)), (0, ()), (-1, ()), (1.0, ((0,),)), (MAX_WIDTH + 1, ()),
    (1, "0"), (1, iter([(0,)])), (1, {0: (0,)}),
    (1, ((True,),)), (1, ((False,),)), (1, ((0.0,),)), (1, (("1",),)),
    (1, ((2,),)), (1, ((-1,),)), (1, ((float("nan"),),)),
    (1, ((float("inf"),),)), (1, ((0, 1),)), (2, ((0,),)), (1, ((),)),
    (1, ((0,),) * (MAX_FRAMES + 1)), (MAX_WIDTH, ((0,),) * MAX_FRAMES),
])
def test_strict_bounded_trace_inputs(width, frames):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        SyndromeTrace(width, frames)


@pytest.mark.parametrize("kwargs", [
    {"partition_widths": ()}, {"partition_widths": (0,)}, {"partition_widths": (True,)},
    {"partition_widths": (1.0,)}, {"partition_widths": (MAX_WIDTH + 1,)},
    {"partition_widths": (MAX_WIDTH, 1), "base_delay_ticks": (0, 0)},
    {"partition_widths": (1,) * (MAX_PARTITIONS + 1)},
    {"base_delay_ticks": ()}, {"base_delay_ticks": (0, 0)}, {"base_delay_ticks": (-1,)},
    {"base_delay_ticks": (True,)}, {"base_delay_ticks": (0.0,)},
    {"base_delay_ticks": (MAX_TICKS + 1,)},
    {"jitter_ticks": ((1, 2),)}, {"jitter_ticks": ((True,),)}, {"jitter_ticks": ((0.5,),)},
    {"jitter_ticks": ((-1,),)}, {"jitter_ticks": ((MAX_TICKS + 1,),)},
    {"base_delay_ticks": (MAX_TICKS,), "jitter_ticks": ((1,),)},
    {"jitter_ticks": ((0,),) * (MAX_FRAMES + 1)},
    {"partition_widths": (1,) * 256, "base_delay_ticks": (0,) * 256,
     "jitter_ticks": ((0,) * 256,) * 257},
    {"frame_period_ticks": 0}, {"frame_period_ticks": True}, {"frame_period_ticks": 1.0},
    {"deadline_ticks": -1}, {"deadline_ticks": float("inf")}, {"deadline_ticks": True},
    {"decode_ticks": -1}, {"decode_ticks": True}, {"decode_ticks": MAX_TICKS + 1},
])
def test_strict_bounded_plan_inputs(kwargs):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        DeliveryPlan(**{"partition_widths": (1,), "base_delay_ticks": (0,), **kwargs})


@pytest.mark.parametrize("trace, plan", [
    (SyndromeTrace(2, ((1, 0),)), DeliveryPlan((1,), (0,))),
    (SyndromeTrace(1, ((1,), (0,))), DeliveryPlan((1,), (0,), ((0,),))),
    (SyndromeTrace(1, ()), DeliveryPlan((1,), (0,), ((0,),))),
    (SyndromeTrace(256, ((0,) * 256,) * 257), DeliveryPlan((1,) * 256, (0,) * 256)),
    (SyndromeTrace(1, ((1,), (0,))), DeliveryPlan((1,), (0,), frame_period_ticks=MAX_TICKS, deadline_ticks=1)),
    (SyndromeTrace(1, ((1,), (0,))), DeliveryPlan((1,), (MAX_TICKS,))),
    (SyndromeTrace(1, ((1,), (0,))), DeliveryPlan((1,), (0,), decode_ticks=MAX_TICKS)),
    (SyndromeTrace(1, ((1,),)), DeliveryPlan((1,), (MAX_TICKS,), decode_ticks=1)),
])
def test_trace_plan_and_derived_time_errors_precede_callbacks(trace, plan):
    def forbidden(frame):
        raise AssertionError("invalid input must be rejected before decoder call")

    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(trace, plan, forbidden)


@pytest.mark.parametrize("kwargs", [
    {"correction_width": 0}, {"correction_width": True}, {"correction_width": 1.0},
    {"correction_width": MAX_WIDTH + 1}, {"decoder_label": ""},
    {"decoder_label": " padded "}, {"decoder_label": "line\nbreak"},
    {"decoder_label": "x" * 129}, {"decoder_label": "\ud800"}, {"decoder_label": 5},
])
def test_invalid_output_contract_and_decoder_label(kwargs):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(**kwargs)


def test_invalid_callback_and_object_types():
    trace, plan = example()
    for bad_trace, bad_plan, decoder in [((), plan, toy_correction), (trace, {}, toy_correction), (trace, plan, None)]:
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            run_syndrome_delivery(bad_trace, bad_plan, decoder, correction_width=2)


@pytest.mark.parametrize("output", [(True, 0), (0.0, 1), (2, 0), (-1, 0), (0,), (0, 1, 0),
                                    None, "01", {0: 0, 1: 1}, iter([0, 1]), (float("nan"), 0)])
def test_decoder_output_is_exact_bounded_binary_vector(output):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(decoder=lambda frame: output)


def test_decoder_output_bound_is_checked_before_callback():
    trace = SyndromeTrace(1, ((0,),) * MAX_FRAMES)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(trace, DeliveryPlan((1,), (0,)), lambda frame: pytest.fail("must not run"), correction_width=MAX_WIDTH)


def test_mutable_callback_outputs_are_snapshotted_per_call():
    result = [0, 0]

    def reused(frame):
        result[:] = toy_correction(frame)
        return result

    assert run(decoder=reused)["baseline_corrections"] == [[0, 1], [1, 0], [1, 1]]


def test_callback_failure_is_not_silently_converted_to_success():
    def broken(frame):
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        run(decoder=broken)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(outputs_match=False),
    lambda value: value["baseline_corrections"][0].__setitem__(0, 1),
    lambda value: value["replay_corrections"][0].__setitem__(0, True),
    lambda value: value["chronological_reassembled_frames"][0].__setitem__(0, 0),
    lambda value: value["arrivals"].reverse(),
    lambda value: value["timings"][0].update(replay_finish_tick=0),
    lambda value: value["input"].update(deadline_ticks=100),
    lambda value: value["claim_boundary"].update(hardware_latency_measurement=True),
    lambda value: value.update(output_mismatch_indices=[0]),
    lambda value: value.update(unexpected="extra"),
    lambda value: value.pop("arrivals"),
    lambda value: value.update(input_sha256="0" * 64),
])
def test_tampering_rejected_even_with_recomputed_outer_hash(mutate):
    trace, plan = example()
    receipt = deepcopy(run(trace, plan))
    mutate(receipt)
    receipt["sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_syndrome_delivery_receipt(receipt, trace, plan, toy_correction, correction_width=2)


def test_hash_tamper_wrong_inputs_and_wrong_callback_rejected():
    trace, plan = example()
    receipt = run(trace, plan)
    for candidate in (None, [], {**receipt, "sha256": "0" * 64},
                      {**receipt, "compared_frame_count": 3.0}):
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            validate_syndrome_delivery_receipt(candidate, trace, plan, toy_correction, correction_width=2)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_syndrome_delivery_receipt(receipt, trace, DeliveryPlan((4,), (0,)), toy_correction, correction_width=2)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_syndrome_delivery_receipt(receipt, trace, plan, lambda frame: (0, 0), correction_width=2)


def test_cyclic_or_extra_deep_receipt_rejected_without_unbounded_recursion():
    trace, plan = example()
    receipt = run(trace, plan)
    receipt["input"] = receipt
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_syndrome_delivery_receipt(receipt, trace, plan, toy_correction, correction_width=2)


def test_partition_choices_preserve_ordered_binary_inputs_for_small_exhaustive_trace():
    frames = tuple(itertools.product((0, 1), repeat=4))
    trace = SyndromeTrace(4, frames)
    for widths in ((4,), (1, 3), (2, 2), (1, 1, 2), (1, 1, 1, 1)):
        delays = tuple(range(len(widths)))
        jitter = tuple(tuple((len(frames) - index) % 3 for _ in widths) for index in range(len(frames)))
        receipt = run(trace, DeliveryPlan(widths, delays, jitter, deadline_ticks=4))
        assert receipt["inputs_match"] is receipt["outputs_match"] is True
        assert receipt["baseline_corrections"] == [list(toy_correction(frame)) for frame in frames]


def test_claim_boundary_is_explicit_and_immutable():
    receipt = run()
    assert receipt["claim_boundary"] == dict(CLAIM_BOUNDARY)
    assert receipt["claim_boundary"]["synthetic_classical_transport_only"] is True
    for claim in ("patchwork_decoder_or_compiler", "space_time_partition_decoding", "hardware_latency_measurement",
                  "qec_advantage_evidence", "decoder_promotion", "corrections_proven_valid", "callback_purity_proven"):
        assert receipt["claim_boundary"][claim] is False
    with pytest.raises(TypeError):
        CLAIM_BOUNDARY["decoder_promotion"] = True


def test_maximum_single_frame_width_and_tick_boundary_are_valid():
    trace = SyndromeTrace(MAX_WIDTH, ((0,) * MAX_WIDTH,))
    plan = DeliveryPlan((MAX_WIDTH,), (MAX_TICKS,), deadline_ticks=MAX_TICKS)
    receipt = run(trace, plan)
    assert receipt["timings"][0]["replay_finish_tick"] == MAX_TICKS
    assert receipt["replay_deadline_miss_count"] == 0


def test_unchanged_native_reference_bp_adapter_preserves_outputs():
    import numpy as np
    from qec.decoder.bp_decoder_reference import bp_decode

    parity_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    llr = np.asarray([2.0, 2.0, 2.0])
    trace = SyndromeTrace(2, ((0, 0), (1, 0), (0, 1), (1, 1)))
    plan = DeliveryPlan((1, 1), (2, 1), ((3, 0), (-2, -1), (0, 1), (-1, 0)),
                        deadline_ticks=3, decode_ticks=1)

    def native(frame):
        correction, _iterations = bp_decode(parity_check, llr, max_iters=20,
                                             syndrome_vec=np.asarray(frame, dtype=np.uint8),
                                             postprocess="osd0")
        return tuple(int(bit) for bit in correction)

    receipt = run(trace, plan, native, correction_width=3, decoder_label="unchanged_reference_bp_adapter")
    assert receipt["baseline_corrections"] == [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert receipt["baseline_corrections"] == receipt["replay_corrections"]
    for frame, correction in zip(trace.frames, receipt["replay_corrections"]):
        assert tuple(int(bit) for bit in (parity_check @ correction) % 2) == frame
    assert receipt["outputs_match"] is True
    assert receipt["claim_boundary"]["corrections_proven_valid"] is False
    validate_syndrome_delivery_receipt(receipt, trace, plan, native, correction_width=3,
                                       decoder_label="unchanged_reference_bp_adapter")


@pytest.mark.parametrize("target, field, value", [
    ("trace", "syndrome_width", True),
    ("trace", "syndrome_width", MAX_WIDTH + 1),
    ("trace", "frames", ((2, 0, 1, 0),)),
    ("trace", "frames", ((True, 0, 1, 0),)),
    ("trace", "frames", ((0,),)),
    ("trace", "frames", iter(((0, 0, 0, 0),))),
    ("plan", "partition_widths", (2, True)),
    ("plan", "partition_widths", (1, 1)),
    ("plan", "base_delay_ticks", (-1, 0)),
    ("plan", "base_delay_ticks", (1.0, 0)),
    ("plan", "jitter_ticks", ((0,),)),
    ("plan", "jitter_ticks", ((0.0, 0),) * 3),
    ("plan", "jitter_ticks", ((-4, 0),) * 3),
    ("plan", "decode_ticks", True),
])
def test_corrupted_frozen_inputs_are_revalidated_before_callback(target, field, value):
    trace, plan = example()
    object.__setattr__(trace if target == "trace" else plan, field, value)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(trace, plan, lambda frame: pytest.fail("must not run"))


def test_deleted_frozen_field_is_invalid_input():
    trace, plan = example()
    object.__delattr__(trace, "frames")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run(trace, plan, lambda frame: pytest.fail("must not run"))


def test_validated_input_snapshot_isolated_from_callback_mutation():
    trace, plan = example()
    original = trace.frames

    def mutating(frame):
        object.__setattr__(trace, "frames", ())
        object.__setattr__(plan, "deadline_ticks", -1)
        return toy_correction(frame)

    receipt = run(trace, plan, mutating)
    assert receipt["compared_frame_count"] == len(original)
    assert receipt["chronological_reassembled_frames"] == [list(frame) for frame in original]
    assert receipt["input"]["deadline_ticks"] == 6
    assert receipt["outputs_match"] is True


def test_receipt_object_keys_must_be_exact_strings():
    class NonExactString(str):
        pass

    trace, plan = example()
    receipt = run(trace, plan)
    receipt = {NonExactString(key): value for key, value in receipt.items()}
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_syndrome_delivery_receipt(receipt, trace, plan, toy_correction, correction_width=2)
