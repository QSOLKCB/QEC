# Opt-in CSS witnesses and syndrome-delivery replay

These Layer-5 experiments are isolated research checks. Importing QEC does not
activate them; no decoder, channel, runtime, promotion rule, historical fixture,
release version, or command registry is changed.

## Research motivation and boundary

- [Bichmann, Schmid and Wille, arXiv:2610.18218v1](https://arxiv.org/abs/2610.18218v1)
  distinguishes physical coordinate transformations from changes of generator
  representation. Here we check a supplied binary CSS witness. We do not
  implement the paper's hybrid equivalence-search procedure or local-Clifford
  equivalence. Matching ranks alone is never used to establish equivalence.
- [Chamberland et al., Patchwork, arXiv:2610.22112v1](https://arxiv.org/abs/2610.22112v1)
  describes lattice-surgery compilation and wavefront space-time partitioned
  decoding. Its simulated comparisons motivate keeping explicit comparison
  inputs and assumptions. Our separate experiment fragments classical syndrome
  delivery, reassembles the same trace, and calls the same decoder. It does not
  implement Patchwork, lattice surgery, commit/buffer/cleanup decoding, or
  independent partition decoders. Transport fragmentation is not a substitute
  for that algorithm.

All bundled examples are locally constructed synthetic inputs, not the papers'
experimental data or a reproduction of their results. No hardware latency,
logical-error-rate advantage, quantum-state transfer, universal equivalence,
or automatic decoder authority is established.

The Brillouin-fibre spectral-response proposal is deferred: physical response
and avoided-crossing data need their own measurement units, uncertainty and
source-data contract. Those data alone do not validate QEC or quantum-state
transfer, so they are not mixed into these conformance checks.

## Binary CSS presentation witness

`qec.experiments.css_equivalence` accepts binary check matrices Hx and Hz,
invertible row transforms Ux and Uz, and one shared qubit permutation p. Its
convention is `new[j] = old[p[j]]`. These are unsigned binary checks with
conventional +1 pure-X/pure-Z generators; signed stabilizer phases are not
represented. It checks CSS commutation, exact transformed
matrices, ranks, encoded-qubit count, and covariance of the supplied X/Z error
syndromes. The verifier rebuilds the result from the caller's trusted inputs;
merely recomputing an artifact hash cannot make a false witness valid.

This is an algebraic presentation change, not mutation of a live Tanner graph.
Row-basis changes can alter row/column degrees. Neither graph optimizers nor
protected decoders are fed these transformed matrices. Arithmetic is strictly
GF(2), without silently reducing other values modulo two. Qutrit GF(3) and
ququart algebra require separate constructions and are rejected here.

## Classical delivery experiment

`qec.experiments.syndrome_delivery` uses immutable `SyndromeTrace` and
`DeliveryPlan` inputs. Contiguous fragment widths must cover the syndrome.
Explicit integer base delays and a frame-by-fragment jitter matrix determine
arrival events. Reassembly preserves the original frame order even when later
frames arrive first. The same caller-supplied pure decoder is invoked separately
on baseline and replay inputs, and its actual exact binary outputs are compared.

The decoder callback must be deterministic and side-effect-free. The harness
cannot prove callback purity or decoder correctness. It reports output mismatch
rather than promoting either result. Callback execution time is not measured.
Declared decode costs and deadlines are modeled integer ticks with no implicit
physical unit, randomness, wall clock, sleep, network, or real-time guarantee.
Completion exactly at a deadline is on time. A zero-frame trace is vacuous.

## Reproducible evidence

The companion generation script and new fixture record their exact source-file
hashes, versioned research links, inputs, outputs and claim boundaries. Historical
fixtures remain untouched. Use the documented `--check` command below to compare
fresh canonical bytes against the checked-in fixture rather than trusting an
unverified success flag.

```sh
python -m pip install -e '.[dev]'
python scripts/generate_qec_research_evidence.py --check
python -m pytest -q tests/test_css_equivalence.py tests/test_syndrome_delivery.py tests/test_qec_research_evidence.py
python -m pytest -q
```

To intentionally regenerate the new fixture after reviewed source changes, run
`python scripts/generate_qec_research_evidence.py` without `--check`. The generator
never overwrites historical evidence. Its source hashes bind the listed files,
not a complete transitive dependency closure or runtime attestation. Its native
adapter calls the unchanged reference BP decoder with fixed three-bit repetition
checks and OSD-0 postprocessing; the comparison covers correction bits only.
The test independently checks that all four supplied syndromes are reproduced
by those corrections. This finite check is not a general decoder proof.
