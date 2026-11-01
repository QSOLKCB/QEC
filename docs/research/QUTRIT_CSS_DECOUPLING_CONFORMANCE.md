<!-- SPDX-License-Identifier: MPL-2.0 -->
# Exact qutrit circuit donor and finite conformance

[He, Wang, Song and Gu, *Toric codes all the way down: decoupling
translation-invariant topological CSS codes in higher dimensions*,
arXiv:2610.40592v1](https://arxiv.org/abs/2610.40592v1), submitted
29 October 2026, supplies an exact prime-field Python implementation.
Its `p=3` arithmetic is a useful external reference for QEC's qutrit algebra.

This addition pins [css-decoupling commit
`a48048c1097f4d1b37a55eca9a1796a823f0c124`](https://github.com/hmx21/css-decoupling/tree/a48048c1097f4d1b37a55eca9a1796a823f0c124).
The donor is optional fixture-generation tooling, with no new QEC runtime or
default test dependency. Existing decoders, historical fixtures, governance
contracts and published release assignments remain unchanged.

This is supplemental coverage for the exact qutrit baseline described in the
[roadmap's donor interfaces](../roadmap/ROADMAP.md#donor-interfaces-and-preservation).
It does not advance a release milestone or establish acoustic/physical
protection. The migrated roadmap remains byte-bound to its preservation
manifest; this new note carries the research addition separately.

## Saved cases and independent checks

The [fixture](../../tests/fixtures/research/qutrit-css-decoupling-v1.json)
contains exact Laurent matrices, gate sequences, inverse actions and provenance.
These are synthetic conformance inputs, not the paper's experimental data.

| Case | Purpose |
|---|---|
| Identity | Empty-circuit baseline |
| SUM weights 1 and 2 | Both nonzero GF(3) coefficients, signed two-axis shifts |
| SWAP | Species ordering |
| Mixed circuit | Noncommuting gate order, SWAP and negative shifts |
| General-algorithm toric circuit | Actual donor `decouple` output: 17 gates, six registers per cell, four input ancillas |
| Toric periods 2×2, 2×3, 3×3 | Donor finite ranks and two logical qutrits per torus |

The toric input is the donor README's
`HX = [x-1, y-1]`, `HZ = [1-y^-1, x^-1-1]` over GF(3).
The generator explicitly selects `algorithm="general"`, `output="circuit"`,
`linear_backend="sympy"` and `contraction_strategy="normalized"`.
This exercises the higher-dimensional article's general construction on its
small two-dimensional input, rather than the default specialized 2D algorithm.
It requires `circuit_ready` and calls the donor's `verify_result` with the
expected input matrices before exporting anything.

The [offline tests](../../tests/test_qutrit_css_decoupling_conformance.py)
expand Laurent monomials on finite tori and independently replay each gate as
local Pauli conjugation. They compare both X/Z actions and inverse actions,
two-sided inverse products and the GF(3) symplectic pairing. Periods 1×1, 3×2
and 2×3 include collapsed shifts and distinguish both displacement signs.
Negative controls change a weight, reverse a shift, reverse gate order and
remove a SWAP; each must disagree with the saved action.

The same fixture's toric checks enter QEC's existing `QutritStabilizerCode` after
dependent generators are removed. Tests compare ranks and logical dimensions,
check syndrome covariance under the mixed circuit, and exhaustively correct
all 144 single-site nonidentity qutrit Paulis on the 3×3 torus using QEC's
unchanged bounded `ExactDecoder`.

Conventions are explicit: column vectors, X coordinates before Z coordinates,
first listed gate first. SUM has its control at `r+displacement` and target at
`r`, with `target += weight*control`. Finite arrays use lexicographic cells and
species within each cell. Global Pauli phases are omitted, as in QEC's existing
Pauli representation; this is not an arbitrary state-vector simulator.

## Reproduction

Normal tests need only QEC's development dependencies:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_qutrit_css_decoupling_conformance.py
```

To regenerate and compare against the donor, use Python 3.11 or newer in an
isolated environment (executed here with Python 3.12.14):

```sh
git clone https://github.com/hmx21/css-decoupling.git ../css-decoupling
git -C ../css-decoupling checkout --detach a48048c1097f4d1b37a55eca9a1796a823f0c124
python -m pip install 'sympy==1.14.0' 'mpmath==1.3.0'
python scripts/generate_qutrit_css_fixtures.py --donor-checkout ../css-decoupling --check
```

The generator requires the pinned, clean checkout and checks the actual import
location and dependency versions. `--check` compares every byte. To propose a
new artifact, replace `--check` with `--output /tmp/qutrit-css-candidate.json`;
the output path must not exist. Review differences before accepting a new
version. The fixture records the generator's SHA-256 and a digest of the full
donor certificate rebuilt during generation. The full certificate is not
bundled, and its digest is not itself a correctness proof or attestation.

## Claim boundaries

- The separate [Lean formalization README](https://github.com/hmx21/css-decoupling-lean/blob/d2198341f6d1282a7bc743302482ace1982d17df/README.md)
  formalizes mathematical results. Its README explicitly says it **does not
  verify the Python implementation**. No Lean build was run for this addition.
- The donor's default SymPy/Gröbner backend is **not covered by Proposition 1's
  polynomial runtime bound**. The bounded backend follows the degree bounds
  but reports resource limits on many inputs, including the 2D toric example.
  No runtime scaling result is claimed here.
- Offline tests establish these finite gate actions and QEC algebra/decoder
  comparisons. They do not independently verify the donor's complete
  decoupling certificate, gluing criterion, all possible inputs, arbitrary
  lattices or the general theorem. Regeneration runs the donor's own verifier.
- This fixture is GF(3) only. The donor supports prime dimensions; setting
  `p=4` is not a route to QEC's ququart model, and GF(4) is not arithmetic modulo 4.
- There is no hardware, threshold, speedup, acoustic protection or decoder
  promotion claim. The 17-gate circuit includes ancillas; it is not a
  six-data-qutrit code or an ancilla-free implementation.

The donor is MIT-licensed, copyright 2026 Mingxin He; its notice is preserved in
[THIRD_PARTY_LICENSES/css-decoupling-MIT.txt](../../THIRD_PARTY_LICENSES/css-decoupling-MIT.txt).
No donor implementation is vendored. The QEC-authored generator, tests and
documentation follow the repository's MPL-2.0 policy. The original donor's
notices and mathematical attribution remain applicable to its outputs.
