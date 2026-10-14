# NB eigensolver restart reproducibility

## Policy: local-pcg64-seed0-if-supported-v1

The three sparse `eigs` calls in `NonBacktrackingEigenvectorFlowAnalyzer`
(compute_modes right solve, build_flow_field right solve, and build_flow_field
left solve) use a fresh local `Generator(PCG64(0))` for each call when SciPy exposes
its `rng` keyword. No shared mutable generator or ambient global RNG seed is used.

SciPy added that keyword in 1.17.0. Its documented default uses operating-system
entropy when `rng=None`; the installed 1.18.1 solver can request new random
restart vectors even when `v0` was supplied. A supplied starting vector therefore
does not alone control that replay path. See the official
[current eigs API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html).

QEC permits SciPy >=1.10. The
[1.10.1 API](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.sparse.linalg.eigs.html)
lacks `rng`. Capability detection omits the keyword for those older signatures;
this preserves their calling convention and behavior, not a new reproducibility
guarantee. The old-API compatibility path is tested with a signature stand-in
and actual SciPy runtimes in `.github/workflows/nb-eigensolver-compatibility.yml`:
1.10.1 on Python 3.10 (NumPy 1.24.4), 1.16.3 on Python 3.12 (NumPy 2.3.5),
1.17.0 on Python 3.12 (NumPy 2.3.5), and 1.18.1 on Python 3.12 (NumPy 2.5.4).
The matrix covers the minimum supported SciPy API, the last pre-keyword release,
the first keyword release, and the observed modern regression environment.
The existing full repository and container gates remain required in addition to
these affected spectral/downstream tests. A passing compatibility test does not
establish repeatability across library versions or numerical platforms.

## Deliberately unchanged scientific decisions

The predicate remains `abs(eigenvalue) > bulk_radius`, and flow weights still use
the raw positive excess. Precision retains its original output-rounding meaning.
No resolution band, boundary tolerance, solver fallback, eigenpair pairing rule,
test fixture, or artifact schema is changed. No historical generated result is
regenerated or overwritten. Newly computed outputs can differ because the restart
stream is now fixed. In particular, a one-ulp positive excess is still retained. Exact-boundary
numerical classification can therefore remain wrong or platform-sensitive.
The dense regular-cycle fixture still demonstrates that issue independently of
ARPACK restart randomness. A precision-aware boundary convention is a separate
scientific decision requiring separate approval and evidence lineage.

## Scope, failures and evidence

Observed repeatability on one SciPy/NumPy/BLAS matrix is not proof for other
versions or graphs. The unguarded independent left solve can still fail for
other inputs; local RNG control does not supply a fallback or validate
left/right eigenpair matching. Dense-solver, equal-modulus and ill-conditioning
limitations remain.

When producing new replay evidence, record this policy ID, whether the installed
`eigs` signature supports `rng`, PCG64 and seed 0, exact SciPy/NumPy and BLAS
versions/configuration, plus source revision and experiment configuration.
The helper does not automatically add that metadata to old artifact schemas.
Use a separately versioned run manifest or an explicitly approved additive
sidecar. Preserve old artifacts, hashes and fixtures unchanged, and write new
runs to a separate evidence directory. Repeatability plumbing can change the
realized numerical solve even though the scientific predicate is unchanged.
