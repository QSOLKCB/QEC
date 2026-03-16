# Spectral Basin Visualization (v12.7.0)

## Concept

A **spectral basin** is a region in spectral phase space where Tanner
graph mutations produce similar non-backtracking (NB) stability
behavior.

For this repository, the trajectory state is tracked with:

- `spectral_radius` (from NB flow analyzer output)
- `ipr` (inverse participation ratio of NB variable flow)
- `nb_energy` (directed-edge flow energy)

These signals are deterministic and rounded to 12 decimals.

## Mutation Trajectory Interpretation

`SpectralBasinVisualizer.trace_mutation_trajectory(...)` applies
deterministic NB-gradient mutation steps and records a point at each
iteration.

Typical interpretation:

- decreasing spectral radius + increasing IPR may indicate migration
  toward a more localized instability regime
- flatter trajectories indicate a relatively stable basin
- abrupt jumps indicate crossing between basins

## Relationship to NB Instability Field

The NB instability field defines edge/node pressure for deterministic
rewiring. The basin trajectory is a compact projection of that
higher-dimensional field:

Tanner graph mutation
      ↓
NB instability field
      ↓
graph evolution
      ↓
trajectory through spectral phase space

This gives a practical bridge between local mutation decisions and
global spectral evolution during graph search.
