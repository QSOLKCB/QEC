# NB Instability Gradient Mutation (v12.6.0)

## Overview

NB instability gradient mutation uses the dominant non-backtracking (NB)
eigenvector field to drive deterministic Tanner-graph rewiring.

Pipeline:

1. Compute NB directed-edge flow eigenvector.
2. Convert to edge instability scores.
3. Aggregate to node instability.
4. Build an instability gradient on each Tanner edge.
5. Rewire edges toward lower-instability targets while preserving degrees.

This approximates a local descent flow in graph-topology space.

## Instability Gradient Definition

For each Tanner edge `(ci, vi)`:

- `edge_score(ci,vi) = |v_(vi->ci)| + |v_(ci->vi)|`
- `node_score(node) = sum(edge_scores incident to node)`
- `gradient(ci,vi) = node_score(ci) - node_score(vi)`

In implementation, variable nodes are represented with an offset namespace
inside node-instability maps to keep check and variable ids disjoint.

## Mutation Rule

A candidate rewiring follows:

`(ci, vi) -> (ci, vj)`

with a degree-preserving partner swap on another check `cj`:

- remove `(ci,vi)` and `(cj,vj)`
- add `(ci,vj)` and `(cj,vi)`

Acceptance criterion:

- `gradient(ci,vi) > gradient(ci,vj)`

Additional constraints:

- preserve row and column degrees
- keep bipartite Tanner structure
- avoid duplicate edges
- optional 4-cycle avoidance
- deterministic ordering for all tie-breaks

## Continuous Mutation Flow

`mutate_flow(H, iterations)` applies one minimal accepted gradient swap per
iteration and recomputes the gradient field after each step.

This produces a deterministic discrete flow trajectory over graph topology.

## Example (ASCII)

```
Edge Instability Field

(c0,v2) 0.830000000000 -> v3
(c1,v4) 0.720000000000 -> v5
(c3,v1) 0.410000000000 -> v2
```

Use `src/qec/experiments/gradient_flow_plot.py`:

- `render_gradient_ascii(H)` for text output
- `plot_gradient_matplotlib(H)` for optional plotting when matplotlib exists

## Determinism Notes

- no random sampling is used by gradient analyzer or mutator
- all rankings use deterministic sorted order with explicit tie-breakers
- outputs are rounded to fixed precision (`float64` + 12 decimals)
- sparse and dense inputs are both supported
