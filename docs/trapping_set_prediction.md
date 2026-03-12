# NB Eigenvector Trapping-Set Prediction (v12.8)

This feature predicts structurally unstable Tanner-graph regions **before** decoding by using non-backtracking (NB) eigenvector localization.

## Concept

Pipeline:

1. Compute the NB dominant eigenpair (via existing NB flow utilities).
2. Measure localization (IPR) of the resulting eigenvector signal.
3. Convert localized edge signal into per-variable instability scores.
4. Threshold high-instability variables and cluster connected candidates.
5. Produce a scalar `risk_score` that summarizes trapping-set severity.

## Why this correlates with decoding failure

Localized NB eigenvectors concentrate on fragile subgraphs where message passing tends to recirculate and stall.
These regions are common precursors to trapping-set behaviour, which can increase FER under iterative decoding.

## Output signals

`NBTrappingSetPredictor.predict_trapping_regions(H)` returns:

- `node_scores`: instability per variable node.
- `edge_scores`: instability per Tanner edge `(check, variable)`.
- `candidate_sets`: connected high-instability variable clusters.
- `ipr`: localization strength.
- `spectral_radius`: NB spectral radius.
- `risk_score`: scalar severity estimate (`mean(candidate node score) * ipr`).

All floating outputs are rounded to 12 decimals for deterministic behavior.

## Mutation steering integration

`NBGradientMutator` now supports opt-in flag:

- `avoid_predicted_trapping_sets: bool = False`

When enabled, mutation candidate ranking is penalized on predicted unstable variables:

- `adjusted_score = score / (1 + node_instability)`

Default behavior is unchanged when flag is disabled.

## Phase-space integration

Spectral phase-diagram points now include `trapping_risk`, enabling risk heatmaps in the spectral phase-space explorer.

