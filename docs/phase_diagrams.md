# Spectral Instability Phase Diagrams (v12.5.0)

## Concept

Belief-propagation decoders on QLDPC codes exhibit sharp phase
transitions between convergent and unstable decoding regimes.  These
transitions correlate with spectral properties of the underlying
Tanner graph's non-backtracking (NB) operator.

A **spectral instability phase diagram** maps decoder behaviour
(Frame Error Rate) across two axes:

- **x-axis**: NB spectral radius — measures the dominant eigenvalue
  magnitude of the non-backtracking matrix.  Higher values indicate
  stronger structural amplification of BP messages.

- **y-axis**: Channel error rate — the physical noise level applied
  to the code.

- **color/intensity**: FER — the fraction of decode trials that fail
  at each (spectral_radius, error_rate) point.

The resulting diagram reveals:

1. **Stable regions** (low FER): where the graph structure supports
   reliable BP convergence even at moderate noise.

2. **Unstable regions** (high FER): where spectral amplification
   causes BP to diverge or oscillate.

3. **Phase boundaries**: the critical spectral radius beyond which
   FER increases sharply for a given error rate.

## Spectral Metrics

Three key metrics are extracted per Tanner graph:

### NB Spectral Radius

The dominant eigenvalue of the non-backtracking operator.  This
controls the amplification rate of BP messages along non-backtracking
walks.  Graphs with spectral radius near or above the BP threshold
are structurally predisposed to instability.

### Inverse Participation Ratio (IPR)

Measures localization of the dominant NB eigenvector.  High IPR
indicates that instability energy concentrates on a small subset
of variable nodes (trapping-set signature).  Low IPR indicates
distributed flow.

### Flow Alignment

Cosine similarity between the NB eigenvector flow direction and the
BP residual gradient.  High alignment confirms that structural
instability (spectral) predicts decoder instability (BP).

## Interpreting Phase Diagrams

### Reading the Heatmap

```
error_rate →
spectral_radius ↓

0.02 | █
0.05 | ███
0.10 | █████
0.15 | ███████
```

- Darker/denser cells indicate higher FER.
- The boundary between light and dark regions is the **phase
  transition line**.
- Graphs above this line (high spectral radius, high error rate)
  are in the unstable decoding regime.

### Key Observations

- At low error rates, even high-spectral-radius graphs may decode
  successfully (noise is insufficient to excite the unstable mode).

- At high error rates, even low-spectral-radius graphs may fail
  (noise overwhelms the code).

- The interesting region is the **intermediate zone** where graph
  structure determines success or failure.

## Mutation Evaluation Workflow

The v12.5 ablation experiment compares four mutation strategies:

1. **Baseline**: no graph modification.
2. **Random swap**: degree-preserving random edge rewiring.
3. **NB swap**: eigenvector-guided edge rewiring targeting
   high-flow edges.
4. **NB x IPR swap**: IPR-weighted variant that amplifies targeting
   when flow is localized.

### Workflow

1. Generate N random Tanner graphs.
2. Apply each mutation strategy to each graph.
3. For each (strategy, graph) pair:
   - Extract spectral metrics (spectral_radius, IPR).
   - Run decoder at a fixed error rate to measure FER.
4. Plot FER vs spectral_radius per strategy.

### Expected Outcome

If spectral-guided mutation is effective:

- NB swap and NB x IPR swap should shift graphs toward **lower
  spectral radius** and correspondingly **lower FER**.
- Random swap should show no systematic improvement.
- The FER vs spectral_radius plot should show NB-guided points
  clustering in the lower-left (low SR, low FER) quadrant.

## Running Experiments

### Phase Diagram Generation

```python
from src.qec.experiments.spectral_phase_diagram import (
    SpectralPhaseDiagramGenerator,
)
from src.qec.experiments.phase_diagram_plot import (
    render_ascii_heatmap,
)

gen = SpectralPhaseDiagramGenerator(base_seed=42)
result = gen.generate_phase_diagram(
    graphs=my_graphs,
    error_rates=[0.02, 0.05, 0.10, 0.15],
    trials_per_point=100,
)
print(render_ascii_heatmap(result))
```

### Mutation Ablation

```python
from experiments.run_nb_mutation_ablation import run_ablation
from src.qec.experiments.fer_vs_spectral_radius import (
    render_ascii_fer_vs_spectral_radius,
)

results = run_ablation(num_graphs=200, fer_trials=50)
print(render_ascii_fer_vs_spectral_radius(results))
```
