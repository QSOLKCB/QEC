# Experimental Research Tools (v12.4.0)

These are experimental research instruments for evaluating NB-guided
mutation strategies and quantum stabilizer structures. They live
outside the core discovery engine and introduce no new required
dependencies.

## NB Flow Heatmap Visualization

**Module:** `src/qec/experiments/nb_flow_heatmap.py`

Visualizes non-backtracking flow magnitude on Tanner graph edges.

### Edge Scoring

For each undirected edge (ci, vi) in the Tanner graph, the score is:

    score(ci, vi) = |v_(vi→ci)| + |v_(ci→vi)|

where v is the dominant NB eigenvector. Scores are normalized to [0, 1].

### Usage

```python
from src.qec.experiments.nb_flow_heatmap import (
    compute_edge_flow_scores_from_H,
    format_ascii_heatmap,
    plot_flow_heatmap,
)

scores = compute_edge_flow_scores_from_H(H)
print(format_ascii_heatmap(scores, top_k=10))

# Optional matplotlib plot (requires matplotlib)
fig = plot_flow_heatmap(H, scores, output_path="heatmap.png")
```

### Output Formats

- **ASCII heatmap**: Bar chart showing top-k edges by NB flow magnitude.
- **Matplotlib plot** (optional): Bipartite Tanner graph with edges
  colored by NB magnitude. Requires matplotlib to be installed.

---

## NB Mutation Ablation Experiment

**Script:** `experiments/nb_mutation_ablation.py`

Compares three mutation strategies on randomly generated Tanner graphs
to evaluate whether NB-guided mutation improves graph structure.

### Strategies

1. **Baseline** — no mutation applied.
2. **Random mutation** — degree-preserving random edge swaps.
3. **NB-guided mutation** — edge swaps guided by NB eigenvector magnitude.

### Metrics Collected

- `girth` — shortest cycle length
- `cycle_count_4`, `cycle_count_6` — short cycle counts
- `nb_ipr` — inverse participation ratio of NB flow
- `max_flow`, `mean_flow` — NB flow statistics
- `flow_localization` — IPR-based flow concentration
- `runtime_s` — wall-clock time
- `mutations_applied` — number of edge swaps applied

### Usage

```python
from experiments.nb_mutation_ablation import run_ablation, serialize_ablation_results

results = run_ablation(
    m=6, n=12, row_weight=4,
    k_mutations=3, num_trials=5, master_seed=42,
)
print(serialize_ablation_results(results))
```

### Determinism

All randomness flows through SHA-256 sub-seed derivation from the
master seed. Results are byte-identical across runs.

---

## IPR-Weighted NB Mutation (Experimental)

**Config option:** `use_ipr_weight` on `NBGuidedMutator`.

When enabled, edge scores are multiplied by the IPR of the NB
eigenvector. This amplifies mutation pressure when flow is
concentrated (localized), hypothesizing that localized flow indicates
trapping-set structures.

### Configuration

```python
from src.qec.discovery.mutation_nb_guided import NBGuidedMutator

mutator = NBGuidedMutator(enabled=True, use_ipr_weight=True)
# or from config dict:
config = {"nb_mutation": {"enabled": True, "use_ipr_weight": True}}
mutator = NBGuidedMutator.from_config(config)
```

Default: `use_ipr_weight=False` (standard scoring).

---

## QuTiP Quantum Experiment Harness (Optional)

**Directory:** `src/qec/experiments/qutip/`

Experimental quantum simulations of stabilizer structures using QuTiP.

**QuTiP is an optional dependency.** It is not required for any core
QEC functionality. Install separately if needed:

```
pip install qutip
```

### Modules

- `stabilizer_mapping.py` — Maps H rows to Pauli stabilizer operators
  and constructs stabilizer Hamiltonians.
- `qutip_harness.py` — Runs stabilizer energy and time-evolution
  experiments.
- `noise_experiment.py` — Simulates depolarizing noise on stabilizer
  structures.

### Example

```python
from src.qec.experiments.qutip.qutip_harness import (
    run_stabilizer_energy_experiment,
    run_time_evolution_experiment,
)

result = run_stabilizer_energy_experiment(H)
print(f"Ground energy: {result['ground_energy']}")

evolution = run_time_evolution_experiment(H, t_max=10.0, num_steps=50)
```

### Stabilizer Mapping

Each row of the parity-check matrix H is mapped to a Pauli-Z
stabilizer operator. For example, the row `[0, 1, 1, 0]` maps to
`IZZI`. The stabilizer Hamiltonian is:

    H_stab = coupling * sum_i S_i

### Noise Experiment

Simulates Lindblad master equation evolution with single-qubit
depolarizing collapse operators, measuring stabilizer expectation
decay over time.
