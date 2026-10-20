ALGORITHM_PATTERNS.md

QSOL QEC Canonical Algorithm Patterns

This document defines approved algorithm implementation patterns for the QSOL QEC repository.

Its purpose is to guide AI agents and contributors when implementing:

spectral graph algorithms

Tanner graph analysis

decoding experiments

instability diagnostics

Algorithms should follow the patterns defined here unless explicitly instructed otherwise.

1. Deterministic Function Pattern

All algorithmic functions must be deterministic.

Canonical pattern:

def compute_metric(graph, *, seed=None):
    """
    Compute deterministic metric for a Tanner graph.

    Parameters
    ----------
    graph : TannerGraph
        Input Tanner graph structure.
    seed : int | None
        Optional deterministic seed.

    Returns
    -------
    float
        Computed metric.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)

    # deterministic computation here

    return metric

Rules:

seed must be explicit

no hidden randomness

no global RNG usage

2. Tanner Graph Traversal Pattern

Graph traversal must use deterministic ordering.

Canonical pattern:

for node in sorted(graph.variable_nodes):
    neighbors = sorted(graph.neighbors(node))

    for neighbor in neighbors:
        process_edge(node, neighbor)

Rules:

always sort node lists

never rely on unordered iteration

avoid Python set iteration without sorting

3. Sparse Spectral Operator Pattern

Large spectral operators must be implemented as LinearOperators.

Never construct dense matrices.

Canonical pattern:

from scipy.sparse.linalg import LinearOperator


def build_nb_operator(graph):

    n = graph.num_directed_edges()

    def matvec(x):
        return non_backtracking_matvec(graph, x)

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=float,
    )

Rules:

memory must scale with |E|

avoid dense matrix creation

compute matrix-vector products on demand

4. Dominant Eigenpair Pattern

Spectral diagnostics require only the dominant eigenpair.

Canonical pattern:

from scipy.sparse.linalg import eigs


vals, vecs = eigs(
    operator,
    k=1,
    which="LR",  # largest real eigenvalue
    tol=1e-6,
)

Rules:

compute only k=1

use which="LR"

never compute the full spectrum

5. Inverse Participation Ratio (IPR)

IPR measures eigenvector localization.

Canonical implementation:

def compute_ipr(v):
    v = np.abs(v)
    return np.sum(v**4) / (np.sum(v**2)**2)

Interpretation:

IPR	Meaning
low	delocalized eigenvector
high	localized trapping set
6. Edge Sensitivity Pattern

Edge instability signals derive from NB eigenvector components.

Canonical proxy:

sensitivity(edge) = |v_i|² · |v_j|²

Implementation pattern:

def edge_sensitivity(edge, eigenvector):

    i, j = edge

    return abs(eigenvector[i])**2 * abs(eigenvector[j])**2

Edges with high sensitivity are candidates for instability.

7. Spectral Heatmap Pattern

Heatmaps project spectral signals onto graph nodes.

Canonical implementation:

def node_heat(graph, eigenvector):

    heat = np.zeros(graph.num_nodes)

    for edge_id, (i, j) in enumerate(graph.directed_edges):

        weight = abs(eigenvector[edge_id])

        heat[i] += weight
        heat[j] += weight

    return heat

Optional scaling:

heat *= IPR
8. Deterministic Graph Repair Pattern

Graph repair must be deterministic.

Canonical loop:

while instability_score(graph) > threshold:

    edges = rank_edges_by_sensitivity(graph)

    candidate = edges[0]

    proposal = propose_swap(graph, candidate)

    if is_valid_qldpc_swap(graph, proposal):
        graph = apply_swap(graph, proposal)

Rules:

one swap at a time

deterministic ranking

strict constraint checks

9. QLDPC Constraint Validation

Graph operations must preserve stabilizer commutativity.

Canonical check:

H_X @ H_Z.T == 0

Implementation pattern:

def validate_qldpc_constraints(Hx, Hz):

    if not np.all((Hx @ Hz.T) % 2 == 0):
        raise ValueError("Invalid QLDPC graph: commutativity violated")

Repairs must abort if constraints fail.

10. Warm-Start Spectral Update Pattern

Incremental updates reuse previous eigenvectors.

Canonical pattern:

eigs(
    operator,
    k=1,
    which="LR",
    v0=previous_eigenvector,
)

Benefits:

faster convergence

avoids full recomputation

11. Experiment Harness Pattern

Experiments must be structured and deterministic.

Canonical structure:

results = []

for graph in graph_set:
    for noise in noise_levels:

        outcome = run_decoder(graph, noise)

        results.append({
            "graph_id": graph.id,
            "noise": noise,
            "success": outcome,
        })

Results must be serialized deterministically.

12. Phase Diagram Experiment Pattern

Phase diagrams follow a fixed grid.

Canonical structure:

for rho in spectral_values:
    for p in noise_values:

        graph = generate_graph_with_rho(rho)

        result = run_decoder(graph, p)

        record(rho, p, result)

Outputs:

CSV grid

JSON dataset

Visualization is external.

13. Input Masking Pattern (Ternary Experiments)

Ternary mitigation operates by modifying input priors only.

Canonical pattern:

def apply_uncertainty_mask(llr, nodes):

    masked = llr.copy()

    for node in nodes:
        masked[node] = 0

    return masked

The decoder remains unchanged.

14. Deterministic Ranking Pattern

Ranking must be stable.

Canonical pattern:

sorted(items, key=lambda x: (score(x), x.id))

Rules:

always provide secondary key

avoid unstable ordering

15. Artifact Serialization Pattern

Experiment outputs must be deterministic.

Canonical pattern:

json.dumps(
    artifact,
    sort_keys=True,
    separators=(",", ":"),
)

Rules:

sorted keys

canonical formatting

no floating precision drift

16. Safety Rule for Spectral Algorithms

Spectral code must never:

build dense NB matrices

compute full spectra

use random initialization without seeds

use stochastic optimization

Approved tools:

scipy.sparse

scipy.sparse.linalg

Krylov eigensolvers

linear operators

Final Principle

Algorithms in this repository must be:

deterministic

mathematically justified

scalable

transparent

Avoid clever tricks.

Prefer simple, provable implementations.

This repository is a scientific instrument, not a heuristic optimizer.
