"""
v8.4.0 — Generation Public API.

Exposes all public generation functions from a single module.
Import from here for stable access to the generation layer.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

# ── v8.4.0: Tanner graph generation ────────────────────────────────
from qec.generation.tanner_graph_generator import (
    generate_tanner_graph_candidates,
)
from qec.generation.candidate_evaluation import (
    evaluate_tanner_graph_candidate,
)
from qec.generation.candidate_ranking import (
    rank_tanner_graph_candidates,
)
from qec.generation.export_generated_graph import (
    export_generated_graph,
)

# ── v8.5.0: Deterministic cycle-avoidant construction ───────────
from qec.generation.deterministic_construction import (
    construct_deterministic_tanner_graph,
)

# ── v10.2.0: PEG-seeded spectral population initialization ────
from qec.generation.peg_generator import (
    generate_peg_tanner_graph,
    generate_peg_population,
)
