"""
Layer 1a — Opt-in structural extensions for the decoder.

Modules in this package provide deterministic, configuration-gated
geometry modifications to the factor graph.  All extensions must satisfy:

- Default behavior is bit-identical to baseline.
- BP loops remain untouched.
- No randomness introduced.
- No schedule mutation.
- Fully deterministic.
"""
