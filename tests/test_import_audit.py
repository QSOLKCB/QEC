"""Import audit tests — prevent regressions (v82.6.2)."""


def test_import_qec_root():
    import qec  # noqa: F401


def test_import_experiments():
    import qec.experiments  # noqa: F401


def test_import_diagnostics():
    import qec.diagnostics  # noqa: F401


def test_import_cross_domain():
    from qec.experiments.cross_domain_mapper import run_cross_domain_mapping  # noqa: F401


def test_import_midi_cube_bridge():
    from qec.experiments.midi_cube_bridge import run_midi_cube_experiment  # noqa: F401


def test_execution_proof_guards_cryptography():
    """execution_proof must import cleanly even when cryptography is broken."""
    from qec.controller.execution_proof import _CRYPTO_AVAILABLE  # noqa: F401
