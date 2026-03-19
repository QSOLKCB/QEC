"""Minimal import-integrity test for qec.experiments (v82.3.1)."""


def test_import_qec():
    import qec  # noqa: F401


def test_import_qec_experiments():
    import qec.experiments  # noqa: F401


def test_import_uff_bridge():
    from qec.experiments.uff_bridge import run_uff_experiment  # noqa: F401


def test_import_midi_cube_bridge():
    from qec.experiments.midi_cube_bridge import run_midi_cube_experiment  # noqa: F401
