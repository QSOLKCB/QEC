"""Import audit tests — prevent src.qec regression (v82.6.1)."""

import pytest


def test_import_qec_root():
    import qec  # noqa: F401


def test_import_experiments():
    import qec.experiments  # noqa: F401


def test_import_diagnostics():
    import qec.diagnostics  # noqa: F401


def test_import_cross_domain():
    # Verify the module resolves without 'src' prefix errors.
    # Skip if a transitive native dependency (e.g. cryptography) is broken.
    try:
        from qec.experiments.cross_domain_mapper import run_cross_domain_mapping  # noqa: F401
    except ModuleNotFoundError as exc:
        if "src" in str(exc):
            raise  # genuine import-path regression
        pytest.skip(f"optional native dependency unavailable: {exc}")
    except BaseException as exc:
        # pyo3_runtime.PanicException inherits from BaseException
        if "src" in str(exc):
            raise
        pytest.skip(f"optional native dependency unavailable: {exc}")
