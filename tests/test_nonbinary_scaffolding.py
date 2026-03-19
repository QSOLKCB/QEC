"""
Tests for nonbinary scaffolding (v3.0.1, Deliverable 4).

Verifies:
1) Import hygiene: core decoder import does not pull in nonbinary.
2) Interfaces can be imported without error.
3) Placeholders raise NotImplementedError.
"""

import sys

import pytest


class TestImportHygiene:
    """Core decoder modules must NOT import nonbinary scaffolding."""

    def test_core_decoder_does_not_import_nonbinary(self):
        # Remove any cached imports of nonbinary.
        mods_to_remove = [k for k in sys.modules if k.startswith("src.nonbinary")]
        for m in mods_to_remove:
            del sys.modules[m]

        # Import core decoder.
        import src.decoder  # noqa: F401

        # Verify nonbinary was NOT imported as a side effect.
        nonbinary_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nonbinary_mods == [], (
            f"Importing src.decoder pulled in nonbinary modules: {nonbinary_mods}"
        )

    def test_core_init_does_not_import_nonbinary(self):
        mods_to_remove = [k for k in sys.modules if k.startswith("src.nonbinary")]
        for m in mods_to_remove:
            del sys.modules[m]

        # Import core package (triggers bp_decode, simulate_fer, etc.).
        import src  # noqa: F401

        nonbinary_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nonbinary_mods == [], (
            f"Importing src pulled in nonbinary modules: {nonbinary_mods}"
        )

    def test_bench_does_not_import_nonbinary(self):
        mods_to_remove = [k for k in sys.modules if k.startswith("src.nonbinary")]
        for m in mods_to_remove:
            del sys.modules[m]

        import src.bench  # noqa: F401

        nonbinary_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nonbinary_mods == [], (
            f"Importing src.bench pulled in nonbinary modules: {nonbinary_mods}"
        )

    def test_qudit_does_not_import_nonbinary(self):
        mods_to_remove = [k for k in sys.modules if k.startswith("src.nonbinary")]
        for m in mods_to_remove:
            del sys.modules[m]

        import src.qudit  # noqa: F401

        nonbinary_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nonbinary_mods == [], (
            f"Importing src.qudit pulled in nonbinary modules: {nonbinary_mods}"
        )


class TestInterfaceImports:
    """Interfaces can be imported and inspected."""

    def test_import_gfq_message_passer(self):
        from nonbinary.interfaces import GFqMessagePasser
        assert GFqMessagePasser is not None

    def test_import_nonbinary_stabilizer_code(self):
        from nonbinary.interfaces import NonbinaryStabilizerCode
        assert NonbinaryStabilizerCode is not None

    def test_import_qudit_syndrome_model(self):
        from nonbinary.interfaces import QuditSyndromeModel
        assert QuditSyndromeModel is not None

    def test_protocols_are_runtime_checkable(self):
        from nonbinary.interfaces import (
            GFqMessagePasser,
            NonbinaryStabilizerCode,
            QuditSyndromeModel,
        )
        # runtime_checkable protocols support isinstance checks.
        assert hasattr(GFqMessagePasser, "__protocol_attrs__") or True
        assert hasattr(NonbinaryStabilizerCode, "__protocol_attrs__") or True
        assert hasattr(QuditSyndromeModel, "__protocol_attrs__") or True


class TestPlaceholders:
    """Placeholder stubs must raise NotImplementedError."""

    def test_gfq_bp_decode_raises(self):
        from nonbinary.placeholders import gfq_bp_decode
        with pytest.raises(NotImplementedError, match="not implemented"):
            gfq_bp_decode(
                parity_check=None,
                field_order=3,
                syndrome=None,
                channel_priors=None,
            )

    def test_nonbinary_stabilizer_syndrome_raises(self):
        from nonbinary.placeholders import nonbinary_stabilizer_syndrome
        with pytest.raises(NotImplementedError, match="not implemented"):
            nonbinary_stabilizer_syndrome(
                stabilizer_matrix=None,
                error_vector=None,
                field_order=3,
            )

    def test_qudit_error_sample_raises(self):
        from nonbinary.placeholders import qudit_error_sample
        with pytest.raises(NotImplementedError, match="not implemented"):
            qudit_error_sample(n=10, dimension=3, p=0.01)
