from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/nexus-bridge.yml")
DOC = Path("docs/replications/NEXUS_HISTORICAL_SOURCE_BOUNDARY.md")


def test_required_nexus_ci_uses_archival_evidence_not_dead_checkout() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "repository: QSOLKCB/NEXUS" not in text
    assert "Checkout published NEXUS v3 baseline" not in text
    assert "Checkout frozen NEXUS v4 source profile" not in text
    assert "archival_evidence_validation" in text
    assert '"source_substitution_allowed": False' in text
    assert (
        "qec-nexus-validate-replication receipt" in text
        and "docs/replications/nexus_v4_0_1_qbraid_receipt.json" in text
    )
    assert (
        "659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8"
        in text
    )


def test_nexus_bridge_docs_state_archival_ci_boundary() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "Required CI archival-source boundary" in text
    assert "does not substitute" in text
    assert "historical repository location" in text
    assert "qec.nexus-historical-source-status.v1" in text
