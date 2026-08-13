# NEXUS historical source boundary

## Required CI archival-source boundary

QEC preserves the historical NEXUS v3.0.0 and v4.0.0 repository, commit, DOI,
and capability identities in `qec.adapters.nexus.source`. Those identities are
evidence metadata and remain distinct from the present availability of the
historical repository location.

The required `NEXUS Bridge` CI job uses `archival_evidence_validation`. It runs
the complete QEC suite, validates the canonical v4.0.1 qBraid replication
receipt, reasserts the exact v3/v4 profile pins, and uploads a canonical
`qec.nexus-historical-source-status.v1` record with the replication receipt.

Required CI does not substitute another repository, branch, tag, or commit when
the historical repository location is unavailable. It also does not treat a DOI
attachment as buildable source unless that exact source archive and digest have
been separately pinned.

Direct execution remains supported when an operator possesses the exact frozen
source bytes. In that case `qec-nexus-bridge attest-build` binds the observed
binary to the declared historical source profile before QEC accepts execution
evidence.

This boundary keeps archival evidence continuously checkable without presenting
a different source tree as a historical rebuild.
