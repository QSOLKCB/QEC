"""Pinned NEXUS source identities and capability profiles for QEC v170.2.0."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class NexusSourceIdentity:
    profile: str
    version: str
    repository: str
    commit: str
    doi: str | None
    doi_status: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("profile", "version", "repository", "commit", "doi_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if not _COMMIT_RE.fullmatch(self.commit):
            raise ValueError("commit must be a lowercase 40-character Git SHA")
        if self.doi is not None and not self.doi.startswith("10.5281/zenodo."):
            raise ValueError("doi must be a Zenodo DOI when present")
        if self.doi_status not in {"published", "pending"}:
            raise ValueError("doi_status must be published or pending")
        if self.doi_status == "published" and self.doi is None:
            raise ValueError("published source identities require a DOI")
        if tuple(sorted(set(self.capabilities))) != self.capabilities:
            raise ValueError("capabilities must be sorted and unique")

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "version": self.version,
            "repository": self.repository,
            "commit": self.commit,
            "doi": self.doi,
            "doi_status": self.doi_status,
            "capabilities": list(self.capabilities),
        }


NEXUS_V3: Final = NexusSourceIdentity(
    profile="v3.0.0",
    version="3.0.0",
    repository="https://github.com/QSOLKCB/NEXUS",
    commit="e078b135322dc12a2565b9c512fc4ba75193dea7",
    doi="10.5281/zenodo.21745329",
    doi_status="published",
    capabilities=("trace", "verify"),
)

NEXUS_V4: Final = NexusSourceIdentity(
    profile="v4.0.0",
    version="4.0.0",
    repository="https://github.com/QSOLKCB/NEXUS",
    commit="1e93a509a28144d70a17fa76b330ae042db7beab",
    doi="10.5281/zenodo.21748514",
    doi_status="published",
    capabilities=(
        "fibonacci",
        "receipt",
        "ternary",
        "trace",
        "verify",
        "verify-parallel",
    ),
)

PROFILES: Final = {source.profile: source for source in (NEXUS_V3, NEXUS_V4)}


def source_profile(name: str) -> NexusSourceIdentity:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown NEXUS source profile: {name}") from exc
