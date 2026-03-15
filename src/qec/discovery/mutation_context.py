from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MutationContext:
    spectral_defect_atlas: Optional[Any] = None
    enable_spectral_defect_atlas: bool = False
    nb_spectral_radius: float = 0.0
