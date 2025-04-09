from typing import Optional
from dataclasses import dataclass


@dataclass
class AuditorFeature:
    name: str
    label: Optional[str] = None
    levels: Optional[list[any]] = None


@dataclass
class AuditorScore:
    name: str
    label: Optional[str] = None
    threshold: Optional[float] = None


@dataclass
class AuditorOutcome:
    name: str
    mapping: Optional[dict[any, int]] = None
