# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Verbindung:
    ziel: str
    gewicht: float
    typ: str


@dataclass
class Konzept:
    id: str
    labels: List[str] = field(default_factory=list)
    semantische_features: List[str] = field(default_factory=list)
    verbindungen: List[Verbindung] = field(default_factory=list)
    aktivierung: float = 0.0
    letzte_aktivierung: datetime = field(default_factory=datetime.now)


@dataclass
class WMItem:
    id: str
    a: float
    role: str = "CONTEXT"
    age: int = 0


@dataclass
class TraceItem:
    tick: int
    src: str
    dst: str
    typ: str
    contrib: float
    layer: str  # "semantic" | "episodic"
