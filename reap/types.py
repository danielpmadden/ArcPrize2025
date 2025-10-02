"""Core type definitions for the REAP project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Grid = List[List[int]]


@dataclass
class Example:
    """Represents a single ARC training or evaluation example."""

    input: Grid
    output: Optional[Grid] = None


Color = int
Coord = Tuple[int, int]
Object = Dict[str, Any]
Template = Tuple[Tuple[int, ...], ...]
TrainPair = Any
TestItem = Any

__all__ = [
    "Color",
    "Coord",
    "Example",
    "Grid",
    "Object",
    "Template",
    "TestItem",
    "TrainPair",
]
