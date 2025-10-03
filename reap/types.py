"""reap.types
=================

Foundational type aliases and lightweight data structures used throughout the
REAP solver package. Centralising these definitions in a dedicated module keeps
inter-module dependencies predictable and makes it easier for future
contributors (including *future us*) to locate canonical representations of ARC
concepts. The aim is to reduce the cognitive overhead when jumping between
modules by ensuring that every file imports the exact same aliases.

The module intentionally stays minimal: no functions are declared here. This is
strictly a definitions-only space so that importing it never triggers runtime
side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Core ARC data representations
# ---------------------------------------------------------------------------
Grid = List[List[int]]
Color = int
Coord = Tuple[int, int]
Template = Tuple[Tuple[int, ...], ...]

# Slightly looser aliases used by the search subsystem. The solver currently
# relies on semi-structured objects coming from JSON task definitions, hence the
# ``Any`` typing. Keeping the aliases explicit documents the intent.
TrainPair = Any
TestItem = Any


@dataclass
class Example:
    """Container for ARC train/test examples.

    Parameters
    ----------
    input:
        The grid presented as the example input.
    output:
        The expected output grid. Test examples omit this value, hence it is
        optional. The default of ``None`` preserves backwards compatibility
        with the original script which handled missing outputs dynamically.

    Notes
    -----
    Storing examples as dataclasses provides dot-attribute access and automatic
    representation/equality semantics while still being lightweight. It also
    makes it easy to expand the structure in the future (e.g. adding metadata)
    without touching caller code.
    """

    input: Grid
    output: Optional[Grid] = None


__all__ = [
    "Grid",
    "Color",
    "Coord",
    "Template",
    "TrainPair",
    "TestItem",
    "Example",
]
