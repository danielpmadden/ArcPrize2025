"""reap.program
================

Lightweight representations of operations and programs for the search engine.
The classes remain intentionally small but now live in their own module so they
can be reused by unit tests or alternative search strategies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List

from .grid_utils import deepcopy_grid, make_grid
from .operations import get_operation
from .types import Grid


@dataclass
class Op:
    """Wrapper for a single DSL operation and its parameter dictionary."""

    name: str
    params: Dict[str, Any]


class Program:
    """Sequence of :class:`Op` objects applied sequentially to a grid."""

    def __init__(self, ops: List[Op]) -> None:
        self.ops = ops

    def apply(self, grid: Grid) -> Grid:
        """Apply each operation to ``grid`` defensively."""

        out = deepcopy_grid(grid)
        for op in self.ops:
            fn = get_operation(op.name)
            try:
                out = fn(out, **op.params)
            except Exception:
                return make_grid(1, 1, 0)
        return out

    def signature(self) -> str:
        """Return a structural fingerprint of the program.

        The signature is a stable hash constructed from the ordered sequence of
        operations and their parameters. It is used for structural
        deduplication both during search and when persisting solved programs in
        the dynamic library.
        """

        payload = [
            (op.name, tuple(sorted(op.params.items())))
            for op in self.ops
        ]
        return hashlib.md5(str(payload).encode()).hexdigest()

    def cost(self) -> float:
        """Heuristic cost balancing program length and parameter verbosity."""

        return len(self.ops) + 0.01 * sum(len(str(value)) for op in self.ops for value in op.params.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return " -> ".join(op.name for op in self.ops)


__all__ = ["Op", "Program"]
