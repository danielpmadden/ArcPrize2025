"""reap.program
================

Lightweight representations of operations and programs for the search engine.
The classes remain intentionally small but now live in their own module so they
can be reused by unit tests or alternative search strategies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .grid_utils import deepcopy_grid, make_grid
from .operations import get_operation
from .types import Grid


def _sanitize_value(value: Any) -> Any:
    """Recursively coerce values into JSON-serialisable primitives."""

    if isinstance(value, dict):
        return {str(key): _sanitize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def sanitize_params(params: Dict[Any, Any]) -> Dict[str, Any]:
    """Normalise parameter dictionaries to be JSON-safe."""

    if not params:
        return {}
    return {str(key): _sanitize_value(value) for key, value in params.items()}


def make_hashable(value: Any) -> Any:
    """Convert arbitrarily nested structures into hash-friendly tuples."""

    if isinstance(value, dict):
        return tuple(sorted((str(k), make_hashable(v)) for k, v in value.items()))
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(make_hashable(item) for item in value)
    if isinstance(value, set):
        frozen = [make_hashable(item) for item in value]
        serialised = [
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            if isinstance(item, (list, tuple))
            else str(item)
            for item in frozen
        ]
        return tuple(sorted(serialised))
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


@dataclass
class Op:
    """Wrapper for a single DSL operation and its parameter dictionary."""

    name: str
    params: Dict[str, Any]


class Program:
    """Sequence of :class:`Op` objects applied sequentially to a grid."""

    def __init__(self, ops: List[Op]) -> None:
        self.ops = [Op(op.name, sanitize_params(op.params)) for op in ops]
        self._struct_payload_cache: Tuple[Tuple[str, Any], ...] | None = None
        self._signature_cache: str | None = None
        self._short_signature_cache: str | None = None

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

    def _struct_payload(self) -> Tuple[Tuple[str, Any], ...]:
        if self._struct_payload_cache is None:
            payload: List[Tuple[str, Any]] = []
            for op in self.ops:
                payload.append((op.name, make_hashable(op.params)))
            self._struct_payload_cache = tuple(payload)
        return self._struct_payload_cache

    def _ensure_signatures(self) -> None:
        payload = self._struct_payload()
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.md5(blob.encode()).hexdigest()
        self._signature_cache = digest
        self._short_signature_cache = digest[:12]

    def signature(self) -> str:
        """Return a structural fingerprint of the program."""

        if self._signature_cache is None:
            self._ensure_signatures()
        return self._signature_cache  # type: ignore[return-value]

    def short_signature(self) -> str:
        """Return a truncated structural signature for beam deduplication."""

        if self._short_signature_cache is None:
            self._ensure_signatures()
        return self._short_signature_cache  # type: ignore[return-value]

    def cost(self) -> float:
        """Heuristic cost balancing program length and parameter verbosity."""

        return len(self.ops) + 0.01 * sum(len(str(value)) for op in self.ops for value in op.params.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return " -> ".join(op.name for op in self.ops)


__all__ = ["Op", "Program", "sanitize_params", "make_hashable"]
