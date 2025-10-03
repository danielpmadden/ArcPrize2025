"""Dynamic macro program library for REAP.

This module implements :class:`LibraryManager`, a lightweight persistence layer
that stores successful programs discovered by the solver. The data is persisted
as JSON to keep the format human-auditable and interoperable with other tools.
Each stored macro records the exact sequence of operations along with metadata
such as creation time, originating training hash, and reuse statistics.

Macros are deserialised back into :class:`~reap.program.Program` instances when
loaded so they can be injected into :data:`~reap.operations.FUNCTION_REGISTRY`
as first-class operators (``lib::<id>``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .program import Op, Program, make_hashable, sanitize_params
from .types import Example


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(seed: str | None = None) -> str:
    """Generate a short, stable identifier suitable for macro keys."""

    if seed is None:
        seed = secrets.token_hex(8)
    digest = hashlib.md5(seed.encode()).hexdigest()
    return f"lib_{digest[:6]}"


def _program_to_ops(program: Program) -> List[Dict[str, Any]]:
    """Serialise ``program`` into JSON-friendly operation dictionaries."""

    ops: List[Dict[str, Any]] = []
    for op in program.ops:
        ops.append({"name": op.name, "params": sanitize_params(op.params)})
    return ops


def _ops_to_program(ops: Sequence[Dict[str, Any]]) -> Program:
    """Deserialize a sequence of operation dictionaries into a Program."""

    return Program([Op(item["name"], dict(item.get("params", {}))) for item in ops])


@dataclass
class LibraryItem:
    """In-memory representation of a persisted macro program."""

    identifier: str
    ops: List[Dict[str, Any]]
    signature: List[str]
    meta: Dict[str, Any]
    name: str | None = None
    behavior_signature: str | None = None

    def to_json(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation."""

        payload = {
            "id": self.identifier,
            "signature": self.signature,
            "ops": self.ops,
            "meta": self.meta,
        }
        if self.name:
            payload["name"] = self.name
        if self.behavior_signature:
            payload["behavior_signature"] = self.behavior_signature
        return payload


class LibraryManager:
    """Persist and load successful programs discovered by the solver."""

    def __init__(self, path: str = "library_db.json") -> None:
        self.path = Path(path)
        self.version = 1
        self.items: Dict[str, LibraryItem] = {}
        self._behavior_index: Dict[str, str] = {}
        self._struct_index: Dict[str, str] = {}
        self._abstract_counts: Counter[Tuple[str, ...]] = Counter()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Populate :attr:`items` from :attr:`path` if it exists."""

        if not self.path.exists():
            self.items = {}
            return
        raw = json.loads(self.path.read_text())
        version = raw.get("version", 1)
        if version != self.version:
            # Future-proofing: keep a best-effort behaviour rather than failing.
            self.version = version
        loaded: Dict[str, LibraryItem] = {}
        behavior_index: Dict[str, str] = {}
        struct_index: Dict[str, str] = {}
        for entry in raw.get("items", []):
            identifier = entry.get("id")
            ops = entry.get("ops", [])
            signature = entry.get("signature", [])
            meta = entry.get("meta", {})
            name = entry.get("name")
            behavior_signature = entry.get("behavior_signature")
            if not identifier or not isinstance(ops, list):
                continue
            if isinstance(meta, dict):
                meta.setdefault("confidence", float(meta.get("confidence", 0.0)))
                meta.setdefault("usage_count", int(meta.get("usage_count", 0)))
            item = LibraryItem(identifier, ops, signature, meta, name, behavior_signature)
            loaded[identifier] = item
            if behavior_signature:
                behavior_index[behavior_signature] = identifier
            struct_sig = _ops_to_program(ops).signature()
            struct_index[struct_sig] = identifier
        self.items = loaded
        self._behavior_index = behavior_index
        self._struct_index = struct_index

    def save(self) -> None:
        """Persist :attr:`items` as JSON."""

        payload = {
            "version": self.version,
            "items": [item.to_json() for item in self.items.values()],
        }
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Registration / accessors
    # ------------------------------------------------------------------
    def register_program(
        self,
        prog: Program,
        train_examples: List[Example],
        name: str | None = None,
        tags: List[str] | None = None,
    ) -> str:
        """Persist ``prog`` as a macro and return its identifier.

        Duplicate programs are ignored – if a macro with the same operation
        signature already exists the existing identifier is returned.
        """

        signature = [op.name for op in prog.ops]
        ops = _program_to_ops(prog)
        key_material = json.dumps(ops, sort_keys=True)
        candidate_id = _stable_id(key_material)
        if candidate_id in self.items:
            return candidate_id

        struct_sig = prog.signature()
        if struct_sig in self._struct_index:
            return self._struct_index[struct_sig]

        behavior_sig = self._program_behavior_signature(prog, train_examples)
        existing = self._behavior_index.get(behavior_sig)
        if existing:
            return existing

        if tags is None:
            tags = []
        train_payload = [{"input": ex.input, "output": ex.output} for ex in train_examples]
        train_blob = json.dumps(train_payload, sort_keys=True)
        train_hash = hashlib.md5(train_blob.encode()).hexdigest()

        meta = {
            "created_at": _now_iso(),
            "train_hash": train_hash,
            "success_count": 1,
            "attempt_count": 1,
            "avg_primary_score": 1.0,
            "avg_secondary_score": 1.0,
            "tags": tags,
            "confidence": 0.8,
            "usage_count": 1,
        }
        item = LibraryItem(candidate_id, ops, signature, meta, name, behavior_sig)
        self.items[candidate_id] = item
        self._behavior_index[behavior_sig] = candidate_id
        self._struct_index[struct_sig] = candidate_id
        return candidate_id

    def iter_macros(self) -> List[Program]:
        """Return library items as :class:`Program` objects."""

        return [_ops_to_program(item.ops) for item in self.items.values()]

    def iter_items(self) -> Iterable[LibraryItem]:
        """Yield :class:`LibraryItem` entries."""

        return self.items.values()

    def touch_success(self, lib_id: str, primary: float, secondary: float) -> None:
        """Update reuse statistics when a macro solves or helps a task."""

        item = self.items.get(lib_id)
        if not item:
            return
        stats = item.meta
        stats["attempt_count"] = int(stats.get("attempt_count", 0)) + 1
        stats["success_count"] = int(stats.get("success_count", 0)) + 1
        prev_primary = float(stats.get("avg_primary_score", 0.0))
        prev_secondary = float(stats.get("avg_secondary_score", 0.0))
        count = max(1, int(stats["success_count"]))
        stats["avg_primary_score"] = prev_primary + (primary - prev_primary) / count
        stats["avg_secondary_score"] = prev_secondary + (secondary - prev_secondary) / count
        score_primary = max(0.0, min(1.0, float(primary)))
        score_secondary = max(0.0, min(1.0, float(secondary)))
        combined = 0.6 * score_primary + 0.4 * score_secondary
        current_conf = float(stats.get("confidence", 0.5))
        stats["confidence"] = min(1.0, 0.9 * current_conf + 0.1 * combined)
        stats["usage_count"] = int(stats.get("usage_count", 0)) + 1

    def record_use(self, lib_id: str) -> None:
        """Record that a macro was evaluated during search."""

        item = self.items.get(lib_id)
        if not item:
            return
        stats = item.meta
        stats["usage_count"] = int(stats.get("usage_count", 0)) + 1

    # ------------------------------------------------------------------
    # Abstraction utilities
    # ------------------------------------------------------------------
    def abstract_from_program(
        self,
        program: Program,
        train_examples: List[Example],
        *,
        min_length: int = 2,
        max_length: int = 4,
        name_prefix: str = "abs",
        include_macros: bool = False,
    ) -> List[str]:
        """Derive higher-level macros from repeated operation sequences.

        The procedure scans ``program`` for contiguous windows of operations and
        counts how frequently each sequence appears across abstraction calls.
        When a sequence has been observed at least twice and is not already
        present in the library, a new macro entry is registered.
        """

        if not program.ops:
            return []
        created: List[str] = []
        windows: List[Tuple[Tuple[str, ...], Sequence[Op]]] = []
        max_length = min(max_length, len(program.ops))
        for length in range(min_length, max_length + 1):
            for start in range(0, len(program.ops) - length + 1):
                slice_ops = program.ops[start : start + length]
                window = tuple(op.name for op in slice_ops)
                windows.append((window, slice_ops))
        for window, slice_ops in windows:
            if not include_macros and any("::" in op.name for op in slice_ops):
                continue
            self._abstract_counts[window] += 1
            if self._abstract_counts[window] < 2:
                continue
            if any(item.signature == list(window) for item in self.items.values()):
                continue
            macro_name = f"{name_prefix}_{len(self.items) + len(created)}"
            macro_id = self.register_sequence_macro(
                slice_ops,
                name=macro_name,
                tags=["abstract", "auto"],
                priority=1.5,
            )
            created.append(macro_id)
        return created

    def grow_from_program(
        self,
        program: Program,
        train_examples: List[Example],
        *,
        mode: str = "none",
    ) -> List[str]:
        """Apply the configured abstraction ``mode`` to ``program``."""

        mode = (mode or "none").lower()
        if mode == "none":
            return []
        include_macros = mode == "hierarchical"
        created = self.abstract_from_program(
            program,
            train_examples,
            include_macros=include_macros,
        )
        if include_macros and created:
            # Re-run once to allow newly-created macros to combine.
            self.abstract_from_program(program, train_examples, include_macros=True)
        return created

    def register_sequence_macro(
        self,
        ops: Sequence[Op],
        *,
        name: str | None = None,
        tags: List[str] | None = None,
        priority: float = 1.0,
    ) -> str:
        """Register a macro from a raw operation sequence without examples."""

        program = Program([Op(op.name, dict(op.params)) for op in ops])
        signature = [op.name for op in program.ops]
        ops_payload = _program_to_ops(program)
        key_material = json.dumps(ops_payload, sort_keys=True)
        candidate_id = _stable_id(key_material)
        if candidate_id in self.items:
            return candidate_id
        meta = {
            "created_at": _now_iso(),
            "tags": tags or ["abstract"],
            "abstract": True,
            "priority": priority,
            "confidence": max(priority, 0.5),
            "usage_count": 0,
        }
        item = LibraryItem(candidate_id, ops_payload, signature, meta, name, None)
        self.items[candidate_id] = item
        self._struct_index[program.signature()] = candidate_id
        return candidate_id

    def _program_behavior_signature(self, prog: Program, train_examples: List[Example]) -> str:
        """Return a canonical signature for ``prog`` on ``train_examples``."""

        outputs: List[Any] = []
        for example in train_examples:
            try:
                outputs.append(prog.apply(example.input))
            except Exception:
                outputs.append([[0]])
        blob = json.dumps(make_hashable(outputs), sort_keys=True, separators=(",", ":"))
        return hashlib.md5(blob.encode()).hexdigest()


__all__ = ["LibraryManager", "LibraryItem"]
