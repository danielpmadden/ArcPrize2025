"""reap.memory
================

Persistence helpers for caching solutions between runs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from .constants import MEMORY_DB
from .types import Grid


def hash_train(train: List[Dict[str, Grid]]) -> str:
    """Stable hash of training pairs used as a memory key."""

    return hashlib.md5(json.dumps(train, sort_keys=True).encode()).hexdigest()


def _upgrade_legacy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise legacy payloads that pre-date rule confidence tracking."""

    if "solutions" in payload or "rule_confidence" in payload:
        payload.setdefault("solutions", {})
        payload.setdefault("rule_confidence", {})
        payload.setdefault("task_family_stats", {})
        return payload
    return {"solutions": payload, "rule_confidence": {}, "task_family_stats": {}}


def load_memory_db() -> Dict[str, Any]:
    """Load memoised solutions and auxiliary stats from :data:`MEMORY_DB`."""

    path = Path(MEMORY_DB)
    if not path.exists():
        return {"solutions": {}, "rule_confidence": {}, "task_family_stats": {}}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return {"solutions": {}, "rule_confidence": {}, "task_family_stats": {}}
    return _upgrade_legacy_payload(raw)


def save_memory_db(db: Dict[str, Any]) -> None:
    """Persist ``db`` to :data:`MEMORY_DB` with indentation for readability."""

    payload = _upgrade_legacy_payload(db)
    Path(MEMORY_DB).write_text(json.dumps(payload, indent=2))


__all__ = ["hash_train", "load_memory_db", "save_memory_db"]
