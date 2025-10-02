"""reap.memory
================

Persistence helpers for caching solutions between runs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List

from .constants import MEMORY_DB
from .types import Grid


def hash_train(train: List[Dict[str, Grid]]) -> str:
    """Stable hash of training pairs used as a memory key."""

    return hashlib.md5(json.dumps(train, sort_keys=True).encode()).hexdigest()


def load_memory_db() -> Dict[str, List[Grid]]:
    """Load memoised solutions from :data:`MEMORY_DB`."""

    path = Path(MEMORY_DB)
    return json.loads(path.read_text()) if path.exists() else {}


def save_memory_db(db: Dict[str, List[Grid]]) -> None:
    """Persist ``db`` to :data:`MEMORY_DB` with indentation for readability."""

    Path(MEMORY_DB).write_text(json.dumps(db, indent=2))


__all__ = ["hash_train", "load_memory_db", "save_memory_db"]
