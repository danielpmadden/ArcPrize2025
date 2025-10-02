"""reap.constants
=================

Global constants used across the solver. Keeping them here avoids import cycles
between modules and makes it easier to discover configurable paths.
"""

from __future__ import annotations

MEMORY_DB = "memory_db.json"
FAIL_LOG = "missing_ops.jsonl"

__all__ = ["MEMORY_DB", "FAIL_LOG"]
