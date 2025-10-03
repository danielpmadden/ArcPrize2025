"""reap.logging_utils
======================

Simple logging utilities, mainly for recording unsolved tasks for later DSL
audits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import FAIL_LOG


def log_missing(task_id: str, task: Any) -> None:
    """Append a JSON line describing ``task`` to :data:`FAIL_LOG`."""

    entry = {
        "task_id": task_id,
        "train": [{"input": pair.input, "output": pair.output} for pair in getattr(task, "train", [])],
        "test": [{"input": item.input} for item in getattr(task, "test", [])],
    }
    with Path(FAIL_LOG).open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


__all__ = ["log_missing"]
