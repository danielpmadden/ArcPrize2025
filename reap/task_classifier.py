"""Task family classification heuristics for REAP.

The module inspects training pairs and emits lightweight feature flags that
capture the dominant transformation style present in the task. These signals are
used to bias operator ordering inside the search without committing to any
irreversible pruning so the solver remains sound.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .dsl import flip, rotate
from .grid_utils import dims
from .objects import extract_objects
from .types import Example


@dataclass(frozen=True)
class TaskProfile:
    """Summary of task properties used for downstream heuristics."""

    families: Sequence[str]
    feature_flags: Dict[str, Any]
    prioritised_ops: List[str]


def _color_histogram(grid: Sequence[Sequence[int]]) -> Counter[int]:
    hist: Counter[int] = Counter()
    for row in grid:
        for value in row:
            hist[value] += 1
    return hist


def _has_symmetry_pair(example: Example) -> bool:
    """Return ``True`` when input/output relate by a flip or rotation."""

    inp, out = example.input, example.output
    if dims(inp) != dims(out):
        return False
    for axis in ("h", "v"):
        if flip(inp, axis) == out:
            return True
    for angle in (90, 180, 270):
        if rotate(inp, angle) == out:
            return True
    return False


def _detect_repetition(train: Sequence[Example]) -> bool:
    """Detect tiling/repetition by comparing grid dimensions."""

    for example in train:
        in_dims = dims(example.input)
        out_dims = dims(example.output)
        if in_dims == out_dims:
            continue
        if in_dims[0] and in_dims[1]:
            if out_dims[0] % in_dims[0] == 0 and out_dims[1] % in_dims[1] == 0:
                return True
    return False


def _detect_object_motion(train: Sequence[Example]) -> bool:
    """Detect when objects move or are copied without changing counts."""

    try:
        for example in train:
            src_objects = extract_objects(example.input)
            dst_objects = extract_objects(example.output)
            if len(src_objects) != len(dst_objects):
                return False
            # Compare bounding boxes; if any differs, treat as movement.
            for src, dst in zip(src_objects, dst_objects):
                if src.bounding_box != dst.bounding_box:
                    return True
    except Exception:
        return False
    return False


def _detect_color_shift(train: Sequence[Example]) -> bool:
    """Detect large colour distribution changes."""

    for example in train:
        if _color_histogram(example.input) != _color_histogram(example.output):
            return True
    return False


def classify_task(train: Sequence[Example]) -> TaskProfile:
    """Classify task families and propose operator biases.

    The classifier is deliberately permissive – the returned families are
    heuristics rather than mutually exclusive labels. Downstream callers can use
    the ``prioritised_ops`` list to gently re-order enumeration without removing
    any operations entirely.
    """

    families: List[str] = []
    feature_flags: Dict[str, Any] = {}
    prioritised_ops: List[str] = []

    if not train:
        return TaskProfile(families, feature_flags, prioritised_ops)

    if all(_has_symmetry_pair(example) for example in train):
        families.append("symmetry")
        prioritised_ops.extend(["flip", "rotate", "mirror_symmetry", "complete_symmetry"])
        feature_flags["symmetry"] = True
    else:
        feature_flags["symmetry"] = False

    if _detect_repetition(train):
        families.append("repetition")
        prioritised_ops.extend(["tile_to_target", "repeat_scale", "repeat_objects_horiz", "distribute_evenly"])
        feature_flags["repetition"] = True
    else:
        feature_flags["repetition"] = False

    if _detect_object_motion(train):
        families.append("object_motion")
        prioritised_ops.extend(["translate_object", "translate_all_by_centroid", "copy_dominant_object", "align_to_edge"])
        feature_flags["object_motion"] = True
    else:
        feature_flags["object_motion"] = False

    if _detect_color_shift(train):
        families.append("color_shift")
        prioritised_ops.extend(["map_color", "colorize_by_size", "remove_color", "keep_objects_with_color"])
        feature_flags["color_shift"] = True
    else:
        feature_flags["color_shift"] = False

    feature_flags["families"] = tuple(families)
    # Deduplicate while preserving order
    seen = set()
    prioritised_ops_unique = []
    for name in prioritised_ops:
        if name in seen:
            continue
        seen.add(name)
        prioritised_ops_unique.append(name)

    return TaskProfile(tuple(families), feature_flags, prioritised_ops_unique)


__all__ = ["TaskProfile", "classify_task"]
