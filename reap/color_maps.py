"""reap.color_maps
===================

Heuristics for inferring colour remappings between input and output grids. These
helpers provide small yet crucial hints to the search module by precomputing
potential ``map_color`` arguments from the training data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .grid_utils import dims
from .types import Color, Grid


def infer_color_map(grid_in: Grid, grid_out: Grid) -> Optional[Dict[Color, Color]]:
    """Infer a bijection between colours of ``grid_in`` and ``grid_out``.

    Returns ``None`` when dimensions mismatch or when a colour would need to map
    to two different targets. The implementation mirrors the behaviour of the
    original script but is now easier to reference.
    """

    if dims(grid_in) != dims(grid_out):
        return None
    mapping: Dict[Color, Color] = {}
    mapped_values: set[Color] = set()
    height, width = dims(grid_in)
    for r in range(height):
        for c in range(width):
            source, target = grid_in[r][c], grid_out[r][c]
            if source == target:
                continue
            if source in mapping and mapping[source] != target:
                return None
            mapping[source] = target
            mapped_values.add(target)
    if len(mapped_values) < len(mapping):
        return None
    return mapping


def infer_color_maps_from_train(task: any) -> List[Dict[int, int]]:
    """Aggregate compatible colour maps across all training pairs."""

    per_pair = []
    for pair in getattr(task, "train", []):
        if dims(pair.input) != dims(pair.output):
            return []
        mapping = infer_color_map(pair.input, pair.output)
        if mapping is None:
            return []
        per_pair.append(mapping)
    if not per_pair:
        return []
    merged: Dict[int, int] = {}
    for mapping in per_pair:
        for key, value in mapping.items():
            if key in merged and merged[key] != value:
                return []
            merged[key] = value
    return [merged] if merged else []


__all__ = [
    "infer_color_map",
    "infer_color_maps_from_train",
]
