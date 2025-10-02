"""reap.operations
===================

Central registry mapping operation names to callables. The search engine relies
on this dictionary to instantiate programs dynamically. By keeping the mapping in
one module we avoid circular dependencies and can document rationale for each
inclusion.
"""

from __future__ import annotations

from typing import Callable, Dict

from .dsl import (
    compress_block,
    crop,
    expand_block,
    fill_holes,
    flip,
    grow_block,
    map_color,
    overlay,
    pad,
    repeat_pattern,
    repeat_scale,
    replace_region,
    rotate,
    tile_to_target,
    tile_with_padding,
    mirror_symmetry,
)
from .objects import (
    fill_object,
    keep_largest_object,
    learn_object_transformation_maps,
    outline_object,
    place_in_center,
    place_in_corner,
    remove_color,
    transform_by_object_template,
    translate_all_by_centroid,
    translate_object,
)

FUNCTION_REGISTRY: Dict[str, Callable] = {
    "rotate": rotate,
    "flip": flip,
    "pad": pad,
    "crop": crop,
    "overlay": overlay,
    "map_color": map_color,
    "tile_to_target": tile_to_target,
    "repeat_scale": repeat_scale,
    "fill_holes": fill_holes,
    "mirror_symmetry": mirror_symmetry,
    "compress_block": compress_block,
    "expand_block": expand_block,
    "repeat_pattern": repeat_pattern,
    "tile_with_padding": tile_with_padding,
    "replace_region": replace_region,
    "grow_block": grow_block,
    "transform_by_object_template": transform_by_object_template,
    "translate_all_by_centroid": translate_all_by_centroid,
    "translate_object": translate_object,
    "place_in_center": place_in_center,
    "place_in_corner": place_in_corner,
    "remove_color": remove_color,
    "keep_largest_object": keep_largest_object,
    "outline_object": outline_object,
    "fill_object": fill_object,
}


def get_operation(name: str) -> Callable:
    """Lookup ``name`` in :data:`FUNCTION_REGISTRY` with a helpful error."""

    try:
        return FUNCTION_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown operation {name!r}. Registry keys: {sorted(FUNCTION_REGISTRY)}") from exc


__all__ = ["FUNCTION_REGISTRY", "get_operation", "learn_object_transformation_maps"]
