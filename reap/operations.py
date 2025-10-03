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
    complete_symmetry,
    crop,
    expand_block,
    flood_fill_from,
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
    project_profile,
    tile_to_target,
    tile_with_padding,
    mirror_symmetry,
)
from .objects import (
    align_to_edge,
    colorize_by_size,
    copy_dominant_object,
    distribute_evenly,
    fill_object,
    find_shapes,
    keep_largest_object,
    learn_object_transformation_maps,
    outline_object,
    place_in_center,
    place_in_corner,
    remove_small_objects,
    remove_color,
    repeat_objects_horiz,
    snap_to_grid,
    transform_by_object_template,
    translate_all_by_centroid,
    translate_object,
    keep_objects_with_color,
)

FUNCTION_REGISTRY: Dict[str, Callable] = {
    "rotate": rotate,
    "flip": flip,
    "pad": pad,
    "crop": crop,
    "overlay": overlay,
    "map_color": map_color,
    "complete_symmetry": complete_symmetry,
    "tile_to_target": tile_to_target,
    "repeat_scale": repeat_scale,
    "project_profile": project_profile,
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
    "find_shapes": find_shapes,
    "remove_color": remove_color,
    "keep_largest_object": keep_largest_object,
    "outline_object": outline_object,
    "fill_object": fill_object,
    "copy_dominant_object": copy_dominant_object,
    "align_to_edge": align_to_edge,
    "colorize_by_size": colorize_by_size,
    "remove_small_objects": remove_small_objects,
    "keep_objects_with_color": keep_objects_with_color,
    "repeat_objects_horiz": repeat_objects_horiz,
    "distribute_evenly": distribute_evenly,
    "snap_to_grid": snap_to_grid,
    "flood_fill_from": flood_fill_from,
}


def get_operation(name: str) -> Callable:
    """Lookup ``name`` in :data:`FUNCTION_REGISTRY` with a helpful error."""

    try:
        return FUNCTION_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown operation {name!r}. Registry keys: {sorted(FUNCTION_REGISTRY)}") from exc


__all__ = ["FUNCTION_REGISTRY", "get_operation", "learn_object_transformation_maps"]
