"""reap.objects
================

Object-centric utilities: extracting connected components, redrawing them, and
running higher-level manipulations like centroid translation. The legacy script
mixed these routines with unrelated grid math, which made maintenance painful.
By isolating them here we can reason about object lifecycles in one place and
add new behaviours without accidentally touching search code.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from .dsl import map_color, rotate
from .grid_utils import deepcopy_grid, dims, grid_to_template, make_grid
from .types import Grid, Template


@lru_cache(maxsize=4096)
def _cached_extract_objects(grid_tuple: Tuple[Tuple[int, ...], ...], connectivity: int = 4):
    """Return object metadata for ``grid_tuple`` using BFS flood fill.

    Parameters
    ----------
    grid_tuple:
        Hashable representation of the grid (tuple of tuples) used as a cache key.
    connectivity:
        Either 4 or 8; determines neighbourhood structure during extraction.

    Returns
    -------
    list[dict]
        Metadata dictionaries describing discovered objects. Each entry contains
        the original colour, bounding box, cropped grid, binary shape template,
        and pixel count. The return structure intentionally mirrors the original
        implementation so downstream logic can remain unchanged.
    """

    height = len(grid_tuple)
    width = len(grid_tuple[0]) if grid_tuple else 0
    grid = [list(row) for row in grid_tuple]
    visited = set()
    objects: List[Dict[str, Any]] = []
    directions = (
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if connectivity == 4
        else [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    )

    for r in range(height):
        for c in range(width):
            if grid[r][c] == 0 or (r, c) in visited:
                continue
            colour = grid[r][c]
            queue = deque([(r, c)])
            pixels = {(r, c)}
            visited.add((r, c))
            while queue:
                pr, pc = queue.popleft()
                for dr, dc in directions:
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if (nr, nc) not in visited and grid[nr][nc] == colour:
                            visited.add((nr, nc))
                            pixels.add((nr, nc))
                            queue.append((nr, nc))
            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            bbox = (min(rows), min(cols), max(rows) + 1, max(cols) + 1)
            obj_grid = [[grid[y][x] if (y, x) in pixels else 0 for x in range(width)] for y in range(height)]
            crop = [row[bbox[1]:bbox[3]] for row in obj_grid[bbox[0]:bbox[2]]]
            shape = [[1 if value != 0 else 0 for value in row] for row in crop]
            objects.append({
                "color": colour,
                "pixels": pixels,
                "bbox": bbox,
                "grid": crop,
                "shape": grid_to_template(shape),
                "size": len(pixels),
            })
    return objects


def extract_objects(grid: Grid, connectivity: int = 4) -> List[Dict[str, Any]]:
    """Public wrapper for cached object extraction."""

    if not grid:
        return []
    grid_tuple = tuple(tuple(row) for row in grid)
    return _cached_extract_objects(grid_tuple, connectivity)


def draw_objects(height: int, width: int, objects: List[Dict[str, Any]], bg: int = 0) -> Grid:
    """Render object crops back into a grid of size ``height`` x ``width``."""

    out = make_grid(height, width, bg)
    for obj in objects:
        r0, c0, _, _ = obj["bbox"]
        gh, gw = dims(obj["grid"])
        for rr in range(gh):
            for cc in range(gw):
                value = obj["grid"][rr][cc]
                if value != 0 and 0 <= r0 + rr < height and 0 <= c0 + cc < width:
                    out[r0 + rr][c0 + cc] = value
    return out


def learn_object_transformation_maps(task: Any) -> Dict[Template, Dict[str, Any]]:
    """Infer per-shape transformation hints from training pairs."""

    mapping: Dict[Template, Dict[str, Any]] = {}
    for pair in getattr(task, "train", []):
        in_objects = extract_objects(pair.input)
        out_objects = extract_objects(pair.output)
        if len(in_objects) != len(out_objects):
            continue
        for obj in in_objects:
            candidates = [cand for cand in out_objects if cand["color"] == obj["color"]]
            if len(candidates) != 1:
                continue
            target = candidates[0]
            shape = obj["shape"]
            rule = mapping.setdefault(shape, {})
            if obj["color"] != target["color"]:
                rule["color_map"] = {obj["color"]: target["color"]}
            ishaped = [[1 if v != 0 else 0 for v in row] for row in obj["grid"]]
            oshaped = [[1 if v != 0 else 0 for v in row] for row in target["grid"]]
            for angle in (90, 180, 270):
                if rotate(ishaped, angle) == oshaped:
                    rule["rotate"] = angle
                    break
    return mapping


def transform_by_object_template(grid: Grid, transform_map: Dict[Template, Dict[str, Any]]) -> Grid:
    """Apply learned per-shape transforms to every object in ``grid``."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    new_objects = []
    for obj in objects:
        rule = transform_map.get(obj["shape"])
        new_grid = deepcopy_grid(obj["grid"])
        if rule:
            if "color_map" in rule:
                new_grid = map_color(new_grid, rule["color_map"])
            if "rotate" in rule:
                new_grid = rotate(new_grid, rule["rotate"])
        new_objects.append({"color": obj["color"], "pixels": None, "bbox": obj["bbox"], "grid": new_grid})
    return draw_objects(height, width, new_objects)


def translate_object(grid: Grid, dr: int, dc: int) -> Grid:
    """Translate every object by ``(dr, dc)`` ignoring collisions."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    if not objects:
        return deepcopy_grid(grid)
    out = make_grid(height, width)
    for obj in objects:
        r0, c0, _, _ = obj["bbox"]
        gh, gw = dims(obj["grid"])
        for rr in range(gh):
            for cc in range(gw):
                value = obj["grid"][rr][cc]
                if value != 0:
                    nr, nc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        out[nr][nc] = value
    return out


def place_in_center(grid: Grid) -> Grid:
    """Reposition each object so its bounding box is centred."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    if not objects:
        return deepcopy_grid(grid)
    out = make_grid(height, width)
    for obj in objects:
        gh, gw = dims(obj["grid"])
        r0 = max(0, (height - gh) // 2)
        c0 = max(0, (width - gw) // 2)
        for rr in range(gh):
            for cc in range(gw):
                value = obj["grid"][rr][cc]
                if value != 0:
                    out[r0 + rr][c0 + cc] = value
    return out


def place_in_corner(grid: Grid, which: str = "tl") -> Grid:
    """Place each object in a selected corner (``tl``, ``tr``, ``bl``, ``br``)."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    if not objects:
        return deepcopy_grid(grid)
    out = make_grid(height, width)
    for obj in objects:
        gh, gw = dims(obj["grid"])
        if which == "tl":
            r0, c0 = 0, 0
        elif which == "tr":
            r0, c0 = 0, width - gw
        elif which == "bl":
            r0, c0 = height - gh, 0
        else:
            r0, c0 = height - gh, width - gw
        for rr in range(gh):
            for cc in range(gw):
                value = obj["grid"][rr][cc]
                if value != 0:
                    out[r0 + rr][c0 + cc] = value
    return out


def remove_color(grid: Grid, color: int) -> Grid:
    """Replace every instance of ``color`` with zero."""

    height, width = dims(grid)
    return [[0 if grid[r][c] == color else grid[r][c] for c in range(width)] for r in range(height)]


def keep_largest_object(grid: Grid) -> Grid:
    """Keep only the object with the largest pixel count."""

    objects = extract_objects(grid)
    if not objects:
        return deepcopy_grid(grid)
    largest = max(objects, key=lambda obj: obj["size"])
    height, width = dims(grid)
    return draw_objects(height, width, [largest])


def outline_object(grid: Grid) -> Grid:
    """Return only the outline of objects (four-neighbourhood)."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    out = make_grid(height, width)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for obj in objects:
        for r in range(height):
            for c in range(width):
                if obj["grid"][r][c] != 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and obj["grid"][nr][nc] == 0:
                            out[r][c] = obj["grid"][r][c]
    return out


def fill_object(grid: Grid, color: int) -> Grid:
    """Flood-fill each object with ``color``."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    out = make_grid(height, width)
    for obj in objects:
        for r in range(height):
            for c in range(width):
                if obj["grid"][r][c] != 0:
                    out[r][c] = color
    return out


def translate_all_by_centroid(grid: Grid, target_r: int, target_c: int) -> Grid:
    """Move all pixels so that the centroid matches ``(target_r, target_c)``."""

    height, width = dims(grid)
    objects = extract_objects(grid)
    if not objects:
        return deepcopy_grid(grid)
    all_pixels = [pixel for obj in objects for pixel in obj["pixels"]]
    centroid_r = sum(p[0] for p in all_pixels) // len(all_pixels)
    centroid_c = sum(p[1] for p in all_pixels) // len(all_pixels)
    delta_r, delta_c = target_r - centroid_r, target_c - centroid_c
    out = make_grid(height, width)
    for r, c in all_pixels:
        value = grid[r][c]
        nr, nc = r + delta_r, c + delta_c
        if 0 <= nr < height and 0 <= nc < width:
            out[nr][nc] = value
    return out


__all__ = [
    "extract_objects",
    "draw_objects",
    "learn_object_transformation_maps",
    "transform_by_object_template",
    "translate_object",
    "place_in_center",
    "place_in_corner",
    "remove_color",
    "keep_largest_object",
    "outline_object",
    "fill_object",
    "translate_all_by_centroid",
]
