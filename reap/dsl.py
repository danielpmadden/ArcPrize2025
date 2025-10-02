"""reap.dsl
================

Functional transformations that operate directly on grids. This module mirrors
what used to be the bulk of the monolithic REAP script: grid rotations, flips,
padding, cropping, and more experimental operators such as block compression.

By collocating these functions here we keep higher-level orchestration modules
(e.g., search or solver heuristics) focused on strategy rather than pixel-level
manipulation. Every operation is wrapped with extensive documentation describing
side effects, invariants, and rationale so the next person picking up the code
has a trustworthy reference.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable, Tuple

import numpy as np

from .grid_utils import clamp, deepcopy_grid, dims, make_grid
from .types import Color, Grid


def enforce_invariants(fn):
    """Decorator ensuring ARC grid invariants after an operation.

    The original script sprinkled defensive assertions throughout individual
    functions. Wrapping them in a decorator keeps the checks centralised while
    still giving each operator a clear, declarative implementation.

    Parameters
    ----------
    fn:
        Callable implementing a grid transformation.

    Returns
    -------
    callable
        Wrapped function that validates the returned grid before handing it back
        to the caller.
    """

    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if not isinstance(out, list):
            raise AssertionError(f"{fn.__name__}: expected list grid, got {type(out)!r}")
        if out:
            width = len(out[0])
            for row in out:
                if len(row) != width:
                    raise AssertionError(f"{fn.__name__}: inconsistent row widths")
                for value in row:
                    if not isinstance(value, int):
                        raise AssertionError(f"{fn.__name__}: non-int cell {value!r}")
                    if not 0 <= value <= 9:
                        raise AssertionError(f"{fn.__name__}: colour {value} outside 0..9")
        return out

    return wrapper


@enforce_invariants
def rotate(grid: Grid, angle: int) -> Grid:
    """Rotate ``grid`` clockwise by the requested angle.

    Parameters
    ----------
    grid:
        Source grid to rotate.
    angle:
        Rotation angle in degrees. The implementation normalises the value into
        ``{0, 90, 180, 270}`` and falls back to a copy for unsupported inputs.
    """

    height, width = dims(grid)
    angle %= 360
    if angle == 0:
        return deepcopy_grid(grid)
    if angle == 90:
        out = make_grid(width, height)
        for r in range(height):
            for c in range(width):
                out[c][height - 1 - r] = grid[r][c]
        return out
    if angle == 180:
        out = make_grid(height, width)
        for r in range(height):
            for c in range(width):
                out[height - 1 - r][width - 1 - c] = grid[r][c]
        return out
    if angle == 270:
        out = make_grid(width, height)
        for r in range(height):
            for c in range(width):
                out[width - 1 - c][r] = grid[r][c]
        return out
    return deepcopy_grid(grid)


@enforce_invariants
def flip(grid: Grid, axis: str) -> Grid:
    """Mirror ``grid`` along the requested axis."""

    height, width = dims(grid)
    out = make_grid(height, width)
    if axis == "h":
        for r in range(height):
            for c in range(width):
                out[r][width - 1 - c] = grid[r][c]
    elif axis == "v":
        for r in range(height):
            for c in range(width):
                out[height - 1 - r][c] = grid[r][c]
    else:
        out = deepcopy_grid(grid)
    return out


@enforce_invariants
def pad(grid: Grid, top: int, bottom: int, left: int, right: int, value: int = 0) -> Grid:
    """Pad ``grid`` with ``value``.

    The helper mirrors NumPy's pad semantics but remains intentionally simple so
    it is easy to audit and debug in pure Python.
    """

    height, width = dims(grid)
    out = make_grid(height + top + bottom, width + left + right, value)
    for r in range(height):
        for c in range(width):
            out[r + top][c + left] = grid[r][c]
    return out


@enforce_invariants
def crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Crop ``grid`` to the provided bounding box.

    Coordinates are clamped to valid ranges so callers can pass slightly noisy
    inputs without risking crashes. Returning an empty grid when ``r1 <= r0`` or
    ``c1 <= c0`` mimics the behaviour from the original script.
    """

    height, width = dims(grid)
    r0, r1 = clamp(r0, 0, height), clamp(r1, 0, height)
    c0, c1 = clamp(c0, 0, width), clamp(c1, 0, width)
    if r1 <= r0 or c1 <= c0:
        return []
    return [row[c0:c1] for row in grid[r0:r1]]


@enforce_invariants
def overlay(base: Grid, top: Grid, mode: str = "top_nonzero_over") -> Grid:
    """Composite ``top`` onto ``base`` using a simple alpha rule."""

    base_h, base_w = dims(base)
    top_h, top_w = dims(top)
    if (base_h, base_w) != (top_h, top_w):
        return deepcopy_grid(base)
    out = deepcopy_grid(base)
    for r in range(base_h):
        for c in range(base_w):
            value = top[r][c]
            if mode == "top_nonzero_over" and value != 0:
                out[r][c] = value
    return out


@enforce_invariants
def map_color(grid: Grid, color_map: Dict[Color, Color]) -> Grid:
    """Apply a lookup-based recolouring."""

    height, width = dims(grid)
    return [[color_map.get(grid[r][c], grid[r][c]) for c in range(width)] for r in range(height)]


@enforce_invariants
def complete_symmetry(grid: Grid, axis: str = "h") -> Grid:
    """Fill missing halves by mirroring content across ``axis``.

    The function keeps existing pixels and only copies colours into empty cells.
    ``axis`` follows the same convention as :func:`flip` (``"h"`` mirrors along
    the vertical axis, ``"v"`` along the horizontal axis).
    """

    out = deepcopy_grid(grid)
    height, width = dims(grid)
    if axis not in {"h", "v"}:
        return out
    if axis == "h":
        for r in range(height):
            for c in range(width // 2):
                mirror = width - 1 - c
                left, right = out[r][c], out[r][mirror]
                if left and not right:
                    out[r][mirror] = left
                elif right and not left:
                    out[r][c] = right
    else:
        for c in range(width):
            for r in range(height // 2):
                mirror = height - 1 - r
                top, bottom = out[r][c], out[mirror][c]
                if top and not bottom:
                    out[mirror][c] = top
                elif bottom and not top:
                    out[r][c] = bottom
    return out


@enforce_invariants
def tile_to_target(grid: Grid, target_h: int, target_w: int) -> Grid:
    """Tile ``grid`` until reaching ``target_h`` x ``target_w`` dimensions."""

    height, width = dims(grid)
    if height == 0 or width == 0 or target_h <= 0 or target_w <= 0:
        return []
    out = make_grid(target_h, target_w)
    for r in range(target_h):
        source_row = grid[r % height]
        for c in range(target_w):
            out[r][c] = source_row[c % width]
    return out


@enforce_invariants
def repeat_scale(grid: Grid, factor: int) -> Grid:
    """Nearest-neighbour upscale by ``factor``."""

    if factor <= 1:
        return deepcopy_grid(grid)
    height, width = dims(grid)
    out = make_grid(height * factor, width * factor)
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            row_start, col_start = r * factor, c * factor
            for dr in range(factor):
                for dc in range(factor):
                    out[row_start + dr][col_start + dc] = value
    return out


@enforce_invariants
def project_profile(grid: Grid, axis: str = "h") -> Grid:
    """Collapse a grid along ``axis`` using logical OR style projection.

    The function preserves the grid dimensions by filling entire rows or columns
    with the dominant non-zero colour encountered along that slice. This is
    useful for tasks requiring silhouettes or bars to be extrapolated.
    """

    height, width = dims(grid)
    out = make_grid(height, width)
    if axis == "h":
        for c in range(width):
            column = [grid[r][c] for r in range(height) if grid[r][c] != 0]
            if not column:
                continue
            colour = Counter(column).most_common(1)[0][0]
            for r in range(height):
                out[r][c] = colour
    else:
        for r in range(height):
            row_vals = [grid[r][c] for c in range(width) if grid[r][c] != 0]
            if not row_vals:
                continue
            colour = Counter(row_vals).most_common(1)[0][0]
            for c in range(width):
                out[r][c] = colour
    return out


@enforce_invariants
def flood_fill_from(grid: Grid, color_src: int, color_dst: int) -> Grid:
    """Flood fill contiguous regions of ``color_src`` with ``color_dst``."""

    if color_src == color_dst:
        return deepcopy_grid(grid)
    height, width = dims(grid)
    out = deepcopy_grid(grid)
    visited = set()
    for r in range(height):
        for c in range(width):
            if out[r][c] != color_src or (r, c) in visited:
                continue
            queue = deque([(r, c)])
            visited.add((r, c))
            while queue:
                pr, pc = queue.popleft()
                out[pr][pc] = color_dst
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                        if out[nr][nc] == color_src:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return out


@enforce_invariants
def fill_holes(grid: Grid, fill_color: int = 0) -> Grid:
    """Fill fully enclosed zero regions with ``fill_color``."""

    height, width = dims(grid)
    out = deepcopy_grid(grid)
    visited = [[False] * width for _ in range(height)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for r in range(height):
        for c in range(width):
            if out[r][c] != 0 or visited[r][c]:
                continue
            region = []
            touches_border = False
            queue = deque([(r, c)])
            visited[r][c] = True
            while queue:
                pr, pc = queue.popleft()
                region.append((pr, pc))
                if pr in (0, height - 1) or pc in (0, width - 1):
                    touches_border = True
                for dr, dc in directions:
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and out[nr][nc] == 0:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            if not touches_border:
                for pr, pc in region:
                    out[pr][pc] = fill_color
    return out


@enforce_invariants
def mirror_symmetry(grid: Grid, axis: str) -> Grid:
    """Mirror the grid to enforce symmetry along ``axis`` (``'h'`` or ``'v'``)."""

    height, width = dims(grid)
    out = deepcopy_grid(grid)
    if axis == "h":
        for r in range(height):
            for c in range(width // 2):
                out[r][width - 1 - c] = grid[r][c]
    elif axis == "v":
        for r in range(height // 2):
            for c in range(width):
                out[height - 1 - r][c] = grid[r][c]
    return out


@enforce_invariants
def compress_block(grid: Grid, factor: int) -> Grid:
    """Downsample ``grid`` by grouping ``factor`` x ``factor`` blocks."""

    if factor <= 0:
        return deepcopy_grid(grid)
    height, width = dims(grid)
    if height == 0 or width == 0:
        return []
    out_h, out_w = height // factor, width // factor
    out = np.zeros((out_h, out_w), dtype=int)
    for i in range(out_h):
        for j in range(out_w):
            block = [grid[ii][jj]
                     for ii in range(i * factor, (i + 1) * factor)
                     for jj in range(j * factor, (j + 1) * factor)]
            values = [v for v in block if v != 0]
            out[i, j] = max(set(values), key=values.count) if values else 0
    return out.tolist()


@enforce_invariants
def expand_block(grid: Grid, factor: int) -> Grid:
    """Expand each cell into an ``factor`` x ``factor`` block."""

    if factor <= 1:
        return deepcopy_grid(grid)
    arr = np.array(grid)
    out = arr.repeat(factor, axis=0).repeat(factor, axis=1)
    return out.tolist()


@enforce_invariants
def repeat_pattern(grid: Grid, stride: int) -> Grid:
    """Repeat non-zero cells horizontally with a fixed stride."""

    arr = np.array(grid)
    height, width = arr.shape if arr.size else (0, 0)
    out = arr.copy()
    for i in range(height):
        for j in range(width):
            if arr[i, j] != 0:
                for k in range(1, stride):
                    if j + k < width:
                        out[i, j + k] = arr[i, j]
    return out.tolist()


@enforce_invariants
def tile_with_padding(grid: Grid, pad: int, direction: str = "right") -> Grid:
    """Pad the grid on the specified side using zeros."""

    arr = np.array(grid)
    if direction == "right":
        out = np.pad(arr, ((0, 0), (pad, pad)), constant_values=0)
    elif direction == "down":
        out = np.pad(arr, ((pad, pad), (0, 0)), constant_values=0)
    else:
        out = np.pad(arr, ((pad, pad), (pad, pad)), constant_values=0)
    return out.tolist()


@enforce_invariants
def replace_region(grid: Grid, color: int, bounds: Tuple[int, int, int, int]) -> Grid:
    """Overwrite a rectangular region with ``color``."""

    arr = np.array(grid)
    r1, r2, c1, c2 = bounds
    arr[r1:r2, c1:c2] = color
    return arr.tolist()


@enforce_invariants
def grow_block(grid: Grid, target_color: int) -> Grid:
    """Grow non-zero cells to their eight-neighbourhood using ``target_color``."""

    arr = np.array(grid)
    height, width = arr.shape if arr.size else (0, 0)
    out = arr.copy()
    for i in range(height):
        for j in range(width):
            if arr[i, j] != 0:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and out[ni, nj] == 0:
                            out[ni, nj] = target_color
    return out.tolist()


__all__ = [
    "enforce_invariants",
    "rotate",
    "flip",
    "pad",
    "crop",
    "overlay",
    "map_color",
    "complete_symmetry",
    "tile_to_target",
    "repeat_scale",
    "project_profile",
    "fill_holes",
    "mirror_symmetry",
    "compress_block",
    "expand_block",
    "repeat_pattern",
    "tile_with_padding",
    "replace_region",
    "grow_block",
    "flood_fill_from",
]
