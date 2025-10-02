"""Utility helpers and DSL operations for manipulating ARC grids."""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, List, Tuple

from .types import Color, Grid


def dims(grid: Grid) -> Tuple[int, int]:
    return (0, 0) if not grid or not grid[0] else (len(grid), len(grid[0]))


def deepcopy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def eq_grid(a: Grid, b: Grid) -> bool:
    return a == b


def valid_grid(grid: Grid) -> bool:
    if not isinstance(grid, list) or not grid:
        return True
    if not all(isinstance(row, list) for row in grid):
        return False
    if not grid[0]:
        return True
    width = len(grid[0])
    return all(
        len(row) == width and all(isinstance(v, int) and 0 <= v <= 9 for v in row)
        for row in grid
    )


def make_grid(height: int, width: int, fill: int = 0) -> Grid:
    return [[fill] * width for _ in range(height)]


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def grid_to_template(grid: Grid):
    return tuple(tuple(row) for row in grid)


def enforce_invariants(fn: Callable[..., Grid]) -> Callable[..., Grid]:
    """Decorator to ensure grids are valid after every op."""

    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        assert isinstance(out, list), f"{fn.__name__}: Grid must be list"
        if out:
            width = len(out[0])
            for row in out:
                assert len(row) == width, f"{fn.__name__}: inconsistent row width"
                for value in row:
                    assert isinstance(value, int), f"{fn.__name__}: non-int cell"
                    assert 0 <= value <= 9, f"{fn.__name__}: invalid color {value}"
        return out

    return wrapper


@enforce_invariants
def rotate(grid: Grid, angle: int) -> Grid:
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
    height, width = dims(grid)
    out = make_grid(height + top + bottom, width + left + right, value)
    for r in range(height):
        for c in range(width):
            out[r + top][c + left] = grid[r][c]
    return out


@enforce_invariants
def crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    height, width = dims(grid)
    r0, r1 = clamp(r0, 0, height), clamp(r1, 0, height)
    c0, c1 = clamp(c0, 0, width), clamp(c1, 0, width)
    if r1 <= r0 or c1 <= c0:
        return []
    return [row[c0:c1] for row in grid[r0:r1]]


@enforce_invariants
def overlay(base: Grid, top: Grid, mode: str = "top_nonzero_over") -> Grid:
    hb, wb = dims(base)
    ht, wt = dims(top)
    if (hb, wb) != (ht, wt):
        return deepcopy_grid(base)
    out = deepcopy_grid(base)
    for r in range(hb):
        for c in range(wb):
            value = top[r][c]
            if mode == "top_nonzero_over" and value != 0:
                out[r][c] = value
    return out


@enforce_invariants
def map_color(grid: Grid, color_map: Dict[Color, Color]) -> Grid:
    height, width = dims(grid)
    return [
        [color_map.get(grid[r][c], grid[r][c]) for c in range(width)]
        for r in range(height)
    ]


@enforce_invariants
def tile_to_target(grid: Grid, target_h: int, target_w: int) -> Grid:
    height, width = dims(grid)
    if height == 0 or width == 0 or target_h <= 0 or target_w <= 0:
        return []
    out = make_grid(target_h, target_w)
    for r in range(target_h):
        row = grid[r % height]
        for c in range(target_w):
            out[r][c] = row[c % width]
    return out


@enforce_invariants
def repeat_scale(grid: Grid, k: int) -> Grid:
    if k <= 1:
        return deepcopy_grid(grid)
    height, width = dims(grid)
    out = make_grid(height * k, width * k)
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            rs, cs = r * k, c * k
            for dr in range(k):
                for dc in range(k):
                    out[rs + dr][cs + dc] = value
    return out


@enforce_invariants
def fill_holes(grid: Grid, fill_color: int = 0) -> Grid:
    height, width = dims(grid)
    out = deepcopy_grid(grid)
    visited = [[False] * width for _ in range(height)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for r in range(height):
        for c in range(width):
            if out[r][c] != 0 or visited[r][c]:
                continue
            region: List[Tuple[int, int]] = []
            is_border = False
            queue: Deque[Tuple[int, int]] = deque([(r, c)])
            visited[r][c] = True
            while queue:
                pr, pc = queue.popleft()
                region.append((pr, pc))
                if pr in (0, height - 1) or pc in (0, width - 1):
                    is_border = True
                for dr, dc in directions:
                    nr, nc = pr + dr, pc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not visited[nr][nc]
                        and out[nr][nc] == 0
                    ):
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            if not is_border:
                for pr, pc in region:
                    out[pr][pc] = fill_color
    return out


@enforce_invariants
def mirror_symmetry(grid: Grid, axis: str) -> Grid:
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


__all__ = [
    "clamp",
    "crop",
    "deepcopy_grid",
    "dims",
    "enforce_invariants",
    "eq_grid",
    "fill_holes",
    "flip",
    "grid_to_template",
    "make_grid",
    "map_color",
    "mirror_symmetry",
    "overlay",
    "pad",
    "repeat_scale",
    "tile_to_target",
    "valid_grid",
]
