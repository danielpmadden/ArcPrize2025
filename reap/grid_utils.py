from __future__ import annotations

from typing import List, Any, Tuple

from .types import Grid, Template

# ---------------------------------------------------------------------------
# Basic geometry / construction helpers
# ---------------------------------------------------------------------------
def dims(grid: Grid) -> Tuple[int, int]: 
    """Return the height and width of a grid.

    Parameters
    ----------
    grid:
        Grid whose dimensions should be measured. ``grid`` may be empty or
        ragged; the function guards against those cases.

    Returns
    -------
    tuple[int, int]
        ``(height, width)`` of the grid. Empty grids return ``(0, 0)``.

    Notes
    -----
    The helper gracefully tolerates unexpected inputs to reduce boilerplate in
    callers. This matches the behaviour of the original single-file script.
    """

    if not grid or not isinstance(grid, list):
        return 0, 0
    if not grid or not grid[0]:
        return len(grid), 0
    return len(grid), len(grid[0])


def deepcopy_grid(grid: Grid) -> Grid:
    """Return a structural copy of ``grid``.

    Parameters
    ----------
    grid:
        Grid to duplicate.

    Returns
    -------
    Grid
        Fresh list-of-lists copy safe for mutation by callers.

    Notes
    -----
    Python's ``copy.deepcopy`` would work but is slower and harder to reason
    about for small grids. An explicit list comprehension is transparent and
    predictable.
    """

    return [row[:] for row in grid]


def eq_grid(left: Grid, right: Grid) -> bool:
    """Check structural equality between two grids."""

    return left == right


def valid_grid(grid: Grid) -> bool:
    """Validate that ``grid`` matches ARC's conventions."""

    if not isinstance(grid, list) or not grid:
        return True
    if not all(isinstance(row, list) for row in grid):
        return False
    if not grid[0]:
        return True
    width = len(grid[0])
    return all(len(row) == width and all(isinstance(val, int) and 0 <= val <= 9 for val in row) for row in grid)


def make_grid(height: int, width: int, fill: int = 0) -> Grid:
    """Construct a grid filled with a single colour."""

    if height <= 0 or width <= 0:
        return []
    return [[fill for _ in range(width)] for _ in range(height)]


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp ``value`` into the inclusive range ``[minimum, maximum]``."""

    return max(minimum, min(value, maximum))


def grid_to_template(grid: Grid) -> List[List[int]]:
    """Return a simplified template form of a grid.

    Each cell is normalized to an int; non-integers are cast to 0.
    This is used for behaviour signatures, not actual solving.
    """
    if not grid:
        return [[0]]
    try:
        return [[int(cell) if isinstance(cell, (int, bool)) else 0 for cell in row] for row in grid]
    except Exception:
        # fallback safe template
        h = len(grid)
        w = len(grid[0]) if grid else 0
        return [[0] * w for _ in range(h)]



__all__ = [
    "dims",
    "deepcopy_grid",
    "eq_grid",
    "valid_grid",
    "make_grid",
    "clamp",
    "grid_to_template",
]
