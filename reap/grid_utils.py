"""reap.grid_utils
====================

Low-level helpers for manipulating ARC grids. The original project interleaved
these utilities with the DSL and search code, which made it hard to reason about
side effects. Splitting them into their own module lets every other component
(such as the solver, search strategies, or CLI) depend on a consistent,
well-documented toolbox.

Every function aims to be side-effect free and defensive: we validate shapes,
provide clear documentation about edge cases, and strive to make potential
failure modes explicit via docstrings and inline comments.
"""

from __future__ import annotations

from typing import Tuple

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


def grid_to_template(grid: Grid) -> Template:
    """Convert a grid to a hashable template (tuple of tuples)."""

    return tuple(tuple(row) for row in grid)


__all__ = [
    "dims",
    "deepcopy_grid",
    "eq_grid",
    "valid_grid",
    "make_grid",
    "clamp",
    "grid_to_template",
]
