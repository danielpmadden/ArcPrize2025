"""Grid encoder implementations for serialising ARC grids."""
from __future__ import annotations

from typing import Protocol

from .types import Grid


class GridEncoder(Protocol):
    """Protocol for converting grids to and from textual formats."""

    def to_text(self, grid: Grid) -> str:
        ...

    def to_grid(self, text: str) -> Grid:
        ...


class MinimalGridEncoder:
    """Encodes grids as plain text digits with no separators."""

    def to_text(self, grid: Grid) -> str:
        return "\n".join("".join(str(x) for x in row) for row in grid)

    def to_grid(self, text: str) -> Grid:
        return [[int(x) for x in line] for line in text.strip().splitlines()]


class GridWithSeparationEncoder:
    """Encodes grids with a separator (e.g. '|') between cells."""

    def __init__(self, split_symbol: str = "|") -> None:
        self.split_symbol = split_symbol

    def to_text(self, grid: Grid) -> str:
        return "\n".join(self.split_symbol.join(str(x) for x in row) for row in grid)

    def to_grid(self, text: str) -> Grid:
        return [
            [int(x) for x in line.split(self.split_symbol)]
            for line in text.strip().splitlines()
        ]


class GridCodeBlockEncoder:
    """Wraps another encoder and puts the grid inside a code block."""

    def __init__(self, base_encoder: GridEncoder) -> None:
        self.encoder = base_encoder

    def to_text(self, grid: Grid) -> str:
        return f"```grid\n{self.encoder.to_text(grid)}\n```"

    def to_grid(self, text: str) -> Grid:
        grid_text = text.split("```grid\n")[1].split("\n```")[0]
        return self.encoder.to_grid(grid_text)


DEFAULT_ENCODER: GridEncoder = MinimalGridEncoder()

__all__ = [
    "DEFAULT_ENCODER",
    "GridCodeBlockEncoder",
    "GridEncoder",
    "GridWithSeparationEncoder",
    "MinimalGridEncoder",
]
