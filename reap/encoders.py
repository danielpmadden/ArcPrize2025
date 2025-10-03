"""reap.encoders
=================

Textual encoding helpers for ARC grids. These utilities were originally embedded
in the monolithic REAP script; extracting them into a standalone module keeps the
core solver agnostic to the presentation layer. The encoders are intentionally
lightweight so we can swap or extend them without touching search logic.
"""

from __future__ import annotations

from typing import Protocol

from .types import Grid


class GridEncoder(Protocol):
    """Interface for components capable of converting grids to and from text.

    Implementations should be stateless; callers are free to reuse instances
    across requests. All encoders must accept any valid ARC grid and either
    return a string representation or reconstruct the original grid from text.
    """

    def to_text(self, grid: Grid) -> str:
        """Serialise ``grid`` into a human-readable text snippet."""

    def to_grid(self, text: str) -> Grid:
        """Parse ``text`` back into a grid."""


class MinimalGridEncoder:
    """Plain digit-based encoder mirroring the original defaults.

    Each row becomes a concatenated string of digits; rows are joined by newlines.
    The format is intentionally simple so that it is easy to eyeball when
    debugging logs.
    """

    def to_text(self, grid: Grid) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in grid)

    def to_grid(self, text: str) -> Grid:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return [[int(char) for char in line] for line in lines]


class GridWithSeparationEncoder:
    """Encoder that inserts a custom separator between cells.

    Parameters
    ----------
    split_symbol:
        Token inserted between neighbouring cells. Defaults to ``"|"`` which is
        visually clear even for grayscale grids.
    """

    def __init__(self, split_symbol: str = "|") -> None:
        self.split_symbol = split_symbol

    def to_text(self, grid: Grid) -> str:
        return "\n".join(self.split_symbol.join(str(cell) for cell in row) for row in grid)

    def to_grid(self, text: str) -> Grid:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return [[int(part) for part in line.split(self.split_symbol)] for line in lines]


class GridCodeBlockEncoder:
    """Encoder that wraps another encoder in Markdown-style code fences."""

    def __init__(self, base_encoder: GridEncoder) -> None:
        self._encoder = base_encoder

    def to_text(self, grid: Grid) -> str:
        return f"```grid\n{self._encoder.to_text(grid)}\n```"

    def to_grid(self, text: str) -> Grid:
        start_token = "```grid\n"
        end_token = "\n```"
        if start_token not in text or end_token not in text:
            raise ValueError("Input text does not contain a fenced grid block")
        core = text.split(start_token, 1)[1].split(end_token, 1)[0]
        return self._encoder.to_grid(core)


DEFAULT_ENCODER: GridEncoder = MinimalGridEncoder()
"""Default encoder shared by modules that need a quick text representation."""


def test_encoder(encoder: GridEncoder) -> None:
    """Round-trip ``encoder`` using a simple identity grid for sanity checking.

    Parameters
    ----------
    encoder:
        Implementation to exercise. The helper prints the encoded text and a
        boolean indicating whether decoding returns the original grid.

    Notes
    -----
    The function exists purely for manual testing. It deliberately prints
    instead of returning data to match the ergonomics of the earlier script.
    """

    sample = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    text = encoder.to_text(sample)
    recovered = encoder.to_grid(text)
    print("Encoded:\n", text)
    print("Roundtrip OK:", recovered == sample)


__all__ = [
    "GridEncoder",
    "MinimalGridEncoder",
    "GridWithSeparationEncoder",
    "GridCodeBlockEncoder",
    "DEFAULT_ENCODER",
    "test_encoder",
]
