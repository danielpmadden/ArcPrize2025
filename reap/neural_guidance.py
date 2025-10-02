"""Lightweight neural guidance hooks for REAP.

The solver remains symbolic-first but exposes a small interface to optionally
combine neural heuristics with the handcrafted scoring pipeline. The default
implementation simply returns a neutral score, ensuring that existing behaviour
is unchanged when no external model is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .program import Program
from .types import Grid


@dataclass
class NeuralGuidance:
    """Optional neural scorer that can influence beam search.

    Parameters
    ----------
    model:
        An optional callable taking ``(task, program, input_grid, output_grid)``
        and returning a float in ``[0, 1]``. When ``None`` a neutral score of
        ``0.5`` is used, which keeps the symbolic search unbiased.
    """

    model: Any | None = None

    def score(
        self,
        task: Any,
        program: Program,
        input_grid: Grid,
        output_grid: Grid,
    ) -> float:
        """Return a heuristic score in ``[0, 1]``.

        The default behaviour is intentionally conservative: when no model is
        provided (or the callable raises an exception) a neutral value of ``0.5``
        is returned. This keeps the neural hook side-effect free while allowing
        downstream integrations to plug in richer heuristics later on.
        """

        if self.model is None:
            return 0.5
        try:
            raw = self.model(task, program, input_grid, output_grid)
        except Exception:
            return 0.5
        if raw is None:
            return 0.5
        return float(max(0.0, min(1.0, raw)))


__all__ = ["NeuralGuidance"]
