"""reap.solver
================

High-level orchestration that stitches together feature classification and the
search engine. Functions in this module are the primary public API used by the
CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .dsl import flip, rotate
from .grid_utils import dims, eq_grid
from .objects import extract_objects
from .program import Op, Program
from .search import SearchConfig, SearchStats, bfs_synthesize, pick_two
from .types import Grid


def classify_task(task: Any) -> Dict[str, Any]:
    """Compute coarse heuristics about a task's training pairs."""

    features: Dict[str, Any] = {}
    features["same_shape"] = all(dims(pair.input) == dims(pair.output) for pair in task.train)
    features["sym_h"] = all(eq_grid(pair.input, flip(pair.input, "h")) for pair in task.train)
    features["sym_v"] = all(eq_grid(pair.input, flip(pair.input, "v")) for pair in task.train)
    try:
        inputs = [extract_objects(pair.input) for pair in task.train]
        outputs = [extract_objects(pair.output) for pair in task.train]
        features["obj_count_same"] = all(len(i) == len(o) for i, o in zip(inputs, outputs))
    except Exception:
        features["obj_count_same"] = False
    return features


def micro_tune(output: Grid, target_shape: Tuple[int, int]) -> Grid:
    """Attempt small rotations/flips to make ``output`` match ``target_shape``."""

    if dims(output) == target_shape:
        return output
    for fn in (
        lambda g: rotate(g, 90),
        lambda g: rotate(g, 180),
        lambda g: rotate(g, 270),
        lambda g: flip(g, "h"),
        lambda g: flip(g, "v"),
    ):
        candidate = fn(output)
        if dims(candidate) == target_shape:
            return candidate
    return output


def solve_task(task: Any, time_budget_s: float = 10.0) -> Tuple[List[Dict[str, Grid]], SearchStats]:
    """Solve ``task`` by selecting operations based on coarse heuristics."""

    features = classify_task(task)
    ops = ["rotate", "flip", "map_color", "fill_holes", "mirror_symmetry"]
    if features.get("obj_count_same"):
        ops += [
            "transform_by_object_template",
            "translate_object",
            "place_in_center",
            "place_in_corner",
            "keep_largest_object",
            "outline_object",
            "fill_object",
        ]
    if not features["same_shape"]:
        ops += [
            "pad",
            "crop",
            "tile_to_target",
            "repeat_scale",
            "compress_block",
            "expand_block",
            "tile_with_padding",
        ]
    else:
        ops += ["translate_all_by_centroid"]
    ops.append("remove_color")
    ops += ["repeat_pattern", "replace_region", "grow_block"]
    ops = list(dict.fromkeys(ops))

    cfg = SearchConfig()
    cfg.time_budget_s = time_budget_s
    cfg.allow_ops = ops
    cfg.max_depth = 4
    cfg.beam_size = 64
    if features.get("obj_count_same"):
        cfg.max_depth = max(cfg.max_depth, 5)
        cfg.beam_size = max(cfg.beam_size, 128)
    if not features["same_shape"]:
        cfg.max_depth = max(cfg.max_depth, 6)
        cfg.beam_size = max(cfg.beam_size, 128)

    try:
        programs, stats = bfs_synthesize(task, cfg)
    except Exception as exc:  # pragma: no cover - defensive logging path
        print(f"[WARN] bfs_synthesize crashed: {exc}")
        programs, stats = [], SearchStats()

    program_a, program_b = pick_two(programs)
    if not program_a:
        out_dims = dims(task.train[0].output) if task.train else (0, 0)
        program_a = ProgramFallbacks.tile_to_target(out_dims)
        program_b = ProgramFallbacks.rotate_180()
    elif not program_b:
        program_b = ProgramFallbacks.rotate_180()

    attempts: List[Dict[str, Grid]] = []
    for test_item in task.test:
        target_dims = dims(task.train[0].output) if task.train else dims(test_item.input)
        out_a = program_a.apply(test_item.input)
        out_a = micro_tune(out_a, target_dims)
        out_b = program_b.apply(test_item.input)
        out_b = micro_tune(out_b, target_dims)
        attempts.append({"attempt_1": out_a, "attempt_2": out_b})

    return attempts, stats


class ProgramFallbacks:
    """Factory helpers for fallback programs used when search fails."""

    @staticmethod
    def tile_to_target(target_dims: Tuple[int, int]) -> Program:
        height, width = target_dims
        return Program([Op("tile_to_target", {"target_h": height, "target_w": width})])

    @staticmethod
    def rotate_180() -> Program:
        return Program([Op("rotate", {"angle": 180})])


__all__ = ["solve_task", "classify_task", "micro_tune", "ProgramFallbacks"]
