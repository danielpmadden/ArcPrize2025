"""reap.search
================

Beam/BFS style program enumeration. The goal is to keep the solver's strategic
logic in one place so the CLI and DSL stay lean.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .color_maps import infer_color_maps_from_train
from .grid_utils import dims, eq_grid
from .objects import extract_objects, learn_object_transformation_maps
from .operations import FUNCTION_REGISTRY
from .program import Op, Program
from .types import Grid


@dataclass
class SearchConfig:
    """Configuration knobs for the synthesiser."""

    max_depth: int = 7
    beam_size: int = 256
    time_budget_s: float = 8.0
    allow_ops: List[str] = field(default_factory=list)


@dataclass
class SearchStats:
    """Instrumentation for analysing search runs."""

    time_elapsed: float = 0.0
    depth_reached: int = 0
    total_candidates: int = 0
    kept_after_filter: int = 0
    valids_found: int = 0
    survivors_per_depth: List[int] = field(default_factory=list)
    best_firstpair_distance: Optional[int] = None
    best_dims_match: Optional[bool] = None
    crashes_during_apply: int = 0


cache_transform_maps: Dict[str, Dict[Any, Dict[str, Any]]] = {}


def program_fits_all(program: Program, train: List[Any]) -> bool:
    """Check whether ``program`` matches every training pair exactly."""

    return all(eq_grid(program.apply(pair.input), pair.output) for pair in train)


def enumerate_params_for_op(op_name: str, task: Any) -> List[Dict[str, Any]]:
    """Generate candidate parameter dictionaries for ``op_name``."""

    params: List[Dict[str, Any]] = []
    height_in, width_in = dims(task.train[0].input)
    height_out, width_out = dims(task.train[0].output)

    if op_name == "rotate":
        params = [{"angle": angle} for angle in (90, 180, 270)]
    elif op_name == "flip":
        params = [{"axis": axis} for axis in ("h", "v")]
    elif op_name == "map_color":
        for cmap in infer_color_maps_from_train(task)[:3]:
            params.append({"color_map": cmap})
    elif op_name == "transform_by_object_template":
        key_hash = hashlib.md5(json.dumps([pair.input for pair in task.train]).encode()).hexdigest()
        template_map = cache_transform_maps.get(key_hash)
        if not template_map:
            cache_transform_maps[key_hash] = learn_object_transformation_maps(task)
            template_map = cache_transform_maps[key_hash]
        if template_map:
            params = [{"transform_map": template_map}]
    elif op_name == "translate_all_by_centroid":
        params = [{"target_r": height_in // 2, "target_c": width_in // 2}]
    elif op_name == "tile_to_target":
        params = [{"target_h": height_out, "target_w": width_out}]
    elif op_name == "repeat_scale":
        factors = {2}
        if height_in and width_in and height_out % height_in == 0 and width_out % width_in == 0:
            scale = height_out // height_in
            if scale == width_out // width_in:
                factors.add(scale)
        params = [{"k": factor} for factor in sorted(factors) if factor > 0]
    elif op_name == "pad":
        params = [
            {"top": 1, "bottom": 0, "left": 0, "right": 0, "value": 0},
            {"top": 0, "bottom": 1, "left": 0, "right": 0, "value": 0},
            {"top": 0, "bottom": 0, "left": 1, "right": 0, "value": 0},
            {"top": 0, "bottom": 0, "left": 0, "right": 1, "value": 0},
        ]
    elif op_name == "crop":
        params = []
        if height_in > 2:
            params.append({"r0": 1, "c0": 0, "r1": height_in, "c1": width_in})
            params.append({"r0": 0, "c0": 0, "r1": height_in - 1, "c1": width_in})
        if width_in > 2:
            params.append({"r0": 0, "c0": 1, "r1": height_in, "c1": width_in})
            params.append({"r0": 0, "c0": 0, "r1": height_in, "c1": width_in - 1})
        if height_in > 2 and width_in > 2:
            params.append({"r0": 1, "c0": 1, "r1": height_in - 1, "c1": width_in - 1})
    elif op_name == "fill_holes":
        colours = {value for pair in task.train for row in pair.output for value in row}
        trial = [0] + sorted([colour for colour in colours if colour != 0])[:2]
        params = [{"fill_color": colour} for colour in trial]
    elif op_name == "mirror_symmetry":
        params = [{"axis": "h"}, {"axis": "v"}]
    elif op_name == "place_in_corner":
        params = [{"which": where} for where in ("tl", "tr", "bl", "br")]
    elif op_name == "remove_color":
        colours = {value for pair in task.train for row in pair.input for value in row}
        params = [{"color": colour} for colour in colours if colour != 0]
    elif op_name in {"keep_largest_object", "outline_object"}:
        params = [{}]
    elif op_name == "fill_object":
        colours = {value for pair in task.train for row in pair.output for value in row if value != 0}
        params = [{"color": colour} for colour in colours]
    return params


def hamming_like_distance(left: Grid, right: Grid) -> int:
    """Compare grids via cell mismatches with a heavy mismatch penalty."""

    if dims(left) != dims(right):
        h_left, w_left = dims(left)
        h_right, w_right = dims(right)
        return h_left * w_left + h_right * w_right
    height, width = dims(left)
    distance = 0
    for r in range(height):
        for c in range(width):
            if left[r][c] != right[r][c]:
                distance += 1
    return distance


def bfs_synthesize(task: Any, cfg: SearchConfig) -> Tuple[List[Program], SearchStats]:
    """Enumerate candidate programs under ``cfg`` with simple beam pruning."""

    start = time.time()
    stats = SearchStats()
    first_in = task.train[0].input
    first_out = task.train[0].output
    target_dims = dims(first_out)

    def firstpair_score(program: Program) -> Tuple[int, bool, bool]:
        output = program.apply(first_in)
        distance = hamming_like_distance(output, first_out)
        dims_match = dims(output) == target_dims
        try:
            obj_count_match = len(extract_objects(output)) == len(extract_objects(first_out))
        except Exception:
            obj_count_match = False
        return distance, dims_match, obj_count_match

    def fits_first(program: Program) -> bool:
        distance, dims_match, _ = firstpair_score(program)
        if stats.best_firstpair_distance is None or distance < stats.best_firstpair_distance:
            stats.best_firstpair_distance = distance
            stats.best_dims_match = dims_match
        last_op = program.ops[-1].name if program.ops else None
        resize_ops = {
            "compress_block",
            "expand_block",
            "tile_with_padding",
            "pad",
            "crop",
            "tile_to_target",
            "repeat_scale",
        }
        if not dims_match and last_op not in resize_ops and distance > 3:
            return False
        return dims_match or distance <= 5

    beam: List[Program] = [Program([])]
    valids: List[Program] = []

    if program_fits_all(Program([]), task.train):
        valids.append(Program([]))

    for depth in range(cfg.max_depth):
        if time.time() - start > cfg.time_budget_s:
            break
        next_beam: List[Program] = []
        survivors_this_depth = 0

        for program in beam:
            for op_name in cfg.allow_ops:
                for params in enumerate_params_for_op(op_name, task):
                    candidate = Program(program.ops + [Op(op_name, params)])
                    stats.total_candidates += 1
                    try:
                        if not fits_first(candidate):
                            continue
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    survivors_this_depth += 1
                    if program_fits_all(candidate, task.train):
                        valids.append(candidate)
                    next_beam.append(candidate)

        hard_cap = max(cfg.beam_size * 5, cfg.beam_size)
        if len(next_beam) > hard_cap:
            def rank_key(program: Program) -> Tuple[float, int]:
                distance, *_ = firstpair_score(program)
                return program.cost(), distance

            next_beam = sorted(next_beam, key=rank_key)[:hard_cap]

        def final_rank_key(program: Program) -> Tuple[float, int]:
            distance, *_ = firstpair_score(program)
            return program.cost(), distance

        next_beam.sort(key=final_rank_key)
        stats.kept_after_filter += len(next_beam)
        beam = next_beam[:cfg.beam_size]
        stats.survivors_per_depth.append(len(beam))
        stats.depth_reached = depth + 1

        if time.time() - start > cfg.time_budget_s:
            break

    stats.time_elapsed = time.time() - start
    stats.valids_found = len(valids)
    unique = {program.signature(): program for program in sorted(valids, key=lambda p: p.cost())}
    return list(unique.values()), stats


def pick_two(programs: List[Program]) -> Tuple[Optional[Program], Optional[Program]]:
    """Return the best two programs for downstream use."""

    if not programs:
        return None, None
    return programs[0], (programs[1] if len(programs) > 1 else None)


__all__ = [
    "SearchConfig",
    "SearchStats",
    "bfs_synthesize",
    "enumerate_params_for_op",
    "program_fits_all",
    "pick_two",
    "hamming_like_distance",
]
