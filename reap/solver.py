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
from .rule_engine import compile_rules, prioritise_rules, suggest_rules
from .search import SearchConfig, SearchStats, bfs_synthesize, pick_two
from .task_classifier import TaskProfile, classify_task as classify_task_profile
from .types import Grid
from .operations import FUNCTION_REGISTRY


MACRO_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _normalise_params(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[Any, Any] = {}
        for key, val in value.items():
            new_key: Any = key
            if isinstance(key, str):
                if key.lstrip("-").isdigit():
                    new_key = int(key)
            result[new_key] = _normalise_params(val)
        return result
    if isinstance(value, list):
        return [_normalise_params(item) for item in value]
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def _program_from_ops(ops: List[Dict[str, Any]]) -> Program:
    return Program([Op(spec["name"], _normalise_params(spec.get("params", {}))) for spec in ops])


def _register_macro(name: str, ops: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    if name in MACRO_REGISTRY:
        return
    program = _program_from_ops(ops)

    def _macro(grid: Grid, *, _program: Program = program) -> Grid:
        return _program.apply(grid)

    FUNCTION_REGISTRY[name] = _macro
    MACRO_REGISTRY[name] = {"program": program, "ops": ops, "meta": meta}


def register_library_macros(items: Any | None) -> List[str]:
    """Inject persisted macros into the function registry."""

    registered: List[str] = []
    if not items:
        return registered
    for item in items:
        identifier = getattr(item, "identifier", None) or item.get("id")
        ops = getattr(item, "ops", None) or item.get("ops", [])
        meta = getattr(item, "meta", None) or item.get("meta", {})
        macro_name = f"lib::{identifier}"
        _register_macro(macro_name, ops, meta)
        registered.append(macro_name)
    return registered


def register_rule_macros(rules: List[Program], prefix: str) -> List[str]:
    names: List[str] = []
    for idx, program in enumerate(rules):
        macro_name = f"rule::{prefix}_{idx}"
        if macro_name in MACRO_REGISTRY:
            continue
        ops = [op.__dict__ for op in program.ops]
        meta = {"kind": "rule", "priority": idx, "ops": ops}
        _register_macro(macro_name, ops, meta)
        names.append(macro_name)
    return names


def unregister_macros(names: List[str]) -> None:
    for name in names:
        MACRO_REGISTRY.pop(name, None)
        FUNCTION_REGISTRY.pop(name, None)


def _base_task_features(task: Any) -> Dict[str, Any]:
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


def classify_task(task: Any) -> Dict[str, Any]:
    """Compute coarse heuristics and family biases for ``task``."""

    base = _base_task_features(task)
    profile: TaskProfile = classify_task_profile(getattr(task, "train", []))
    base.update(profile.feature_flags)
    base["family_prioritised_ops"] = profile.prioritised_ops
    base["families"] = profile.families
    return base


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


def solve_task(
    task: Any,
    time_budget_s: float = 10.0,
    *,
    cfg: SearchConfig | None = None,
    library_items: Any | None = None,
    enable_library: bool = True,
    enable_explore_bias: bool = True,
    enable_revisions: bool = True,
    enable_pooled_revisions: bool = True,
    enable_equivalence: bool = True,
    enable_task_family_bias: bool = True,
    enable_feedback_diff: bool = False,
    rule_confidence: Dict[str, Dict[str, float]] | None = None,
    task_family_stats: Dict[str, Any] | None = None,
    meta_bias_strength: float = 0.0,
    dedup_mode: str = "both",
    neural_guidance: Any | None = None,
    neural_bias: float = 0.0,
    exploration_temp: float | None = None,
    min_diverse: int | None = None,
    library_priority: float | None = None,
) -> Tuple[List[Dict[str, Grid]], SearchStats, List[Program], Dict[str, Dict[str, float]]]:
    """Solve ``task`` by selecting operations based on coarse heuristics."""

    features = classify_task(task)
    if enable_task_family_bias and features.get("families"):
        print(f"[TASK] families detected: {features['families']}")
    ops = [
        "rotate",
        "flip",
        "map_color",
        "fill_holes",
        "mirror_symmetry",
        "complete_symmetry",
        "project_profile",
        "keep_largest_object",
        "remove_color",
        "copy_dominant_object",
        "align_to_edge",
        "colorize_by_size",
        "repeat_objects_horiz",
        "distribute_evenly",
        "snap_to_grid",
        "remove_small_objects",
        "keep_objects_with_color",
        "flood_fill_from",
    ]
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

    registered_macros: List[str] = []
    if enable_library:
        registered_macros.extend(register_library_macros(library_items))

    rules = suggest_rules(getattr(task, "train", []))
    rule_conf_map = rule_confidence or {}
    prioritised = prioritise_rules(rules, rule_conf_map, temperature=0.4)
    compiled_rules = compile_rules(prioritised)
    rule_prefix = "tmp"
    rule_macros = register_rule_macros(compiled_rules, rule_prefix)

    allow_ops = ops + rule_macros
    if enable_library:
        allow_ops += registered_macros
    allow_ops = list(dict.fromkeys(allow_ops))
    if enable_task_family_bias:
        prioritised_ops = features.get("family_prioritised_ops", [])
        if prioritised_ops:
            reordered = prioritised_ops + [op for op in allow_ops if op not in prioritised_ops]
            allow_ops = reordered

    search_cfg = cfg or SearchConfig()
    search_cfg = SearchConfig(**{**search_cfg.__dict__}) if cfg else search_cfg
    search_cfg.time_budget_s = time_budget_s
    search_cfg.allow_ops = allow_ops
    search_cfg.task_features = features
    search_cfg.explore_bias = enable_explore_bias
    search_cfg.enable_revisions = enable_revisions
    search_cfg.use_pooled_revisions = enable_pooled_revisions
    search_cfg.deduplicate_equivalence = enable_equivalence
    search_cfg.dedup_mode = dedup_mode
    search_cfg.feedback_diff = enable_feedback_diff
    if exploration_temp is not None:
        search_cfg.exploration_temp = exploration_temp
    if min_diverse is not None:
        search_cfg.min_diverse = min_diverse
    if library_priority is not None:
        search_cfg.library_priority = library_priority
    search_cfg.neural_guidance = neural_guidance
    search_cfg.neural_bias_weight = neural_bias
    if rule_confidence is not None:
        search_cfg.rule_confidence = rule_confidence
    if enable_task_family_bias:
        prioritised_ops = features.get("family_prioritised_ops", [])
        search_cfg.task_features.setdefault("family_prioritised_ops", prioritised_ops)
    if meta_bias_strength > 0.0:
        search_cfg.meta_bias_strength = meta_bias_strength
    search_cfg.__post_init__()
    preferences: Dict[str, float] = {}
    if task_family_stats and features.get("families"):
        for family in features["families"]:
            entry = task_family_stats.get(family)
            if not isinstance(entry, dict):
                continue
            ops_stats = entry.get("ops", {})
            macros_stats = entry.get("macros", {})
            confidence = float(entry.get("confidence", 0.0))
            for op_name, count in ops_stats.items():
                preferences[op_name] = preferences.get(op_name, 0.0) + float(count)
            for op_name, count in macros_stats.items():
                preferences[op_name] = preferences.get(op_name, 0.0) + 1.2 * float(count)
            if confidence:
                preferences[family] = preferences.get(family, 0.0) + confidence
    if preferences:
        max_val = max(preferences.values())
        if max_val > 0:
            for key in list(preferences):
                preferences[key] = preferences[key] / max_val
        if meta_bias_strength > 0.0:
            print(f"[TASK] meta bias preferences: {preferences}")
    search_cfg.family_preference_scores = preferences

    for name in registered_macros + rule_macros:
        if name in MACRO_REGISTRY:
            search_cfg.macro_metadata[name] = MACRO_REGISTRY[name]

    if features.get("obj_count_same"):
        search_cfg.max_depth = max(search_cfg.max_depth, 5)
        search_cfg.beam_size = max(search_cfg.beam_size, 128)
    if not features["same_shape"]:
        search_cfg.max_depth = max(search_cfg.max_depth, 6)
        search_cfg.beam_size = max(search_cfg.beam_size, 128)
    train_pairs = len(getattr(task, "train", []))
    if train_pairs >= 4:
        search_cfg.beam_size = max(search_cfg.beam_size, 320)
    if train_pairs >= 6:
        search_cfg.beam_size = max(search_cfg.beam_size, 384)
    search_cfg.min_diverse = max(1, min(search_cfg.min_diverse, search_cfg.beam_size))

    try:
        programs, stats = bfs_synthesize(task, search_cfg)
    except Exception as exc:  # pragma: no cover - defensive logging path
        print(f"[WARN] bfs_synthesize crashed: {exc}")
        programs, stats = [], SearchStats()

    unregister_macros(rule_macros)

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

    if cfg is not None:
        cfg.task_features = search_cfg.task_features
        cfg.family_preference_scores = search_cfg.family_preference_scores
        cfg.meta_bias_strength = search_cfg.meta_bias_strength
        cfg.dedup_mode = search_cfg.dedup_mode
        cfg.neural_bias_weight = search_cfg.neural_bias_weight
        cfg.min_diverse = search_cfg.min_diverse
        cfg.library_priority = search_cfg.library_priority
        cfg.exploration_temp = search_cfg.exploration_temp

    return attempts, stats, programs, search_cfg.rule_confidence


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
