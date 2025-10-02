"""reap.search
================

Enhanced beam-search program synthesis with macro support, richer scoring, and
revision mechanisms. The implementation remains intentionally lightweight but
now tracks program behaviour signatures, macro usage, and exploration-biased
selection to better match state-of-the-art ARC solvers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .color_maps import infer_color_maps_from_train
from .grid_utils import dims, eq_grid, grid_to_template
from .objects import extract_objects, learn_object_transformation_maps
from .program import Op, Program
from .rule_engine import update_rule_confidence as update_rule_confidence_stats
from .types import Example, Grid

if TYPE_CHECKING:  # pragma: no cover - type hinting only
    from .neural_guidance import NeuralGuidance


@dataclass
class SearchConfig:
    """Configuration knobs for the synthesiser."""

    max_depth: int = 7
    beam_size: int = 256
    time_budget_s: float = 8.0
    allow_ops: List[str] = field(default_factory=list)
    elite: int = 16
    softmax_T: float = 0.35
    exploration_softmax_temp: float = 0.35
    exploration_temp: float = 1.0
    alpha_primary: float = 1.0
    beta_secondary: float = 1.0
    explore_bias: bool = True
    enable_revisions: bool = True
    parents_per_revision: int = 8
    max_individual_revisions: int = 8
    max_pooled_revisions: int = 4
    use_pooled_revisions: bool = True
    deduplicate_equivalence: bool = True
    dedup_mode: str = "both"
    feedback_diff: bool = False
    macros: Dict[str, Program] = field(default_factory=dict)
    macro_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rule_confidence_enabled: bool = True
    rule_confidence: Dict[str, Dict[str, float]] = field(default_factory=dict)
    seed: int = 1337
    task_features: Dict[str, Any] = field(default_factory=dict)
    meta_bias_strength: float = 0.0
    family_preference_scores: Dict[str, float] = field(default_factory=dict)
    min_diverse: int = 5
    library_priority: float = 2.0
    neural_guidance: "NeuralGuidance | None" = None
    neural_bias_weight: float = 0.0

    def __post_init__(self) -> None:
        # Keep backward compatibility with previous ``softmax_T`` accessors.
        if self.softmax_T != self.exploration_softmax_temp:
            # Prefer whichever value deviates from the default to propagate.
            default = 0.35
            if self.softmax_T != default:
                self.exploration_softmax_temp = self.softmax_T
            else:
                self.softmax_T = self.exploration_softmax_temp
        dedup_pref = (self.dedup_mode or "both").lower()
        if dedup_pref not in {"behavior", "structural", "both"}:
            dedup_pref = "both"
        if not getattr(self, "deduplicate_equivalence", True) and dedup_pref != "structural":
            dedup_pref = "structural"
        self.dedup_mode = dedup_pref
        self.deduplicate_equivalence = self.dedup_mode in {"behavior", "both"}

    @property
    def effective_temperature(self) -> float:
        """Return the combined exploration temperature used for sampling."""

        base = self.exploration_softmax_temp
        return max(1e-6, base * max(self.exploration_temp, 1e-6))

    @property
    def softmax_temperature(self) -> float:
        """Alias exposing the active exploration temperature."""

        return self.exploration_softmax_temp


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
    best_primary: int = 0
    best_secondary: float = 0.0
    num_elites_kept: int = 0
    num_sampled_kept: int = 0
    dedup_struct_hits: int = 0
    dedup_behavior_hits: int = 0
    revision_rounds: int = 0
    revisions_generated: int = 0
    macro_uses: Dict[str, int] = field(default_factory=dict)
    diversity_counts: List[int] = field(default_factory=list)
    temperature_trace: List[float] = field(default_factory=list)
    budget_exhausted: bool = False


@dataclass
class Candidate:
    """Container holding evaluated program metadata."""

    program: Program
    primary: int
    secondary: float
    outputs: List[Grid]
    struct_sig: str
    behavior_sig: str
    last_op: Optional[str]
    neural_score: float
    combined_score: float


cache_transform_maps: Dict[str, Dict[Any, Dict[str, Any]]] = {}


def program_struct_signature(program: Program) -> str:
    """Return a hash describing the program's structure (ops + params)."""

    serial = [
        (op.name, tuple(sorted(op.params.items())))
        for op in program.ops
    ]
    blob = json.dumps(serial, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _behaviour_blob(outputs: Sequence[Grid]) -> str:
    templates = [grid_to_template(grid) for grid in outputs]
    blob = json.dumps(templates, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def program_behavior_signature(program: Program, train: List[Example]) -> str:
    """Return a short hash of program behaviour across training inputs."""

    outputs = []
    for example in train:
        try:
            outputs.append(program.apply(example.input))
        except Exception:
            outputs.append([[0]])
    return _behaviour_blob(outputs)


def secondary_cell_accuracy(pred: Grid, true: Grid) -> float:
    """Mean per-cell accuracy between ``pred`` and ``true`` grids."""

    dims_pred = dims(pred)
    dims_true = dims(true)
    if dims_pred != dims_true or dims_true == (0, 0):
        return 0.0
    height, width = dims_true
    mismatches = 0
    for r in range(height):
        for c in range(width):
            if pred[r][c] != true[r][c]:
                mismatches += 1
    return 1.0 - (mismatches / max(1, height * width))


def score_program(program: Program, train: List[Example]) -> Tuple[int, float]:
    """Compute primary and secondary scores for ``program``."""

    _outputs, primary, secondary = evaluate_program(program, train)
    return primary, secondary


def evaluate_program(program: Program, train: List[Example]) -> Tuple[List[Grid], int, float]:
    """Return model outputs, primary and secondary scores."""

    outputs: List[Grid] = []
    matches = 0
    accuracies: List[float] = []
    for example in train:
        out = program.apply(example.input)
        outputs.append(out)
        if example.output is not None and eq_grid(out, example.output):
            matches += 1
        if example.output is not None:
            accuracies.append(secondary_cell_accuracy(out, example.output))
    secondary = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return outputs, matches, secondary


def compute_neural_score(
    cfg: SearchConfig,
    task: Any,
    program: Program,
    train: List[Example],
    outputs: Sequence[Grid],
) -> float:
    """Return the average neural guidance score for ``program``."""

    guidance = cfg.neural_guidance
    if guidance is None:
        return 0.5
    scores: List[float] = []
    for example, predicted in zip(train, outputs):
        try:
            score = guidance.score(task, program, example.input, predicted)
        except Exception:
            score = 0.5
        scores.append(max(0.0, min(1.0, float(score))))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def program_fits_all(program: Program, train: List[Example]) -> bool:
    """Check if ``program`` solves every training pair exactly."""

    if not train:
        return False
    _outputs, primary, _secondary = evaluate_program(program, train)
    return primary == len(train)


def enumerate_params_for_op(
    op_name: str,
    task: Any,
    *,
    features: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate candidate parameter dictionaries for ``op_name``.

    The enumeration leverages coarse task features to prioritise relevant
    parameters and keeps ranges conservative to avoid search explosion.
    """

    features = features or {}
    params: List[Dict[str, Any]] = []
    first_pair = task.train[0] if getattr(task, "train", None) else None
    height_in, width_in = dims(first_pair.input) if first_pair else (0, 0)
    height_out, width_out = dims(first_pair.output) if first_pair else (0, 0)

    if op_name.startswith("lib::") or op_name.startswith("rule::"):
        return [{}]

    if op_name == "rotate":
        params = [{"angle": angle} for angle in (90, 180, 270)]
    elif op_name == "flip":
        params = [{"axis": axis} for axis in ("h", "v")]
    elif op_name == "map_color":
        candidates = infer_color_maps_from_train(task)[:5]
        params = [{"color_map": cmap} for cmap in candidates]
    elif op_name == "transform_by_object_template":
        key_hash = hashlib.md5(json.dumps([pair.input for pair in task.train]).encode()).hexdigest()
        template_map = cache_transform_maps.get(key_hash)
        if not template_map:
            cache_transform_maps[key_hash] = learn_object_transformation_maps(task)
            template_map = cache_transform_maps[key_hash]
        if template_map:
            params = [{"transform_map": template_map}]
    elif op_name == "translate_object":
        offsets = range(-2, 3)
        params = [{"dr": dr, "dc": dc} for dr in offsets for dc in offsets]
    elif op_name == "translate_all_by_centroid":
        params = [{"target_r": height_in // 2, "target_c": width_in // 2}]
    elif op_name == "tile_to_target":
        params = [{"target_h": height_out, "target_w": width_out}]
    elif op_name == "repeat_scale":
        factors = set()
        if height_in and width_in and height_out % max(1, height_in) == 0 and width_out % max(1, width_in) == 0:
            scale_h = height_out // max(1, height_in)
            scale_w = width_out // max(1, width_in)
            if scale_h == scale_w and scale_h > 1:
                factors.add(scale_h)
        factors.update({2, 3})
        params = [{"k": factor} for factor in sorted(factors)]
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
    elif op_name in {"keep_largest_object", "outline_object", "fill_object"}:
        params = [{}]
    elif op_name == "keep_objects_with_color":
        colours = {value for pair in task.train for row in pair.input for value in row if value != 0}
        params = [{"color": colour} for colour in colours]
    elif op_name == "remove_small_objects":
        params = [{"min_size": size} for size in (2, 3, 4)]
    elif op_name == "repeat_objects_horiz":
        params = [{"count": count, "gap": gap} for count in (2, 3) for gap in (0, 1)]
    elif op_name == "align_to_edge":
        params = [{"which": edge} for edge in ("top", "bottom", "left", "right")]
    elif op_name == "copy_dominant_object":
        params = [{"to": where} for where in ("center", "tl", "tr", "bl", "br")]
    elif op_name == "colorize_by_size":
        params = [{"order": order} for order in ("asc", "desc")]
    elif op_name == "distribute_evenly":
        params = [{"axis": axis} for axis in ("h", "v")]
    elif op_name == "snap_to_grid":
        params = [{"tile_h": th, "tile_w": tw} for th in (1, 2, 3) for tw in (1, 2, 3)]
    elif op_name == "project_profile":
        params = [{"axis": axis} for axis in ("h", "v")]
    elif op_name == "flood_fill_from":
        colours = {value for pair in task.train for row in pair.input for value in row if value != 0}
        params = [{"color_src": colour, "color_dst": dst} for colour in colours for dst in colours if dst != colour]
    else:
        params = [{}]

    if features.get("same_shape") and op_name in {"pad", "crop", "tile_to_target", "repeat_scale"}:
        # Down-prioritise resizing when all train pairs share dimensions.
        params = params[:2] if params else [{}]

    return params or [{}]


def softmax_weights(values: Sequence[float], temperature: float) -> List[float]:
    m = max(values)
    denom = 0.0
    exps: List[float] = []
    for value in values:
        exp_val = math.exp((value - m) / max(temperature, 1e-6))
        exps.append(exp_val)
        denom += exp_val
    return [exp_val / denom if denom else 0.0 for exp_val in exps]


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


def diff_grid(pred: Grid, target: Grid) -> Dict[str, Any]:
    """Compute structured difference information between two grids."""

    dims_pred = dims(pred)
    dims_target = dims(target)
    height = min(dims_pred[0], dims_target[0])
    width = min(dims_pred[1], dims_target[1])
    mismatch_coords = []
    mismatch_pred: Dict[int, int] = {}
    mismatch_target: Dict[int, int] = {}
    for r in range(height):
        for c in range(width):
            if pred[r][c] != target[r][c]:
                mismatch_coords.append((r, c))
                mismatch_pred[pred[r][c]] = mismatch_pred.get(pred[r][c], 0) + 1
                mismatch_target[target[r][c]] = mismatch_target.get(target[r][c], 0) + 1
    diff = {
        "dims_pred": dims_pred,
        "dims_target": dims_target,
        "mismatch_coords": mismatch_coords,
        "mismatch_count": len(mismatch_coords),
        "mismatch_pred": mismatch_pred,
        "mismatch_target": mismatch_target,
    }
    diff["ascii"] = generate_ascii_diff(pred, target)
    return diff


def generate_ascii_diff(pred: Grid, target: Grid) -> str:
    """Return a human-readable ASCII diff between ``pred`` and ``target``."""

    height = max(len(pred), len(target))
    width = max(len(pred[0]) if pred else 0, len(target[0]) if target else 0)
    lines: List[str] = []
    for r in range(height):
        row_chars: List[str] = []
        for c in range(width):
            pred_val = pred[r][c] if r < len(pred) and c < len(pred[0]) else None
            target_val = target[r][c] if r < len(target) and c < len(target[0]) else None
            if pred_val == target_val:
                row_chars.append(str(target_val) if target_val is not None else ".")
            elif pred_val is None or target_val is None:
                row_chars.append("*")
            else:
                row_chars.append("X")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def repair_strategies(diff: Dict[str, Any]) -> List[Op]:
    """Return heuristic repair operations derived from ``diff``."""

    ops: List[Op] = []
    dims_pred = diff["dims_pred"]
    dims_target = diff["dims_target"]
    if dims_pred != dims_target and dims_target != (0, 0):
        h, w = dims_target
        ops.append(Op("tile_to_target", {"target_h": h, "target_w": w}))
        if dims_pred[0]:
            ops.append(Op("repeat_scale", {"k": max(2, h // max(1, dims_pred[0]))}))
        ops.append(Op("pad", {"top": 0, "bottom": max(0, h - dims_pred[0]), "left": 0, "right": max(0, w - dims_pred[1]), "value": 0}))
    if diff["mismatch_count"]:
        mismatch_pred = {k: v for k, v in diff["mismatch_pred"].items() if k != 0}
        mismatch_target = {k: v for k, v in diff["mismatch_target"].items() if k != 0}
        if len(mismatch_pred) == 1 and len(mismatch_target) == 1:
            src = next(iter(mismatch_pred))
            dst = next(iter(mismatch_target))
            ops.append(Op("map_color", {"color_map": {src: dst}}))
        extra_pred_colours = [colour for colour in mismatch_pred if colour not in mismatch_target]
        if extra_pred_colours:
            ops.append(Op("remove_color", {"color": extra_pred_colours[0]}))
        if len(diff["mismatch_coords"]) > 4:
            ops.append(Op("mirror_symmetry", {"axis": "h"}))
    return ops[:3]


def revise_program_individual(
    candidate: Candidate,
    diff: Dict[str, Any],
    *,
    ascii_diff: Optional[str] = None,
) -> List[Program]:
    repairs = repair_strategies(diff)
    programs = []
    for op in repairs[:3]:
        programs.append(Program(candidate.program.ops + [op]))
    return programs


def feedback_revision(
    candidate: Candidate,
    diff: Dict[str, Any],
    *,
    ascii_diff: Optional[str] = None,
) -> List[Program]:
    """Derive rule tweaks by analysing ``diff`` and ASCII feedback."""

    if not candidate.program.ops:
        return []
    variants: List[Program] = []
    prefix = candidate.program.ops[:-1]
    tail = candidate.program.ops[-1]
    dims_target = diff.get("dims_target", (0, 0))

    if tail.name in {"tile_to_target", "pad", "crop"} and dims_target != diff.get("dims_pred"):
        params = dict(tail.params)
        if tail.name == "tile_to_target":
            params["target_h"], params["target_w"] = dims_target
        elif tail.name == "pad":
            target_h, target_w = dims_target
            params.update({"bottom": max(0, target_h - diff["dims_pred"][0]), "right": max(0, target_w - diff["dims_pred"][1])})
        elif tail.name == "crop":
            target_h, target_w = dims_target
            params.update({"r1": target_h, "c1": target_w})
        variants.append(Program(prefix + [Op(tail.name, params)]))

    mismatch_pred = diff.get("mismatch_pred", {})
    mismatch_target = diff.get("mismatch_target", {})
    if tail.name == "map_color" and mismatch_pred and mismatch_target:
        params = dict(tail.params)
        colour_map = dict(params.get("color_map", {}))
        src = max(mismatch_pred, key=mismatch_pred.get)
        dst = max(mismatch_target, key=mismatch_target.get)
        colour_map[src] = dst
        params["color_map"] = colour_map
        variants.append(Program(prefix + [Op("map_color", params)]))

    if ascii_diff and ascii_diff.count("X") > ascii_diff.count("*"):
        variants.append(Program(candidate.program.ops + [Op("remove_small_objects", {"min_size": 2})]))

    return variants[:3]


def pooled_revision(parents: List[Program]) -> Program:
    """Combine sequences from multiple ``parents`` into a hybrid program.

    The implementation is deterministic but performs lightweight mutation
    passes to encourage diversity: duplicated operations are pruned, macro
    operations are prioritised, ``map_color`` dictionaries are merged, and
    trailing operations from later parents are spliced near the head to mimic
    operator swaps.
    """

    if not parents:
        return Program([])

    # Start with the longest parent as the scaffold.
    scaffold = max(parents, key=lambda prog: len(prog.ops))
    merged_ops: List[Op] = [Op(op.name, dict(op.params)) for op in scaffold.ops]

    # Insert unique operations from other parents (insertion).
    seen_structs = {
        json.dumps({"name": op.name, "params": op.params}, sort_keys=True)
        for op in merged_ops
    }
    for parent in parents:
        for op in parent.ops:
            struct = json.dumps({"name": op.name, "params": op.params}, sort_keys=True)
            if struct in seen_structs:
                continue
            merged_ops.append(Op(op.name, dict(op.params)))
            seen_structs.add(struct)

    # Prioritise macros by moving them towards the front (operator swap).
    macros = [op for op in merged_ops if op.name.startswith("lib::")]
    if macros:
        non_macros = [op for op in merged_ops if not op.name.startswith("lib::")]
        merged_ops = macros + non_macros

    # Parameter tweaks: merge colour maps if multiple map_color ops exist.
    colour_ops = [op for op in merged_ops if op.name == "map_color"]
    if len(colour_ops) > 1:
        merged_map: Dict[Any, Any] = {}
        for op in colour_ops:
            merged_map.update(op.params.get("color_map", {}))
        merged_ops = [
            Op("map_color", {"color_map": merged_map}) if op.name == "map_color" else op
            for op in merged_ops
        ]

    # Deletion step: collapse consecutive duplicates after mutations.
    deduped: List[Op] = []
    seen_structs = set()
    for op in merged_ops:
        struct = json.dumps({"name": op.name, "params": op.params}, sort_keys=True)
        if struct in seen_structs:
            continue
        deduped.append(op)
        seen_structs.add(struct)

    return Program(deduped)


def _update_rule_confidence(cfg: SearchConfig, program: Program, success: float) -> None:
    if not cfg.rule_confidence_enabled or not program.ops:
        return
    last = program.ops[-1]
    macro_meta = cfg.macro_metadata.get(last.name)
    if macro_meta and "program" in macro_meta:
        base_program: Program = macro_meta["program"]
        update_rule_confidence_stats(cfg.rule_confidence, base_program.ops, success)
    elif macro_meta and "ops" in macro_meta:
        ops = [Op(spec["name"], spec.get("params", {})) for spec in macro_meta["ops"]]
        update_rule_confidence_stats(cfg.rule_confidence, ops, success)
    else:
        update_rule_confidence_stats(cfg.rule_confidence, [last], success)


def _op_priority(cfg: SearchConfig, op_name: str) -> float:
    priority = 0.0
    if cfg.rule_confidence_enabled:
        stats = cfg.rule_confidence.get(op_name)
        if stats:
            priority += stats.get("ema", 0.0)
    meta = cfg.macro_metadata.get(op_name, {}).get("meta")
    if isinstance(meta, dict):
        priority += float(meta.get("priority", 0.0))
        if "confidence" in meta:
            priority += float(meta["confidence"])
    if cfg.meta_bias_strength > 0.0:
        bias = cfg.family_preference_scores.get(op_name)
        if bias:
            priority += cfg.meta_bias_strength * bias
    if op_name.startswith("lib::"):
        priority += max(0.0, cfg.library_priority)
    return priority


def bfs_synthesize(task: Any, cfg: SearchConfig) -> Tuple[List[Program], SearchStats]:
    """Enumerate candidate programs under ``cfg`` with beam search."""

    start = time.time()
    rng = random.Random(cfg.seed)
    stats = SearchStats()
    train: List[Example] = getattr(task, "train", [])
    if not train:
        return [], stats
    first_in = train[0].input
    first_out = train[0].output
    target_dims = dims(first_out)

    base_program = Program([])
    base_outputs, base_primary, base_secondary = evaluate_program(base_program, train)
    base_neural = compute_neural_score(cfg, task, base_program, train, base_outputs)
    base_symbolic = (
        cfg.alpha_primary * (base_primary / max(1, len(train)))
        + cfg.beta_secondary * base_secondary
    )
    neural_bias = max(0.0, min(1.0, cfg.neural_bias_weight))
    base_combined = (1.0 - neural_bias) * base_symbolic + neural_bias * base_neural
    base_candidate = Candidate(
        base_program,
        base_primary,
        base_secondary,
        base_outputs,
        program_struct_signature(base_program),
        _behaviour_blob(base_outputs),
        None,
        base_neural,
        base_combined,
    )
    beam: List[Candidate] = [base_candidate]
    valids: List[Candidate] = []
    if base_primary == len(train):
        valids.append(base_candidate)
    stats.best_primary = base_primary
    stats.best_secondary = base_secondary
    if base_outputs:
        distance = hamming_like_distance(base_outputs[0], first_out)
        stats.best_firstpair_distance = distance
        stats.best_dims_match = dims(base_outputs[0]) == target_dims

    def candidate_sort_key(candidate: Candidate) -> Tuple[float, float, float, float]:
        return (
            float(candidate.primary),
            float(candidate.secondary),
            float(candidate.combined_score),
            float(-len(candidate.program.ops)),
        )

    for depth in range(cfg.max_depth):
        if time.time() - start > cfg.time_budget_s:
            stats.budget_exhausted = True
            break

        structural_enabled = cfg.dedup_mode in {"structural", "both"}
        behaviour_enabled = cfg.dedup_mode in {"behavior", "both"}
        evaluated_structs: set[str] = set()
        behaviour_cache: Dict[str, Candidate] = {}
        candidate_pool: List[Candidate] = []

        def _should_skip(struct_sig: str) -> bool:
            if not structural_enabled:
                return False
            if struct_sig in evaluated_structs:
                stats.dedup_struct_hits += 1
                return True
            evaluated_structs.add(struct_sig)
            return False

        def _register_candidate(program: Program, outputs: List[Grid], primary: int, secondary: float) -> None:
            struct_sig = program_struct_signature(program)
            behavior_sig = _behaviour_blob(outputs)
            last_op = program.ops[-1].name if program.ops else None
            neural_score = compute_neural_score(cfg, task, program, train, outputs)
            symbolic = (
                cfg.alpha_primary * (primary / max(1, len(train)))
                + cfg.beta_secondary * secondary
            )
            combined = (1.0 - neural_bias) * symbolic + neural_bias * neural_score
            if any(op.name.startswith("lib::") for op in program.ops):
                combined *= max(1.0, cfg.library_priority)
            candidate = Candidate(
                program,
                primary,
                secondary,
                outputs,
                struct_sig,
                behavior_sig,
                last_op,
                neural_score,
                combined,
            )
            candidate_pool.append(candidate)
            existing = behaviour_cache.get(behavior_sig)
            if existing and behaviour_enabled:
                stats.dedup_behavior_hits += 1
                if candidate_sort_key(candidate) <= candidate_sort_key(existing):
                    return
            behaviour_cache[behavior_sig] = candidate
            stats.best_primary = max(stats.best_primary, primary)
            stats.best_secondary = max(stats.best_secondary, secondary)
            if last_op and last_op.startswith("lib::"):
                stats.macro_uses[last_op] = stats.macro_uses.get(last_op, 0) + 1
            _update_rule_confidence(cfg, program, primary / max(1, len(train)))
            if primary == len(train):
                valids.append(candidate)

        for entry in beam:
            parent_program = entry.program
            parent_output = entry.outputs[0] if entry.outputs else parent_program.apply(first_in)
            parent_accuracy = secondary_cell_accuracy(parent_output, first_out)
            parent_primary = int(eq_grid(parent_output, first_out))
            op_order = sorted(cfg.allow_ops, key=lambda name: -_op_priority(cfg, name))
            for op_name in op_order:
                for params in enumerate_params_for_op(op_name, task, features=cfg.task_features):
                    candidate_program = Program(parent_program.ops + [Op(op_name, params)])
                    stats.total_candidates += 1
                    struct_sig = program_struct_signature(candidate_program)
                    if _should_skip(struct_sig):
                        continue
                    try:
                        outputs, primary, secondary = evaluate_program(candidate_program, train)
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    first_output = outputs[0] if outputs else [[0]]
                    first_accuracy = secondary_cell_accuracy(first_output, first_out)
                    first_primary = int(eq_grid(first_output, first_out))
                    if first_primary < parent_primary and first_accuracy + 1e-6 < parent_accuracy:
                        continue
                    distance = hamming_like_distance(first_output, first_out)
                    if stats.best_firstpair_distance is None or distance < stats.best_firstpair_distance:
                        stats.best_firstpair_distance = distance
                        stats.best_dims_match = dims(first_output) == target_dims
                    _register_candidate(candidate_program, outputs, primary, secondary)

        if behaviour_enabled:
            next_candidates = list(behaviour_cache.values())
        else:
            next_candidates = list(candidate_pool)
        if not next_candidates:
            break

        time_elapsed = time.time() - start
        if cfg.enable_revisions and time_elapsed < cfg.time_budget_s * 0.9:
            stats.revision_rounds += 1
            top_parents = sorted(next_candidates, key=candidate_sort_key, reverse=True)[: cfg.parents_per_revision]
            generated = 0
            for parent in top_parents[: cfg.max_individual_revisions]:
                diff = diff_grid(parent.outputs[0], first_out)
                ascii_diff = diff.get("ascii") if cfg.feedback_diff else None
                for revised in revise_program_individual(parent, diff, ascii_diff=ascii_diff):
                    struct_sig = program_struct_signature(revised)
                    if _should_skip(struct_sig):
                        continue
                    stats.total_candidates += 1
                    try:
                        outputs, primary, secondary = evaluate_program(revised, train)
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    first_output = outputs[0] if outputs else [[0]]
                    distance = hamming_like_distance(first_output, first_out)
                    if stats.best_firstpair_distance is None or distance < stats.best_firstpair_distance:
                        stats.best_firstpair_distance = distance
                        stats.best_dims_match = dims(first_output) == target_dims
                    _register_candidate(revised, outputs, primary, secondary)
                    generated += 1
                for revised in feedback_revision(parent, diff, ascii_diff=ascii_diff):
                    struct_sig = program_struct_signature(revised)
                    if _should_skip(struct_sig):
                        continue
                    stats.total_candidates += 1
                    try:
                        outputs, primary, secondary = evaluate_program(revised, train)
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    first_output = outputs[0] if outputs else [[0]]
                    distance = hamming_like_distance(first_output, first_out)
                    if stats.best_firstpair_distance is None or distance < stats.best_firstpair_distance:
                        stats.best_firstpair_distance = distance
                        stats.best_dims_match = dims(first_output) == target_dims
                    _register_candidate(revised, outputs, primary, secondary)
                    generated += 1
            if cfg.use_pooled_revisions and cfg.max_pooled_revisions > 0:
                group_size = max(2, min(3, len(top_parents)))
                for idx in range(0, len(top_parents), group_size):
                    if generated >= cfg.max_pooled_revisions:
                        break
                    group = top_parents[idx : idx + group_size]
                    if len(group) < 2:
                        continue
                    pooled = pooled_revision([cand.program for cand in group])
                    struct_sig = program_struct_signature(pooled)
                    if _should_skip(struct_sig):
                        continue
                    stats.total_candidates += 1
                    try:
                        outputs, primary, secondary = evaluate_program(pooled, train)
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    first_output = outputs[0] if outputs else [[0]]
                    distance = hamming_like_distance(first_output, first_out)
                    if stats.best_firstpair_distance is None or distance < stats.best_firstpair_distance:
                        stats.best_firstpair_distance = distance
                        stats.best_dims_match = dims(first_output) == target_dims
                    _register_candidate(pooled, outputs, primary, secondary)
                    generated += 1
            stats.revisions_generated += generated

        stats.kept_after_filter += len(next_candidates)
        ranked = sorted(list(next_candidates), key=candidate_sort_key, reverse=True)
        elites = ranked[: min(cfg.elite, len(ranked))]
        stats.num_elites_kept += len(elites)
        survivors: List[Candidate] = list(elites)
        remaining = ranked[len(elites) :]
        slots = max(cfg.beam_size - len(survivors), 0)
        stats.temperature_trace.append(cfg.effective_temperature)
        if slots > 0:
            if cfg.explore_bias and time.time() - start < cfg.time_budget_s * 0.95 and remaining:
                scores = [cand.combined_score for cand in remaining]
                pool = remaining[:]
                while pool and len(survivors) < cfg.beam_size:
                    weights = softmax_weights(scores[: len(pool)], cfg.effective_temperature)
                    choice = rng.choices(list(range(len(pool))), weights=weights, k=1)[0]
                    survivors.append(pool.pop(choice))
                    scores.pop(choice)
                stats.num_sampled_kept += len(survivors) - len(elites)
            else:
                survivors.extend(remaining[:slots])
        used_behaviours = {cand.behavior_sig for cand in survivors}
        if cfg.min_diverse > 0:
            idx = len(elites)
            while len(used_behaviours) < cfg.min_diverse and idx < len(ranked):
                candidate = ranked[idx]
                idx += 1
                if candidate.behavior_sig in used_behaviours:
                    continue
                if candidate in survivors:
                    used_behaviours.add(candidate.behavior_sig)
                    continue
                if len(survivors) < cfg.beam_size:
                    survivors.append(candidate)
                    used_behaviours.add(candidate.behavior_sig)
                elif len(survivors) > len(elites):
                    replace_index = min(
                        range(len(elites), len(survivors)),
                        key=lambda idx_: survivors[idx_].combined_score,
                    )
                    survivors[replace_index] = candidate
                    used_behaviours.add(candidate.behavior_sig)
                else:
                    break
        beam = survivors
        stats.survivors_per_depth.append(len(beam))
        stats.diversity_counts.append(len({cand.behavior_sig for cand in beam}))
        stats.depth_reached = depth + 1
        if any(candidate.primary == len(train) for candidate in beam):
            break

    stats.time_elapsed = time.time() - start
    if stats.time_elapsed >= cfg.time_budget_s:
        stats.budget_exhausted = True
    stats.valids_found = sum(1 for cand in valids if cand.primary == len(train))
    unique = {cand.struct_sig: cand for cand in sorted(valids, key=candidate_sort_key, reverse=True)}
    return [cand.program for cand in unique.values()], stats


def pick_two(programs: List[Program]) -> Tuple[Optional[Program], Optional[Program]]:
    """Return the best two programs for downstream use."""

    if not programs:
        return None, None
    ordered = sorted(programs, key=lambda prog: len(prog.ops))
    return ordered[0], (ordered[1] if len(ordered) > 1 else None)


__all__ = [
    "SearchConfig",
    "SearchStats",
    "Candidate",
    "bfs_synthesize",
    "enumerate_params_for_op",
    "program_fits_all",
    "pick_two",
    "hamming_like_distance",
    "diff_grid",
    "generate_ascii_diff",
    "feedback_revision",
    "pooled_revision",
    "program_behavior_signature",
    "program_struct_signature",
    "score_program",
    "softmax_weights",
    "compute_neural_score",
]
