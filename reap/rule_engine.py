"""Condition-action rule synthesis helpers.

The rule system remains intentionally lightweight: conditions are evaluated on
training data to decide which pre-wired action sequences should be considered
by the search. This keeps the approach symbolic and auditable while still
allowing higher-level heuristics to bias the enumeration order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .grid_utils import dims, eq_grid
from .objects import extract_objects
from .program import Op, Program
from .types import Example


@dataclass(frozen=True)
class Condition:
    """Logical predicate used to guard a rule."""

    name: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class Rule:
    """Condition-action rule compiled into a macro candidate."""

    when: List[Condition]
    then_ops: List[Op]
    priority: int = 0


def _collect_colours(grid: List[List[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _has_symmetry(grid: List[List[int]], axis: str) -> bool:
    from .dsl import flip

    return eq_grid(grid, flip(grid, axis))


def _colour_bijection(examples: Iterable[Example]) -> Dict[int, int] | None:
    mapping: Dict[int, int] = {}
    reverse: Dict[int, int] = {}
    for pair in examples:
        src_counts = _collect_colours(pair.input)
        dst_counts = _collect_colours(pair.output)
        for colour in src_counts:
            if colour == 0:
                continue
            candidates = [dst for dst in dst_counts if dst_counts[dst] == src_counts[colour]]
            if not candidates:
                return None
            dst = candidates[0]
            if colour in mapping and mapping[colour] != dst:
                return None
            if dst in reverse and reverse[dst] != colour:
                return None
            mapping[colour] = dst
            reverse[dst] = colour
    return mapping if mapping else None


def suggest_rules(train: List[Example]) -> List[Rule]:
    """Return heuristic condition-action rules inferred from ``train`` pairs."""

    rules: List[Rule] = []
    if not train:
        return rules

    # Symmetry completion
    if all(_has_symmetry(pair.output, "h") for pair in train):
        rules.append(
            Rule(
                when=[Condition("has_symmetry", {"axis": "h"})],
                then_ops=[Op("complete_symmetry", {"axis": "h"})],
                priority=5,
            )
        )
    if all(_has_symmetry(pair.output, "v") for pair in train):
        rules.append(
            Rule(
                when=[Condition("has_symmetry", {"axis": "v"})],
                then_ops=[Op("complete_symmetry", {"axis": "v"})],
                priority=5,
            )
        )

    # Colour bijection implies map_color
    colour_map = _colour_bijection(train)
    if colour_map:
        rules.append(
            Rule(
                when=[Condition("has_colors", {"count": len(colour_map)})],
                then_ops=[Op("map_color", {"color_map": colour_map})],
                priority=10,
            )
        )

    # Object reductions (e.g. keep largest)
    if all(len(extract_objects(pair.input)) > len(extract_objects(pair.output)) for pair in train):
        rules.append(
            Rule(
                when=[Condition("object_count", {"cmp": ">", "k": len(extract_objects(train[0].output))})],
                then_ops=[Op("keep_largest_object", {})],
                priority=8,
            )
        )

    # Size projections when height/width differ but share factors
    dims_in = {dims(pair.input) for pair in train}
    dims_out = {dims(pair.output) for pair in train}
    if len(dims_in) == 1 and len(dims_out) == 1:
        (hin, win) = next(iter(dims_in))
        (hout, wout) = next(iter(dims_out))
        if hin and win and hout and wout:
            if hout == hin and wout != win:
                rules.append(
                    Rule(
                        when=[Condition("grid_size_equals", {"h": hin, "w": win})],
                        then_ops=[Op("repeat_objects_horiz", {"count": wout // max(1, win)})],
                        priority=4,
                    )
                )
            if hout != hin and wout == win:
                rules.append(
                    Rule(
                        when=[Condition("grid_size_equals", {"h": hin, "w": win})],
                        then_ops=[Op("repeat_scale", {"k": max(2, hout // max(1, hin))})],
                        priority=4,
                    )
                )

    return rules


def compile_rules(rules: List[Rule]) -> List[Program]:
    """Convert :class:`Rule` objects to :class:`Program` sequences."""

    return [Program(rule.then_ops) for rule in sorted(rules, key=lambda r: -r.priority)]


def update_rule_confidence(
    confidence: MutableMapping[str, Dict[str, float]],
    rule_ops: Sequence[Op],
    success_score: float,
    *,
    decay: float = 0.95,
) -> None:
    """Update per-primitive confidence statistics.

    Parameters
    ----------
    confidence:
        Mapping tracking ``attempts``, ``success`` and ``ema`` for each DSL
        primitive. The structure mirrors what the search configuration expects.
    rule_ops:
        Operations that were attempted as part of a rule.
    success_score:
        Fraction in ``[0, 1]`` representing how well the rule performed.
    decay:
        Multiplicative decay applied when ``success_score`` is zero to penalise
        repeated failures without permanently banning the operator.
    """

    for op in rule_ops:
        stats = confidence.setdefault(op.name, {"attempts": 0, "success": 0.0, "ema": 0.0})
        stats["attempts"] += 1
        if success_score > 0:
            stats["success"] += success_score
        ema = stats.get("ema", 0.0)
        ema = 0.9 * ema + 0.1 * success_score
        if success_score == 0:
            ema *= decay
        stats["ema"] = ema


def prioritise_rules(
    rules: Sequence[Rule],
    confidence: Mapping[str, Dict[str, float]] | None,
    *,
    temperature: float = 0.5,
    limit: int | None = None,
) -> List[Rule]:
    """Order rules according to confidence-aware softmax sampling."""

    if not rules:
        return []
    if not confidence:
        ordered = sorted(rules, key=lambda rule: -rule.priority)
        return ordered[:limit] if limit is not None else ordered

    weights: List[float] = []
    for rule in rules:
        score = 0.0
        for op in rule.then_ops:
            stats = confidence.get(op.name)
            if stats:
                score += stats.get("ema", 0.0)
        weights.append(score)

    max_w = max(weights) if weights else 0.0
    norm_weights = []
    denom = 0.0
    for weight in weights:
        val = (weight - max_w) / max(temperature, 1e-6)
        exp_w = math.exp(val)
        norm_weights.append(exp_w)
        denom += exp_w
    if denom == 0.0:
        ordered = sorted(rules, key=lambda rule: -rule.priority)
        return ordered[:limit] if limit is not None else ordered

    paired = sorted(
        zip(rules, norm_weights),
        key=lambda item: item[1],
        reverse=True,
    )
    ordered_rules = [rule for rule, _ in paired]
    return ordered_rules[:limit] if limit is not None else ordered_rules


__all__ = [
    "Condition",
    "Rule",
    "suggest_rules",
    "compile_rules",
    "update_rule_confidence",
    "prioritise_rules",
]
