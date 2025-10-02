from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from reap.nl_interface import nl_to_program, program_to_nl, roundtrip_program
from reap.neural_guidance import NeuralGuidance
from reap.program import Op, Program
from reap.program_library import LibraryManager
from reap.rule_engine import update_rule_confidence
from reap.search import (
    SearchConfig,
    bfs_synthesize,
    diff_grid,
    pooled_revision,
    program_behavior_signature,
    repair_strategies,
    score_program,
    softmax_weights,
)
from reap.solver import (
    classify_task,
    register_library_macros,
    solve_task,
    unregister_macros,
)
from reap.types import Example


def make_example(inp, out):
    return Example(input=inp, output=out)


def test_program_behavior_signature_dedup():
    examples = [
        make_example([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
    ]
    prog_a = Program([Op("map_color", {"color_map": {1: 2}})])
    prog_b = Program([
        Op("map_color", {"color_map": {1: 2}}),
        Op("map_color", {"color_map": {2: 2}}),
    ])
    sig_a = program_behavior_signature(prog_a, examples)
    sig_b = program_behavior_signature(prog_b, examples)
    assert sig_a == sig_b


def test_score_program_primary_secondary():
    examples = [
        make_example([[1]], [[2]]),
        make_example([[3]], [[4]]),
    ]
    prog = Program([Op("map_color", {"color_map": {1: 2}})])
    primary, secondary = score_program(prog, examples)
    assert primary == 1
    # First example matches exactly, second is unchanged
    assert secondary == pytest.approx(0.5)


def test_library_manager_roundtrip(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "library.json"))
    examples = [make_example([[1]], [[2]])]
    program = Program([Op("map_color", {"color_map": {1: 2}})])
    lib_id = manager.register_program(program, examples)
    manager.save()

    new_manager = LibraryManager(path=str(tmp_path / "library.json"))
    new_manager.load()
    items = list(new_manager.iter_items())
    assert items and items[0].identifier == lib_id
    assert items[0].meta.get("confidence", 0) > 0

    macro_names = register_library_macros(items)
    try:
        macro_name = f"lib::{lib_id}"
        macro_program = Program([Op(macro_name, {})])
        out = macro_program.apply([[1]])
        assert out == [[2]]
    finally:
        unregister_macros(macro_names)


def test_repair_strategies_map_color_suggested():
    pred = [[1, 0], [0, 1]]
    target = [[2, 0], [0, 2]]
    diff = diff_grid(pred, target)
    ops = repair_strategies(diff)
    assert any(op.name == "map_color" for op in ops)


def test_macro_reuse_smoke(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "lib.json"))
    examples = [make_example([[1, 1]], [[2, 2]])]
    program = Program([Op("map_color", {"color_map": {1: 2}})])
    lib_id = manager.register_program(program, examples)
    manager.save()

    reloaded = LibraryManager(path=str(tmp_path / "lib.json"))
    reloaded.load()
    items = list(reloaded.iter_items())
    macro_names = register_library_macros(items)
    try:
        macro_name = f"lib::{lib_id}"
        prog = Program([Op(macro_name, {})])
        out = prog.apply([[1, 1]])
        assert out == [[2, 2]]
    finally:
        unregister_macros(macro_names)


def test_pooled_revision_merges_ops():
    prog_a = Program([Op("rotate", {"angle": 90}), Op("map_color", {"color_map": {1: 2}})])
    prog_b = Program([Op("flip", {"axis": "h"})])
    merged = pooled_revision([prog_a, prog_b])
    assert [op.name for op in merged.ops] == ["rotate", "map_color", "flip"]


def test_equivalence_collapse_counts():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 1], [1, 1]], [[1, 1], [1, 1]])]
    task.test = []
    cfg = SearchConfig(max_depth=2, beam_size=8, allow_ops=["rotate", "flip"], dedup_mode="behavior")
    cfg.time_budget_s = 0.5
    cfg.task_features = {}
    programs, stats = bfs_synthesize(task, cfg)
    assert stats.dedup_behavior_hits >= 1


def test_rule_confidence_update():
    registry: Dict[str, Dict[str, float]] = {}
    ops = [Op("map_color", {"color_map": {1: 2}})]
    update_rule_confidence(registry, ops, 1.0)
    assert registry["map_color"]["ema"] > 0
    previous = registry["map_color"]["ema"]
    update_rule_confidence(registry, ops, 0.0)
    assert registry["map_color"]["ema"] < previous


def test_task_classification_bias_flags():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 0], [0, 1]], [[0, 1], [1, 0]])]
    features = classify_task(task)
    assert "symmetry" in features.get("families", ())
    assert "flip" in features.get("family_prioritised_ops", [])


def test_library_abstraction_generates_macro(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "lib_abs.json"))
    examples = [make_example([[1]], [[1]])]
    program = Program([Op("rotate", {"angle": 90}), Op("flip", {"axis": "h"})])
    manager.grow_from_program(program, examples, mode="flat")
    created = manager.grow_from_program(program, examples, mode="flat")
    assert created  # second pass should create a macro
    for macro_id in created:
        assert macro_id in manager.items


def test_library_hierarchical_growth(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "hier.json"))
    examples = [make_example([[1]], [[1]])]
    base = Program([Op("rotate", {"angle": 90}), Op("flip", {"axis": "h"})])
    manager.grow_from_program(base, examples, mode="hierarchical")
    created = manager.grow_from_program(base, examples, mode="hierarchical")
    assert created
    for macro_id in created:
        assert manager.items[macro_id].meta.get("confidence", 0.0) >= 0.5


def test_library_confidence_updates(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "conf.json"))
    examples = [make_example([[1, 1]], [[2, 2]])]
    program = Program([Op("map_color", {"color_map": {1: 2}})])
    lib_id = manager.register_program(program, examples)
    before = manager.items[lib_id].meta.get("confidence", 0.0)
    manager.touch_success(lib_id, 1.0, 1.0)
    after = manager.items[lib_id].meta.get("confidence", 0.0)
    assert after > before


def test_softmax_temperature_bias():
    scores = [10.0, 4.0, 1.0]
    cold = softmax_weights(scores, 0.1)
    warm = softmax_weights(scores, 1.5)
    assert cold[0] > warm[0]
    assert warm[-1] > cold[-1]


def test_softmax_sampling_reproducible():
    weights = softmax_weights([5.0, 1.0, 0.5], 0.4)
    rng_a = random.Random(1337)
    picks_a = [rng_a.choices(range(3), weights=weights, k=1)[0] for _ in range(6)]
    rng_b = random.Random(1337)
    picks_b = [rng_b.choices(range(3), weights=weights, k=1)[0] for _ in range(6)]
    assert picks_a == picks_b


def test_nl_roundtrip_equivalence():
    program = Program(
        [
            Op("rotate", {"angle": 90}),
            Op("map_color", {"color_map": {1: 2, 3: 4}}),
        ]
    )
    text = program_to_nl(program)
    rebuilt = nl_to_program(text)
    assert [(op.name, op.params) for op in rebuilt.ops] == [
        (op.name, op.params) for op in program.ops
    ]
    assert [(op.name, op.params) for op in roundtrip_program(program).ops] == [
        (op.name, op.params) for op in program.ops
    ]


def test_meta_bias_preferences():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 0], [0, 1]], [[0, 1], [1, 0]])]
    task.test = []
    cfg = SearchConfig(max_depth=0, beam_size=1)
    cfg_copy = SearchConfig(**cfg.__dict__)
    family_stats = {"symmetry": {"ops": {"flip": 4}, "macros": {}, "confidence": 0.6}}
    solve_task(
        task,
        time_budget_s=0.1,
        cfg=cfg_copy,
        library_items=None,
        enable_library=False,
        enable_explore_bias=False,
        enable_revisions=False,
        enable_pooled_revisions=False,
        enable_equivalence=True,
        enable_task_family_bias=True,
        enable_feedback_diff=False,
        rule_confidence={},
        task_family_stats=family_stats,
        meta_bias_strength=0.8,
    )
    assert cfg_copy.family_preference_scores.get("flip", 0.0) > 0.0


def test_task_classification_repetition_and_color():
    task = type("Spec", (), {})()
    task.train = [
        make_example([[1, 1], [1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
        make_example([[2, 3], [3, 2]], [[3, 2, 3, 2], [2, 3, 2, 3]]),
    ]
    profile = classify_task(task)
    families = profile.get("families", ())
    assert "repetition" in families
    assert "color_shift" in families


def test_neural_guidance_default_score():
    guidance = NeuralGuidance()
    program = Program([])
    assert guidance.score(None, program, [[0]], [[0]]) == pytest.approx(0.5)


def test_dedup_mode_structural_only():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 1], [1, 1]], [[1, 1], [1, 1]])]
    task.test = []
    cfg = SearchConfig(max_depth=1, beam_size=4, allow_ops=["rotate", "rotate"], dedup_mode="structural")
    cfg.time_budget_s = 0.1
    cfg.task_features = {}
    cfg.deduplicate_equivalence = False
    cfg.__post_init__()
    programs, stats = bfs_synthesize(task, cfg)
    assert programs
    assert stats.dedup_struct_hits > 0
    assert stats.dedup_behavior_hits == 0


def test_dedup_mode_behavior_only():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 1], [1, 1]], [[1, 1], [1, 1]])]
    task.test = []
    cfg = SearchConfig(max_depth=1, beam_size=4, allow_ops=["rotate", "flip"], dedup_mode="behavior")
    cfg.time_budget_s = 0.1
    cfg.task_features = {}
    cfg.__post_init__()
    programs, stats = bfs_synthesize(task, cfg)
    assert programs
    assert stats.dedup_behavior_hits > 0
    assert stats.dedup_struct_hits == 0


def test_dedup_mode_both_hits():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 1], [1, 1]], [[1, 1], [1, 1]])]
    task.test = []
    cfg = SearchConfig(
        max_depth=1,
        beam_size=4,
        allow_ops=["rotate", "rotate", "flip"],
        dedup_mode="both",
    )
    cfg.time_budget_s = 0.1
    cfg.task_features = {}
    cfg.__post_init__()
    programs, stats = bfs_synthesize(task, cfg)
    assert programs
    assert stats.dedup_behavior_hits > 0
    assert stats.dedup_struct_hits > 0


def test_min_diverse_preserves_different_behaviours():
    task = type("Spec", (), {})()
    task.train = [make_example([[1, 0], [0, 0]], [[2, 0], [0, 0]])]
    task.test = []
    cfg = SearchConfig(
        max_depth=1,
        beam_size=3,
        allow_ops=["rotate", "flip", "map_color"],
        dedup_mode="behavior",
        min_diverse=3,
    )
    cfg.time_budget_s = 0.5
    cfg.task_features = {}
    programs, stats = bfs_synthesize(task, cfg)
    assert programs
    assert stats.diversity_counts
    assert stats.diversity_counts[-1] >= 3


def test_time_budget_marks_exhaustion():
    task = type("Spec", (), {})()
    task.train = [make_example([[1]], [[1]])]
    task.test = []
    cfg = SearchConfig(max_depth=3, beam_size=2, allow_ops=["rotate"], dedup_mode="structural")
    cfg.time_budget_s = 0.0
    cfg.task_features = {}
    programs, stats = bfs_synthesize(task, cfg)
    assert programs
    assert stats.budget_exhausted is True


def test_library_priority_increases_macro_usage(tmp_path: Path):
    manager = LibraryManager(path=str(tmp_path / "macro.json"))
    examples = [make_example([[1]], [[2]])]
    base_program = Program([Op("map_color", {"color_map": {1: 2}})])
    lib_id = manager.register_program(base_program, examples)
    items = list(manager.iter_items())
    macro_names = register_library_macros(items)
    try:
        macro_name = macro_names[0]
        task = type("Spec", (), {})()
        task.train = examples
        task.test = []
        cfg = SearchConfig(
            max_depth=1,
            beam_size=2,
            allow_ops=["map_color", macro_name],
            dedup_mode="both",
            library_priority=5.0,
        )
        cfg.time_budget_s = 0.5
        cfg.task_features = {}
        cfg.macro_metadata = {macro_name: {"meta": {"priority": 1.0, "confidence": 1.0}}}
        programs, stats = bfs_synthesize(task, cfg)
        assert programs
        assert any(name in stats.macro_uses for name in macro_names)
    finally:
        unregister_macros(macro_names)
