"""reap.cli
============

Command-line entry point mirroring the original REAP script but delegating most
logic to the modular package components.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from .logging_utils import log_missing
from .grid_utils import eq_grid
from .memory import hash_train, load_memory_db, save_memory_db
from .neural_guidance import NeuralGuidance
from .nl_interface import program_to_nl
from .program_library import LibraryManager
from .search import SearchConfig
from .solver import solve_task
from .types import Example


def solve_single_task(
    task_id: str,
    spec: dict,
    memory_solutions: Dict[str, Any],
    rule_confidence: Dict[str, Dict[str, float]],
    time_per_task: float,
    solver_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Worker function executed in subprocesses."""

    from time import time as now

    start_time = now()
    parsed = type("Spec", (), {})()
    parsed.train = [Example(input=pair["input"], output=pair["output"]) for pair in spec["train"]]
    parsed.test = [Example(input=item["input"]) for item in spec["test"]]

    train_payload = [{"input": pair.input, "output": pair.output} for pair in parsed.train]
    train_hash = hash_train(train_payload)
    if train_hash in memory_solutions:
        grids = memory_solutions[train_hash]
        elapsed = now() - start_time
        return {
            "tid": task_id,
            "submission": [{"attempt_1": grid, "attempt_2": grid} for grid in grids],
            "from_mem": True,
            "elapsed": elapsed,
            "stats": None,
            "train_hash": train_hash,
            "train": parsed.train,
            "programs": [],
            "rule_confidence": rule_confidence,
            "features": {},
            "nl_programs": [] if solver_options.get("export_nl") else None,
        }

    cfg_template: SearchConfig = solver_options["config"]
    cfg_copy = SearchConfig(**cfg_template.__dict__)
    attempts, stats, programs, updated_rule_conf = solve_task(
        parsed,
        time_budget_s=time_per_task,
        cfg=cfg_copy,
        library_items=solver_options.get("library_items"),
        enable_library=solver_options.get("enable_library", True),
        enable_explore_bias=solver_options.get("enable_explore_bias", True),
        enable_revisions=solver_options.get("enable_revisions", True),
        enable_pooled_revisions=solver_options.get("enable_pooled_revisions", True),
        enable_equivalence=solver_options.get("enable_equivalence", True),
        enable_task_family_bias=solver_options.get("task_family_bias", True),
        enable_feedback_diff=solver_options.get("feedback_diff", False),
        rule_confidence={k: dict(v) for k, v in rule_confidence.items()},
        task_family_stats=solver_options.get("task_family_stats"),
        meta_bias_strength=solver_options.get("meta_bias_strength", 0.0),
        dedup_mode=solver_options.get("dedup_mode", "both"),
        neural_guidance=solver_options.get("neural_guidance"),
        neural_bias=solver_options.get("neural_bias", 0.0),
        exploration_temp=solver_options.get("exploration_temp"),
        min_diverse=solver_options.get("min_diverse"),
        library_priority=solver_options.get("library_priority"),
    )
    elapsed = now() - start_time
    features = dict(cfg_copy.task_features)
    nl_programs: List[str] | None = None
    if solver_options.get("export_nl"):
        nl_programs = [program_to_nl(program) for program in programs]
    return {
        "tid": task_id,
        "submission": attempts,
        "from_mem": False,
        "elapsed": elapsed,
        "stats": stats,
        "train_hash": train_hash,
        "train": parsed.train,
        "programs": programs,
        "rule_confidence": updated_rule_conf,
        "features": features,
        "nl_programs": nl_programs,
    }


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the solver."""

    parser = argparse.ArgumentParser("REAP Solver")
    parser.add_argument("--infile", required=True, help="ARC tasks JSON input")
    parser.add_argument("--outfile", default="submission.json", help="Output predictions file")
    parser.add_argument("--time_per_task", type=float, default=20.0, help="Time budget per task (sec)")
    parser.add_argument("--time-budget", type=float, default=None, help="Override time budget per task (sec)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes (parallelism)")
    parser.add_argument("--no-library", action="store_true", help="Disable dynamic program library usage")
    parser.add_argument("--no-explore-bias", action="store_true", help="Disable exploration-biased sampling")
    parser.add_argument("--softmax-T", type=float, default=0.35, help="[deprecated] softmax temperature alias")
    parser.add_argument(
        "--exploration_softmax_temp",
        type=float,
        default=None,
        help="Softmax temperature controlling exploration bias",
    )
    parser.add_argument("--elite", type=int, default=16, help="Number of elite programs kept each depth")
    parser.add_argument("--alpha", type=float, default=1.0, help="Primary score weight for sampling")
    parser.add_argument("--beta", type=float, default=1.0, help="Secondary score weight for sampling")
    parser.add_argument("--exploration-temp", type=float, default=1.0, help="Exploration temperature multiplier")
    parser.add_argument("--min-diverse", type=int, default=5, help="Minimum diverse candidates kept per depth")
    parser.add_argument("--revisions", dest="revisions", action="store_true", help="Enable feedback-driven revisions")
    parser.add_argument("--no-revisions", dest="revisions", action="store_false")
    parser.set_defaults(revisions=True)
    parser.add_argument("--parents-per-revision", type=int, default=8, help="Parents considered per revision round")
    parser.add_argument("--max-individual-revisions", type=int, default=8, help="Max individual revisions generated")
    parser.add_argument("--max-pooled-revisions", type=int, default=4, help="Max pooled revisions generated")
    parser.add_argument("--use-pooled-revisions", dest="pooled_revisions", action="store_true", help="Enable pooled revision synthesis")
    parser.add_argument("--no-pooled-revisions", dest="pooled_revisions", action="store_false")
    parser.set_defaults(pooled_revisions=True)
    parser.add_argument("--deduplicate-equivalence", dest="deduplicate_equivalence", action="store_true", help="Collapse behaviour-equivalent programs")
    parser.add_argument("--no-deduplicate-equivalence", dest="deduplicate_equivalence", action="store_false")
    parser.set_defaults(deduplicate_equivalence=True)
    parser.add_argument("--dedup-mode", choices=["behavior", "structural", "both"], default="both", help="Deduplication strategy")
    parser.add_argument("--enable-rule-confidence", dest="rule_confidence", action="store_true", help="Enable rule confidence tracking")
    parser.add_argument("--disable-rule-confidence", dest="rule_confidence", action="store_false")
    parser.set_defaults(rule_confidence=True)
    parser.add_argument("--task-family-bias", dest="task_family_bias", action="store_true", help="Bias DSL ordering using task classification")
    parser.add_argument("--no-task-family-bias", dest="task_family_bias", action="store_false")
    parser.set_defaults(task_family_bias=True)
    parser.add_argument("--abstract-library", action="store_true", help="Enable flat macro abstraction (legacy flag)")
    parser.add_argument(
        "--library_growth_mode",
        choices=["none", "flat", "hierarchical"],
        default="none",
        help="Library abstraction strategy applied to solved programs",
    )
    parser.add_argument("--feedback-diff", dest="feedback_diff", action="store_true", help="Attach ASCII diffs during revisions")
    parser.add_argument("--no-feedback-diff", dest="feedback_diff", action="store_false")
    parser.set_defaults(feedback_diff=False)
    parser.add_argument("--export_nl", action="store_true", help="Export natural-language program descriptions")
    parser.add_argument("--neural-bias", type=float, default=0.0, help="Blend factor for neural guidance (0 disables)")
    parser.add_argument("--library-priority", type=float, default=2.0, help="Priority boost for library macros")
    parser.add_argument(
        "--meta_bias_strength",
        type=float,
        default=0.0,
        help="Strength of task-family meta-learning bias (0 disables)",
    )
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.infile).read_text())
    memory_payload = load_memory_db()
    solutions_db = memory_payload.setdefault("solutions", {})
    rule_confidence = memory_payload.setdefault("rule_confidence", {})
    task_family_stats = memory_payload.setdefault("task_family_stats", {})
    library = LibraryManager()
    library.load()
    library_items = list(library.iter_items())
    solver_config = SearchConfig()
    time_budget = args.time_budget if args.time_budget is not None else args.time_per_task
    guidance = NeuralGuidance()
    temperature = (
        args.exploration_softmax_temp
        if args.exploration_softmax_temp is not None
        else args.softmax_T
    )
    solver_config.softmax_T = temperature
    solver_config.exploration_softmax_temp = temperature
    solver_config.elite = args.elite
    solver_config.alpha_primary = args.alpha
    solver_config.beta_secondary = args.beta
    solver_config.exploration_temp = args.exploration_temp
    solver_config.min_diverse = args.min_diverse
    solver_config.explore_bias = not args.no_explore_bias
    solver_config.enable_revisions = args.revisions
    solver_config.parents_per_revision = args.parents_per_revision
    solver_config.max_individual_revisions = args.max_individual_revisions
    solver_config.max_pooled_revisions = args.max_pooled_revisions
    solver_config.rule_confidence_enabled = args.rule_confidence
    solver_config.use_pooled_revisions = args.pooled_revisions
    solver_config.deduplicate_equivalence = args.deduplicate_equivalence
    solver_config.dedup_mode = args.dedup_mode
    solver_config.feedback_diff = args.feedback_diff
    solver_config.meta_bias_strength = args.meta_bias_strength
    solver_config.neural_bias_weight = args.neural_bias
    solver_config.library_priority = args.library_priority
    solver_config.__post_init__()
    submission: Dict[str, Any] = {}
    summary_metrics: Dict[str, Any] = {}
    if args.abstract_library and args.library_growth_mode == "none":
        args.library_growth_mode = "flat"
    task_nl: Dict[str, List[str]] = {}

    stats_csv_path = Path(args.outfile).with_suffix(".stats.csv")
    with stats_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "task_id",
            "elapsed",
            "from_mem",
            "depth_reached",
            "valids_found",
            "best_firstpair_distance",
            "survivors_per_depth",
            "kept_after_filter",
            "total_candidates",
            "crashes_during_apply",
            "best_dims_match",
            "best_primary",
            "best_secondary",
            "dedup_struct_hits",
            "dedup_behavior_hits",
            "revision_rounds",
            "revisions_generated",
            "num_elites_kept",
            "num_sampled_kept",
            "macro_uses",
            "budget_exhausted",
            "diversity_counts",
            "temperature_trace",
        ])

        if args.max_workers <= 0:
            args.max_workers = multiprocessing.cpu_count()

        solver_options = {
            "config": solver_config,
            "library_items": None if args.no_library else library_items,
            "enable_library": not args.no_library,
            "enable_explore_bias": not args.no_explore_bias,
            "enable_revisions": args.revisions,
            "enable_pooled_revisions": args.pooled_revisions,
            "enable_equivalence": args.deduplicate_equivalence,
            "task_family_bias": args.task_family_bias,
            "feedback_diff": args.feedback_diff,
            "abstract_library": args.abstract_library,
            "library_growth_mode": args.library_growth_mode,
            "export_nl": args.export_nl,
            "task_family_stats": task_family_stats,
            "meta_bias_strength": args.meta_bias_strength,
            "dedup_mode": args.dedup_mode,
            "neural_guidance": guidance,
            "neural_bias": args.neural_bias,
            "exploration_temp": args.exploration_temp,
            "min_diverse": args.min_diverse,
            "library_priority": args.library_priority,
            "time_budget": time_budget,
        }

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    solve_single_task,
                    task_id,
                    raw[task_id],
                    solutions_db,
                    rule_confidence,
                    time_budget,
                    solver_options,
                ): task_id
                for task_id in raw
            }

            for index, future in enumerate(as_completed(futures), start=1):
                task_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[{index}/{len(raw)}] Task {task_id} crashed: {exc}")
                    continue

                submission[task_id] = result["submission"]
                features = result.get("features", {})
                if args.export_nl:
                    nl_payload = result.get("nl_programs") or []
                    task_nl[task_id] = nl_payload
                updated_conf = result.get("rule_confidence")
                if updated_conf:
                    for op_name, stats_update in updated_conf.items():
                        current = rule_confidence.get(op_name, {"attempts": 0, "success": 0.0, "ema": 0.0})
                        current.update(stats_update)
                        rule_confidence[op_name] = current

                stats = result.get("stats")
                if result["from_mem"]:
                    print(f"[{index}/{len(raw)}] Task {task_id} solved from memory in {result['elapsed']:.2f}s.")
                    empty_tail = [""] * 20
                    empty_tail[16] = json.dumps({})
                    writer.writerow([task_id, result["elapsed"], True] + empty_tail)
                    summary_metrics[task_id] = {
                        "elapsed": result["elapsed"],
                        "from_memory": True,
                        "candidates": 0,
                        "budget_exhausted": False,
                        "diversity_counts": [],
                        "temperature_trace": [],
                    }
                else:
                    if not stats:
                        print(f"[{index}/{len(raw)}] Task {task_id} failed (no stats).")
                        empty_tail = [""] * 20
                        empty_tail[16] = json.dumps({})
                        writer.writerow([task_id, result["elapsed"], False] + empty_tail)
                        summary_metrics[task_id] = {
                            "elapsed": result["elapsed"],
                            "candidates": 0,
                            "budget_exhausted": False,
                            "diversity_counts": [],
                            "temperature_trace": [],
                        }
                    else:
                        print(
                            f"[{index}/{len(raw)}] Task {task_id} done in {result['elapsed']:.2f}s | "
                            f"depth={stats.depth_reached} valids={stats.valids_found} "
                            f"best_d={stats.best_firstpair_distance} beam={stats.survivors_per_depth}"
                        )
                        writer.writerow([
                            task_id,
                            result["elapsed"],
                            False,
                            getattr(stats, "depth_reached", ""),
                            getattr(stats, "valids_found", ""),
                            getattr(stats, "best_firstpair_distance", ""),
                            getattr(stats, "survivors_per_depth", ""),
                            getattr(stats, "kept_after_filter", ""),
                            getattr(stats, "total_candidates", ""),
                            getattr(stats, "crashes_during_apply", ""),
                            getattr(stats, "best_dims_match", ""),
                            getattr(stats, "best_primary", ""),
                            getattr(stats, "best_secondary", ""),
                            getattr(stats, "dedup_struct_hits", ""),
                            getattr(stats, "dedup_behavior_hits", ""),
                            getattr(stats, "revision_rounds", ""),
                            getattr(stats, "revisions_generated", ""),
                            getattr(stats, "num_elites_kept", ""),
                            getattr(stats, "num_sampled_kept", ""),
                            json.dumps(getattr(stats, "macro_uses", {})),
                            getattr(stats, "budget_exhausted", ""),
                            json.dumps(getattr(stats, "diversity_counts", [])),
                            json.dumps(getattr(stats, "temperature_trace", [])),
                        ])

                        summary_metrics[task_id] = {
                            "elapsed": result["elapsed"],
                            "candidates": getattr(stats, "total_candidates", 0),
                            "budget_exhausted": getattr(stats, "budget_exhausted", False),
                            "diversity_counts": getattr(stats, "diversity_counts", []),
                            "temperature_trace": getattr(stats, "temperature_trace", []),
                        }

                    if "train" in result and len(result["train"]) == len(raw[task_id]["test"]):
                        try:
                            correct = all(
                                eq_grid(res["attempt_1"], pair.output)
                                for res, pair in zip(result["submission"], result["train"])
                            )
                        except Exception:
                            correct = False
                        if correct:
                            print("   -> Correct on training! Added to memory.")
                            solutions_db[result["train_hash"]] = [res["attempt_1"] for res in result["submission"]]
                    else:
                        print("   -> Incorrect. Logging for DSL review.")
                        dummy = type("Spec", (), {"train": result["train"], "test": []})
                        log_missing(task_id, dummy)

                programs = result.get("programs", [])
                train_examples = result.get("train", [])
                solved_program_ops: List[str] | None = None
                if not args.no_library and not result.get("from_mem") and stats:
                    for macro_name, count in getattr(stats, "macro_uses", {}).items():
                        if macro_name.startswith("lib::"):
                            lib_id = macro_name.split("::", 1)[1]
                            for _ in range(max(1, int(count))):
                                library.record_use(lib_id)
                if not args.no_library and not result.get("from_mem"):
                    for program in programs:
                        try:
                            primary = sum(1 for pair in train_examples if eq_grid(program.apply(pair.input), pair.output))
                        except Exception:
                            primary = 0
                        if train_examples and primary == len(train_examples):
                            lib_id = library.register_program(program, train_examples)
                            growth_mode = solver_options.get("library_growth_mode", "none")
                            if growth_mode != "none":
                                library.grow_from_program(
                                    program,
                                    train_examples,
                                    mode=growth_mode,
                                )
                            solved_program_ops = [op.name for op in program.ops]
                            success_ratio = primary / max(1, len(train_examples))
                            for op in program.ops:
                                if op.name.startswith("lib::"):
                                    library.touch_success(
                                        op.name.split("::", 1)[1],
                                        success_ratio,
                                        getattr(stats, "best_secondary", 0.0),
                                    )
                            break
                    library_items = list(library.iter_items())
                    solver_options["library_items"] = library_items

                if solved_program_ops is None and train_examples:
                    for program in programs:
                        try:
                            primary = sum(
                                1 for pair in train_examples if eq_grid(program.apply(pair.input), pair.output)
                            )
                        except Exception:
                            continue
                        if primary == len(train_examples):
                            solved_program_ops = [op.name for op in program.ops]
                            break

                if solved_program_ops and features.get("families"):
                    for family in features["families"]:
                        bucket = task_family_stats.setdefault(family, {"ops": {}, "macros": {}, "confidence": 0.0})
                        for op_name in solved_program_ops:
                            target = bucket["macros" if op_name.startswith("lib::") else "ops"]
                            target[op_name] = target.get(op_name, 0) + 1
                        bucket["confidence"] = min(1.0, bucket.get("confidence", 0.0) + 0.1)

    memory_payload["solutions"] = solutions_db
    memory_payload["rule_confidence"] = rule_confidence
    memory_payload["task_family_stats"] = task_family_stats
    save_memory_db(memory_payload)
    if not args.no_library:
        library.save()
    Path(args.outfile).write_text(json.dumps(submission, indent=2))
    print("\nSubmissions saved to", args.outfile)
    print(f"Per-task stats saved to {stats_csv_path}")
    if args.export_nl:
        nl_path = Path(args.outfile).with_suffix(".nl.txt")
        lines: List[str] = []
        for task_id in sorted(task_nl):
            lines.append(f"# {task_id}")
            programs = task_nl[task_id]
            if not programs:
                lines.append("(no programs exported)")
            else:
                for idx, desc in enumerate(programs, start=1):
                    lines.append(f"Program {idx}:")
                    lines.append(desc)
                    lines.append("")
            lines.append("")
        nl_path.write_text("\n".join(lines).strip() + "\n")
        print(f"Natural-language programs saved to {nl_path}")
    summary_path = Path(args.outfile).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_metrics, indent=2))
    print(f"Summary metrics saved to {summary_path}")


__all__ = ["main", "solve_single_task"]
