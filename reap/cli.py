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
from typing import Any, Dict

from .logging_utils import log_missing
from .grid_utils import eq_grid
from .memory import hash_train, load_memory_db, save_memory_db
from .solver import solve_task
from .types import Example


def solve_single_task(task_id: str, spec: dict, memory_db: Dict[str, Any], time_per_task: float) -> Dict[str, Any]:
    """Worker function executed in subprocesses."""

    from time import time as now

    start_time = now()
    parsed = type("Spec", (), {})()
    parsed.train = [Example(input=pair["input"], output=pair["output"]) for pair in spec["train"]]
    parsed.test = [Example(input=item["input"]) for item in spec["test"]]

    train_payload = [{"input": pair.input, "output": pair.output} for pair in parsed.train]
    train_hash = hash_train(train_payload)
    if train_hash in memory_db:
        grids = memory_db[train_hash]
        elapsed = now() - start_time
        return {
            "tid": task_id,
            "submission": [{"attempt_1": grid, "attempt_2": grid} for grid in grids],
            "from_mem": True,
            "elapsed": elapsed,
            "stats": None,
            "train_hash": train_hash,
            "train": parsed.train,
        }

    attempts, stats = solve_task(parsed, time_budget_s=time_per_task)
    elapsed = now() - start_time
    return {
        "tid": task_id,
        "submission": attempts,
        "from_mem": False,
        "elapsed": elapsed,
        "stats": stats,
        "train_hash": train_hash,
        "train": parsed.train,
    }


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the solver."""

    parser = argparse.ArgumentParser("REAP Solver")
    parser.add_argument("--infile", required=True, help="ARC tasks JSON input")
    parser.add_argument("--outfile", default="submission.json", help="Output predictions file")
    parser.add_argument("--time_per_task", type=float, default=20.0, help="Time budget per task (sec)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes (parallelism)")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.infile).read_text())
    memory_db = load_memory_db()
    submission: Dict[str, Any] = {}

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
        ])

        if args.max_workers <= 0:
            args.max_workers = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(solve_single_task, task_id, raw[task_id], memory_db, args.time_per_task): task_id
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

                if result["from_mem"]:
                    print(f"[{index}/{len(raw)}] Task {task_id} solved from memory in {result['elapsed']:.2f}s.")
                    writer.writerow([task_id, result["elapsed"], True, "", "", "", "", "", "", "", ""])
                else:
                    stats = result["stats"]
                    if not stats:
                        print(f"[{index}/{len(raw)}] Task {task_id} failed (no stats).")
                        writer.writerow([task_id, result["elapsed"], False, "", "", "", "", "", "", "", ""])
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
                        ])

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
                            memory_db[result["train_hash"]] = [res["attempt_1"] for res in result["submission"]]
                        else:
                            print("   -> Incorrect. Logging for DSL review.")
                            dummy = type("Spec", (), {"train": result["train"], "test": []})
                            log_missing(task_id, dummy)

    save_memory_db(memory_db)
    Path(args.outfile).write_text(json.dumps(submission, indent=2))
    print("\nSubmissions saved to", args.outfile)
    print(f"Per-task stats saved to {stats_csv_path}")


__all__ = ["main", "solve_single_task"]
