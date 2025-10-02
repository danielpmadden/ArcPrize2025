# ArcPrize2025 — REAP Solver

REAP (Recursive Emergent Abstraction Program) is a heuristic search
solver for the [Abstraction and Reasoning Challenge (ARC)](https://arcchallenge.com/).
This repository packages the "REAP" solver used in the Arc Prize 2025
competition. The solver explores a domain-specific language (DSL) of grid
transformations, performs beam search over candidate programs, and leverages a
local memory of past solutions to improve performance on related tasks.

## Features

- **Domain-Specific Language**: Rotate, flip, color-map, pad/crop, tiling,
  scaling, repetition, object-centric transforms, and more.
- **Beam Search Synthesizer**: Configurable breadth-first beam search with
  adaptive depth/beam sizing based on task heuristics.
- **Task Classification Heuristics**: Detects shape and object statistics to
  prioritize relevant operators.
- **Local Task Memory**: Stores successful solutions keyed by training hash for
  instant reuse on identical tasks.
- **Failure Logging**: Records unsolved tasks for later DSL extension.
- **Parallel Execution**: Solves multiple tasks concurrently using the
  `ProcessPoolExecutor` and a user-defined worker count.
- **Detailed Instrumentation**: Emits per-task statistics (beam size, depth,
  candidate counts) to a CSV sidecar file.

## Requirements

The solver depends only on the Python standard library. It has been tested with
Python 3.10+, but any modern Python 3 interpreter with `dataclasses` support
should work.

## Usage

```
python REAP.py --infile arc_tasks.json --outfile submission.json \
    --time_per_task 20 --max-workers 4
```

### Command-line arguments

| Flag | Description |
| ---- | ----------- |
| `--infile` | Path to an ARC-style task JSON file (required). |
| `--outfile` | Destination path for the solver predictions. Defaults to `submission.json`. |
| `--time_per_task` | Maximum number of seconds spent per task (floating point). Defaults to `20.0`. |
| `--max-workers` | Number of worker processes to spawn. Use `0` or a negative value to use all CPU cores. Defaults to `1`. |

### Input format

The solver expects the same JSON structure used by the ARC-Kaggle benchmark:

```json
{
  "task_id": {
    "train": [
      {"input": [[...], ...], "output": [[...], ...]},
      ...
    ],
    "test": [
      {"input": [[...], ...]},
      ...
    ]
  },
  ...
}
```

Each grid is a 2D list of integers (colors) in the range `0`–`9`.

### Output files

- `submission.json`: Mapping from task id to a list of predictions. For each
  test example the solver provides `attempt_1` and `attempt_2` grids (both
  2D lists of colors).
- `submission.json.stats.csv`: Per-task instrumentation emitted alongside the
  submission. Columns capture elapsed time, beam-search depth, counts of valid
  programs, best distance, candidate counts, and more.
- `memory_db.json`: Automatically created/updated database of previously solved
  tasks. When a known task is encountered, the solver bypasses search and reuses
  the stored outputs.
- `missing_ops.jsonl`: Log of challenging tasks to revisit. Each line contains a
  JSON object with the task id and grids.

## Customisation Tips

- **Adjust search aggressiveness** by editing the defaults in `SearchConfig`
  inside `REAP.py` (beam size, depth, pruning thresholds, etc.).
- **Extend the DSL** by adding new primitives (e.g., additional object-based
  operations) and registering them with the synthesizer.
- **Disable memory** by deleting `memory_db.json` before running the solver, or
  by modifying the calls to `load_memory_db` / `save_memory_db`.

## Development Workflow

1. Prepare a JSON file containing ARC tasks.
2. Run the solver with a chosen time budget and worker count.
3. Inspect `submission.json.stats.csv` to understand search behavior and
   identify tasks that could benefit from new operators.
4. Review `missing_ops.jsonl` to prioritize DSL improvements.

## License

This repository does not currently include an explicit license. If you plan to
reuse the solver, please contact the repository maintainers.
