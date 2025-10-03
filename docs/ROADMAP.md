# Development Roadmap

This roadmap outlines a sequence of daily engineering tasks for improving the REAP solver. Tasks are grouped from low-effort fixes, through moderate enhancements, to ambitious longer-term investments. Each task is scoped to fit within a single working day.

## Low-effort wins

### Day 1 — Normalize formatting and type hints
- **Why it matters:** Cleaning up obvious style inconsistencies and adding missing type hints reduces cognitive load for future contributors and surfaces latent bugs via static analyzers.
- **Post-task checks:** Run `python -m compileall REAP.py` and a linter (e.g., `ruff` or `flake8`) to ensure no syntax or style regressions.

### Day 2 — Fix multiprocessing executor mismatch
- **Why it matters:** `ProcessPoolExecutor` is imported as `ThreadPoolExecutor`, which silently forces threading. Correcting this restores true parallelism and prevents hard-to-diagnose performance ceilings.
- **Post-task checks:** Execute a short benchmark solving a handful of tasks with `--max-workers > 1` and confirm multiple processes spawn (e.g., via logging or `ps` output).

### Day 3 — Harden grid validation utilities
- **Why it matters:** `valid_grid` currently returns `True` for empty grids and does not strongly validate nested lists. Tightening validation and adding unit tests prevents invalid states from propagating into the search.
- **Post-task checks:** Run newly added unit tests and ensure solver still runs on a sample ARC task without raising assertions.

### Day 4 — Add CLI flag documentation and usage examples
- **Why it matters:** Improved README guidance reduces onboarding friction for new users and avoids misconfigured runs that waste compute.
- **Post-task checks:** Proofread rendered README (e.g., GitHub preview) and verify that every documented flag matches the argparse definitions.

### Day 5 — Introduce lightweight unit test harness
- **Why it matters:** Even a minimal pytest suite covering encoders, DSL primitives, and memory persistence catches regressions early and enables automated CI adoption later.
- **Post-task checks:** Run `pytest` locally and ensure green results; measure execution time to keep the suite under a few seconds.

## Moderate improvements

### Day 6 — Configurable search parameters file
- **Why it matters:** Externalizing beam search knobs (depth, beam width, pruning thresholds) into a JSON/YAML config allows experimentation without code edits, enabling reproducible tuning.
- **Post-task checks:** Verify the solver reads overrides from the config file, falls back to defaults when absent, and logs the effective configuration at startup.

### Day 7 — Structured logging for solver lifecycle
- **Why it matters:** Replacing ad-hoc prints with structured logging (JSON or key-value) improves observability, making it easier to diagnose timeouts, operator usage, and performance regressions.
- **Post-task checks:** Run the solver on a small task batch and confirm logs include timestamps, task IDs, and key metrics; ensure log level can be adjusted via CLI.

### Day 8 — Graceful timeout and cancellation handling
- **Why it matters:** Currently long-running tasks may exceed the per-task budget without clear feedback. Adding watchdog timers and cancelling runaway searches preserves throughput and prevents worker starvation.
- **Post-task checks:** Create a synthetic slow task, ensure it times out according to the configured limit, and verify the solver records the timeout reason without crashing other workers.

### Day 9 — Memory database versioning and pruning
- **Why it matters:** As `memory_db.json` grows, stale entries and schema changes can cause reuse errors. Version metadata and size-limited pruning maintain reliability over time.
- **Post-task checks:** Simulate loading an older DB, confirm automatic migration or warning, and ensure pruning respects configurable retention limits.

### Day 10 — Rich failure analytics report
- **Why it matters:** Enhancing `missing_ops.jsonl` with operator coverage, heuristic scores, and sample visualizations helps prioritize DSL extensions efficiently.
- **Post-task checks:** After running on a mixed task set, inspect the new report for completeness and verify backward compatibility with existing logs.

## Ambitious enhancements

### Day 11 — Web-based dashboard for run monitoring
- **Why it matters:** A lightweight dashboard (e.g., FastAPI + React) displaying task progress, beam statistics, and memory hits enables interactive triage during long experiments.
- **Post-task checks:** Launch the dashboard during a solver run, ensure live updates work, and confirm the solver continues to run headless when the dashboard is disabled.

### Day 12 — DSL auto-tuning via genetic search
- **Why it matters:** Automatically exploring combinations of DSL primitives and parameter tweaks can uncover high-performing configurations beyond manual tuning, improving solve rates.
- **Post-task checks:** Execute a pilot tuning session on a subset of tasks, monitor convergence metrics, and ensure the solver can persist the best-found configuration.

### Day 13 — Integration with public ARC benchmark datasets
- **Why it matters:** Streamlining downloads and evaluations against ARC Kaggle datasets (and future Arc Prize benchmarks) promotes reproducible comparisons and community contributions.
- **Post-task checks:** Run an end-to-end evaluation script that fetches data, invokes the solver, and produces leaderboard-style metrics without manual intervention.

### Day 14 — UI-assisted solution replay and debugging
- **Why it matters:** An interactive tool visualizing candidate program execution and grid transitions accelerates hypothesis testing for new DSL primitives and helps explain solver decisions.
- **Post-task checks:** Load a solved task, step through candidate programs in the UI, and verify controls for pausing, rewinding, and inspecting grid states operate correctly.

### Day 15 — Pluggable neural guidance interface
- **Why it matters:** Allowing optional integration with learned priors (e.g., logits from a neural policy) positions the solver for hybrid neuro-symbolic research and potential competition gains.
- **Post-task checks:** Implement a mock guidance provider, confirm the solver can run both with and without guidance, and benchmark search speed/accuracy deltas.

