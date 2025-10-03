# REAP Upgrade Overview

This repository extends the original REAP solver with several capabilities that
mirror the playbook used by top-performing ARC solvers.

## New CLI Flags

- `--no-library` – disable loading/saving dynamic library macros.
- `--no-explore-bias` – fall back to deterministic beam pruning.
- `--softmax-T`, `--exploration_softmax_temp`, `--elite`, `--alpha`, `--beta`
  – control exploration-biased sampling.
- `--revisions` / `--no-revisions` – toggle feedback-driven revision loops.
- `--parents-per-revision`, `--max-individual-revisions`,
  `--max-pooled-revisions` – fine-tune revision generation volume.
- `--use-pooled-revisions` / `--no-pooled-revisions` – explicitly enable or
  disable pooled macro recombination.
- `--deduplicate-equivalence` / `--no-deduplicate-equivalence` – control
  behaviour-equivalence collapsing in the beam.
- `--enable-rule-confidence` / `--disable-rule-confidence` – toggle the
  operator confidence tracker.
- `--task-family-bias` / `--no-task-family-bias` – apply task family
  classification to reorder DSL primitives.
- `--abstract-library` – legacy alias for flat macro abstraction.
- `--library_growth_mode` – choose between `none`, `flat`, and `hierarchical`
  macro discovery modes.
- `--feedback-diff` / `--no-feedback-diff` – attach ASCII diffs to revision
  heuristics for richer feedback.
- `--export_nl` – emit natural-language program descriptions alongside JSON
  submissions.
- `--meta_bias_strength` – scale the influence of task-family meta-learning on
  operation ordering.
- `--time-budget` – override the per-task wall-clock limit (`--time_per_task`
  remains available for backward compatibility).
- `--exploration-temp` – multiply the softmax temperature to trade off
  exploration diversity versus exploitation.
- `--min-diverse` – guarantee a minimum number of behaviourally distinct
  candidates survive each depth.
- `--dedup-mode` – select structural, behavioural, or combined deduplication.
- `--neural-bias` – blend optional neural guidance scores into symbolic
  fitness (0 disables the hook).
- `--library-priority` – weight library macros during operator ordering and
  sampling.

## Dynamic Program Library

Successful programs are now persisted to `library_db.json`. When the solver
starts it loads the library and injects macros as operators named
`lib::<identifier>`. These macros are tracked in search statistics and can be
reused across tasks without changing the JSON input/output formats. When
`--library_growth_mode` is set to `flat` or `hierarchical`, the solver also
mines frequent op sequences from recent solutions and registers them as
higher-priority abstract macros with confidence scores that bias enumeration.

Macro metadata now records usage counts and confidence updates across tasks.
The `--library-priority` knob favours library macros during beam expansion, and
per-task macro usage is fed back into the on-disk library for future runs.

Hierarchical growth allows macros to be composed of previously learned macros
when they recur across tasks, enabling DreamCoder-style compression.

Natural-language serialisation is available via `--export_nl`. Each synthesised
program is represented as a short bullet list describing the symbolic steps.
These summaries round-trip back into executable programs for hybrid workflows.

Task-family statistics gathered across runs (symmetry, repetition, colour
shift, and object motion) are persisted in `memory_db.json`. When
`--meta_bias_strength` is non-zero, the search initialisation boosts DSL
primitives and library macros that previously succeeded on the detected
families, improving cross-task transfer.

The optional `reap.neural_guidance.NeuralGuidance` hook allows external models
to supply heuristic scores blended via `--neural-bias`, keeping the default
behaviour deterministic when no model is provided.

## Search Telemetry

Beam search now tracks richer metrics, including primary/secondary fitness,
deduplication hits, revision counts, macro usage, diversity counts, and
temperature traces. Per-task CSV reports include these new columns, and a
companion `<outfile>.summary.json` file captures elapsed time, candidate volume,
and budget status for downstream analysis.

## DSL Additions

Object-centric primitives such as `complete_symmetry`, `copy_dominant_object`,
`align_to_edge`, `distribute_evenly`, `colorize_by_size`, and more are available
in `FUNCTION_REGISTRY`. A lightweight condition-action rule engine suggests
high-level macro candidates during search.

Consult the module docstrings for detailed explanations of each component.
