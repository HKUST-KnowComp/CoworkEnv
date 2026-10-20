# Task construction and quality checks

Construct task definitions, initial service fixtures and graders from your own
seed specifications, then inspect task diversity, known answer leakage and
recorded teacher performance.

Run these commands from the repository root after installing `requirements.txt`.
The utilities need no model endpoint, API key, GPU or cluster access. Paths in
this guide refer to your own task specifications and recorded results; a task
dataset and rollout archive are not bundled.

## Generate a task definition

```bash
python data_construction/d1_generator.py \
  --spec /path/to/seed_spec.json \
  --out outputs/constructed
```

The JSON specification follows `SeedSpec` in
[`d1_generator.py`](../data_construction/d1_generator.py). Required fields are
`task_id`, `task_name`, `difficulty`, `category`, `services`, `prompt_body`,
`language`, `fixtures` and `completion_checks`.

- `services` lists catalog names. `fixtures` maps each service to a non-empty
  mapping of JSON filenames to initial state; the first file is its primary
  fixture. Fixture names must be plain filenames, not paths.
- Each completion check has a `name`, `weight` and `kind`. Weights must be finite,
  non-negative and sum to 1. Supported kinds are `tool_called`,
  `tool_arg_contains`, `tool_arg_excludes`, `tool_arg_number_near` and
  `audit_present`; their fields and semantics are documented in `CompletionCheck`.
- Optional `safety_forbidden_tools` configures the grader's safety gate.
  Optional `leakage_strings` declares answer strings to check in the prompt.
- Use a fresh `task_id` or output directory: existing task directories are not
  overwritten.

Output is `outputs/constructed/tasks/<task_id>/`, containing `task.yaml`,
`grader.py` and `fixtures/<service>/*.json`.

**Generation does not execute or validate the task in an environment.** The
generated YAML references `mock_services/<service>/server.py`, and the grader
imports the `claw_eval` API. Compatible service servers remain external to this
release. The [Claw-Eval adapter](integrations.md) supports a specific Pinbench grader family; it does
**not** execute all graders emitted by this generator. The catalogs describe interfaces;
they do not implement those services. Keep `grader.py` on the verifier side.

The prompt leakage gate only checks supplied strings. Strings containing an
ASCII capital, digit or `@` block generation on a match; other matches produce
warnings. Fixtures are not scanned by this gate. Review task goals, fixtures,
evidence availability and grader predicates before using generated tasks.

## Scan known numeric answers

```bash
python data_construction/leakage_scan.py \
  --task-dir /path/to/task \
  --gt 69.6 49.7
```

The scanner reads `instruction.md`, the prompt and `reference_solution` from
`tests/task.yaml` (or root `task.yaml`), and `.json`, `.csv`, `.txt`, `.md`,
`.yaml` and `.yml` files under `exec/` and `fixtures/`.

Exit codes: **0** means no supplied values matched in scanned text, **2** means a
match was found, and **1** means the input could not be scanned completely.
The CLI ignores integer values 0 and 100 by default; pass `--ignore-ints` with
no values to disable that exception. Matching is numeric and approximate, with
tolerance `max(1e-6, abs(gt) * 1e-4)`; binary files, semantic answers, negative
numbers and scientific notation require separate review. A clean result is not
a general guarantee against answer leakage.

## Check a task pool for near-duplicates

```bash
python data_construction/internal_dedup.py \
  --pool-dir /path/to/task_directories \
  --output outputs/internal_dedup.json
```

Each child directory is a task. Prompts are loaded from `tests/task.yaml`,
`task.yaml` or `instruction.md`. `--candidate /path/to/new_task` adds a task to
the comparison; use a distinct directory name so it does not replace a pool
entry. `--dataset-dir /path/to/tasks --tasks id1 id2` selects explicit task IDs.

The report flags 4-gram overlap greater than 0.50, term-frequency cosine at least
0.85 and template-family counts greater than 3. Cosine is **lexical**, not an
embedding score. Template families are inferred from task names; word-based
tokenization is limited for languages without whitespace segmentation. This is
an all-pairs report, with quadratic pair count. Prompts that cannot be extracted from these layouts are omitted;
file-access errors can still stop the command. It does not remove tasks, fail the command on a flagged pair, or check
contamination against a held-out benchmark.

## Summarize recorded teacher rollouts

```bash
python data_construction/teacher_difficulty_filter.py \
  --jobs-dir /path/to/teacher_jobs \
  --output outputs/teacher_difficulty.json
```

The reader finds `<task_id>__<trial_suffix>/result.json` recursively. A suffix is
alphanumeric. It reads `verifier_result.rewards.completion`, falling back to
`reward`; flat verifier and top-level completion/reward schemas are also
accepted. Scores must be finite and in `[0, 1]`. Missing or invalid scores are
omitted, not treated as task failures. `k` counts accepted scores, not all
attempted runs. Repeated `--jobs-dir` inputs should cover disjoint trials from
the same teacher and evaluation setup to avoid double counting or mixing
incomparable scores.

The report contains per-task `k`, mean, range and variance. Historical labels
are retained: `TRIVIAL` for mean ≥ 0.999, `UNSOLVABLE` for mean ≤ 0.001 and `KEEP`
otherwise. These describe observed scores; they do not prove solvability or
learning value. Inspect failures and sample counts before deciding to filter.
This command reads results only; it does not launch teacher rollouts.
