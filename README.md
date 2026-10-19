# CoworkEnv

CoworkEnv is a project for building executable work environments for agent data
construction, trajectory collection and training.

**This staged release contains selected tools from our existing codebase:** task
construction and quality checks, a Claw-Eval → Harbor adapter for single-service
Pinbench text/tool tasks, and local Harbor trajectory export. The full environment
and training stack remain on the roadmap. No benchmark dataset or model is bundled.

The complete development workflow has been tested locally. This repository
publishes selected components of that workflow; not all files and configurations
needed to reproduce the full development setup are included in this release.

## Available now

- **Task generation:** turn a seed specification into a task definition, initial
  service fixtures and a deterministic grader.
- **Tool catalogs:** 44 tool definitions across 12 service interfaces, including
  email, calendar, contacts, CRM, helpdesk, finance and scheduling.
- **Numeric leakage checks:** look for declared ground-truth values in task
  instructions and recognized text fixtures.
- **Task diversity reports:** flag lexical near-duplicates and concentrated
  template families within a task pool.
- **Teacher difficulty summaries:** aggregate existing rollout completion scores
  and identify tasks with high, low or intermediate observed performance.
- **Environment adapter:** convert selected external Claw-Eval tasks into Harbor
  tasks with a live MCP/service sidecar, bridge-recorded tool evidence and grading.
- **Trajectory export:** convert existing Harbor ATIF trials to local JSONL,
  preserving zero rewards and reporting skipped or invalid trials.

These are existing project components with release cleanup. Offline construction,
conversion and export need no model endpoint, API key, GPU or cluster access.
Running an agent additionally requires Docker and the chosen model's credentials.

## Install

Use Python 3.10 or newer. This release was checked with Python 3.12.

```bash
git clone https://github.com/HKUST-KnowComp/CoworkEnv.git
cd CoworkEnv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run scripts directly from the repository root. The optional adapter and Harbor
integration require **Python 3.12** and their own dependencies:

```bash
python -m pip install -r requirements-adapters.txt -r requirements-export.txt
```

The service-side dependencies are installed inside the generated sidecar image.
See [the integration guide](docs/integrations.md) for scope, versions and validation.

## Use the tools with your data

The paths below are placeholders for your own specifications, task directories
and recorded results. This release does not include a task dataset or rollout
archive.

### Generate a task definition

```bash
python data_construction/d1_generator.py \
  --spec /path/to/seed_spec.json \
  --out outputs/constructed
```

The JSON specification follows `SeedSpec` in
[`d1_generator.py`](data_construction/d1_generator.py). Required fields are
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
release. The adapter below supports a specific Pinbench grader family; it does
**not** execute all graders emitted by this generator. The catalogs describe interfaces;
they do not implement those services. Keep `grader.py` on the verifier side.

The prompt leakage gate only checks supplied strings. Strings containing an
ASCII capital, digit or `@` block generation on a match; other matches produce
warnings. Fixtures are not scanned by this gate. Review task goals, fixtures,
evidence availability and grader predicates before using generated tasks.

### Scan known numeric answers

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

### Check a task pool for near-duplicates

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

### Summarize recorded teacher rollouts

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

## Execute and export an existing task

Use your own trusted Claw-Eval checkout, including its service code and fixtures:

```bash
python adapters/claw_eval/run_adapter.py \
  --claw-eval-root /path/to/claw-eval \
  --task-ids T087_pinbench_market_news_brief \
  --output-dir outputs/claw-eval

# Requires Docker Compose, an MCP-capable Harbor agent and its model credentials.
harbor run --path outputs/claw-eval/T087_pinbench_market_news_brief \
  --agent claude-code --model "$HARBOR_MODEL" \
  --env docker --n-concurrent 1 --job-name market-brief

python scripts/export_rollouts.py --jobs-dir jobs/market-brief \
  --episodes last --output outputs/market-brief.jsonl
```

Set `HARBOR_MODEL` to a model supported by your configured Harbor agent. The
adapter retains the selected task's grader, records HTTP tool dispatches, and
reads the harness's final ATIF answer. A fresh service sidecar provides the
initial fixture state for each trial. This is a task adaptation, not a claim of
official benchmark score equivalence. [Supported tasks, grading boundaries and
export options](docs/integrations.md) are documented separately.

## Repository layout

```text
data_construction/
  d1_generator.py
  d1_catalog/                  # Service tool/endpoint schemas
  leakage_scan.py
  internal_dedup.py
  teacher_difficulty_filter.py
adapters/claw_eval/            # Selected task conversion, MCP bridge, verifier
scripts/export_rollouts.py     # Local Harbor ATIF → JSONL
tests/                        # Regression tests; optional upstream checks
docs/source_manifest.json     # Original file hashes and release changes
docs/integration_source_manifest.json
docs/integrations.md           # Setup, scope and validation limits
third_party/claw-eval.LICENSE # Upstream notice for adapted tool definitions
```

Run checks with `python -m unittest discover -s tests -v`. Optional integration
tests skip if their dependencies or external checkout are absent. See the guide
for the checks repeated on this extracted release and its reproduction scope.

## Roadmap

The following are planned release areas, not currently supported public APIs:

- [ ] Publish the remaining runtime dependencies and complete run configurations
  needed to reproduce the full locally tested workflow.
- [ ] Environment execution: initialization/reset, service management, sandbox
  integration and artifact collection beyond the selected single-service path.
- [ ] Broader trajectory support: multimodal/subagent export and teacher
  rollout collection workflows.
- [ ] Task construction workflow: seed specifications, golden-state checks,
  grader validation and task manifests with source/split information.
- [ ] Evaluation adapters, preserving each benchmark's scoring protocol and
  third-party terms.
- [ ] ClawMark adapter and examples, following a separate backend/dependency
  review and release under the applicable upstream terms. ClawMark is deferred
  from this release; its upstream code/data are not included.
- [ ] SFT conversion and training configurations.
- [ ] RL rollout integration, reproducible training configurations and the
  required upstream patches/version pins.
- [ ] Process-reward evaluation, rubric review and calibration tools, with
  implementation, test and training-validation status reported separately.
- [ ] Dataset/model release indexes and reproducibility records, where available
  and redistributable.

## License and provenance

Project code is released under the [MIT License](LICENSE). Tool-schema catalogs
and a small vendored grader/model subset come from Claw-Eval; its notice is retained in
[third_party/claw-eval.LICENSE](third_party/claw-eval.LICENSE). See
[NOTICE](NOTICE) for attribution and the boundaries of this release. No upstream
benchmark dataset, service implementation, Slime/Megatron code, model checkpoint
or training log is bundled here. Conversion copies selected service/fixture files
from the external checkout into ignored local output, not into this repository.
