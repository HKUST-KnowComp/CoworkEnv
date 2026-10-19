# ClawMark → Harbor adapter

This directory releases the filesystem/email path from the project's existing
ClawMark adapter. It converts trusted external tasks to Harbor multi-step tasks,
retaining stage notifications, environment updates and weighted checkers.
ClawMark task data, assets, oracle solutions and the experimental Notion/Sheets
backends from the development archive are not bundled.

## Use

Use Python 3.12. Conversion and the included runtime managers use the standard
library. To run tasks, install the repository's `requirements-export.txt`
(Harbor 0.24.0), Docker with Compose, and credentials for your chosen Harbor agent.
The generated image installs `sqlite3`, `curl` and CA certificates; email tasks
also use `greenmail/standalone:2.1.8`. No Slime, Megatron, private package index or
cluster checkout is needed for this path.

Supply a trusted external ClawMark checkout. Conversion checks used commit
[`d1b641b3171e584e69a3763c269069f32a13b574`](https://github.com/evolvent-ai/ClawMark/commit/d1b641b3171e584e69a3763c269069f32a13b574):

```bash
git clone https://github.com/evolvent-ai/ClawMark.git /path/to/ClawMark
git -C /path/to/ClawMark checkout d1b641b3171e584e69a3763c269069f32a13b574

# Run from the CoworkEnv repository root.
python adapters/clawmark/run_adapter.py \
  --clawmark-root /path/to/ClawMark \
  --task-ids journalist/task5 \
  --output-dir outputs/clawmark

harbor run --path outputs/clawmark/journalist_task5 \
  --agent claude-code --model "$HARBOR_MODEL" \
  --env docker --resume-trajectory --n-concurrent 1 --job-name journalist-task5
```

Set `HARBOR_MODEL` to a model supported by the configured agent. `CLAWMARK_ROOT`
can replace the checkout flag. `--dry-run` checks the source structure without
building output or launching containers. Conversion imports trusted `task.py`
modules as Python, but does not execute their stage functions on the host.
Existing output is never overwritten; use a fresh directory after code changes.
The run command preserves the agent's session across stages. This requires an
agent with Harbor resume support; omitting `--resume-trajectory` starts a fresh
agent session per step even though the workspace and services persist.

## Execution and scoring

- Only `filesystem` and `email` environments are supported. Historical snapshots
  and Notion, Sheets or calendar backends are rejected. Compatibility checks are
  conservative, not a guarantee for every task or future upstream revision.
- Each rubric stage becomes a Harbor step. Its setup runs the source stage
  function inside the task container, preserving workspace/email state between
  steps. Both `assets/` and later `inject/` resources accompany the runtime.
  Notifications must be literal strings. Dynamic notifications are rejected.
- Email readiness checks wait for actual SMTP/IMAP greetings. Setup shell errors
  and failed email delivery abort the step instead of silently continuing.
  Seeding source is removed from the agent workspace before the agent starts.
- The verifier runs the original stage checkers and computes
  `sum(passed weights) / sum(weights)`. Checker exceptions count as failed checks,
  matching upstream's convention; invalid task/runtime setup fails verification
  without writing a zero reward. Details go to `checker_results.json`.
- `--reward-strategy mean` (default) averages Harbor step rewards; `final` uses
  the last step's reward. A source `final` rubric becomes an additional Harbor
  agent/verifier step. These are explicit adaptation choices, so report the
  protocol alongside scores rather than claiming official benchmark equivalence.
- No placeholder oracle is generated. Optional `--oracle-root` copies your real
  scripts from `<source_id>/<stage>.sh` when present.

Use Harbor's Docker backend and shared verifier mode. `LocalSandbox` is only an
in-container transport; it does not isolate host processes. The generated
GreenMail service has no authentication and publishes no host ports. This path
assumes a fresh isolated task network and is not an adversarially hardened grader.
Container tags and OS packages are not immutable locks.

Harbor keeps multi-step results and agent records. The repository's JSONL exporter
currently expects a trial-level `agent/trajectory.json`; it does not stitch
per-step trajectories into one complete ClawMark episode.

## Checks and provenance

The complete development workflow was tested locally. For this extracted release,
`eda/task1`, `journalist/task5` and `journalist/task7` passed conversion and Harbor
configuration validation. Release tests cover resource preservation, failure
handling, scoring and source-path constraints. The preparation host did not repeat
the complete container build/model rollout; that does not change the original
workflow's local test status.

```bash
CLAWMARK_ROOT=/path/to/ClawMark python -m unittest discover -s tests -p 'test_clawmark.py' -v
```

Project adapter code is MIT. The selected `runtime/clawmark_vendor/` subtree
retains **CC BY-NC 4.0**, not MIT; see its [license](runtime/clawmark_vendor/LICENSE)
and [source notice](runtime/clawmark_vendor/NOTICE.md). This adapter is not endorsed
by ClawMark. File origins and cleanup changes are recorded in
[`docs/clawmark_source_manifest.json`](../../docs/clawmark_source_manifest.json).
