# Environment adaptation and trajectory export

This release selects the existing live MCP adapter and Harbor exporter from
`PocketWorker-release-v1.0-clean.zip`. The archive directory `clawmark-eval/`
was a legacy name for a **Claw-Eval** adapter, not the separate ClawMark
benchmark. The public path is `adapters/claw_eval/`. Source hashes and release
changes are recorded in [the integration manifest](integration_source_manifest.json).

## Versions and installation

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements-adapters.txt -r requirements-export.txt
```

Conversion uses PyYAML 6.0.3 and Pydantic 2.14.1. Execution/export targets Harbor
0.24.0; export uses datasets 5.1.0. The generated sidecar installs its separately
pinned [direct dependencies](../adapters/claw_eval/runtime/mcp_bridge/requirements.txt),
including FastMCP 2.14.5. This is not a complete transitive lock or immutable
container-image lock. No editable checkout, sibling-repository symlink or cluster
path is required.

Claw-Eval tasks, fixtures and services are supplied by an **external checkout**.
Conversion was checked against upstream commit
[`5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7`](https://github.com/claw-eval/claw-eval/commit/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7).
To use that version:

```bash
git clone https://github.com/claw-eval/claw-eval.git /path/to/claw-eval
git -C /path/to/claw-eval checkout 5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7
```

Only convert trusted source: the converter imports task graders as Python code.
Its compatibility checks are not an execution sandbox.

## Supported task path

The release supports a narrow path: exactly one Python service, text input/output,
declared local HTTP tool endpoints, and a `PinbenchAdaptedGrader` subclass using
the unchanged grading implementation. Service-state actions, media-dependent
graders, custom grading helpers and multi-service tasks are rejected. Generic
LLM judges, interactive user simulation and all-task benchmark execution are not
part of this release. The task generator's custom graders use a different path
and are not automatically compatible with this adapter.

These real upstream tasks passed conversion and Harbor configuration validation:

- `T087_pinbench_market_news_brief`
- `T090_pinbench_config_change_plan`
- `T095_pinbench_apm_market_research`

```bash
python adapters/claw_eval/run_adapter.py \
  --claw-eval-root /path/to/claw-eval \
  --task-ids T087_pinbench_market_news_brief \
  --output-dir outputs/claw-eval
```

`CLAW_EVAL_ROOT` can replace the root flag. Existing task directories are never
overwritten. `--dry-run` checks source/grader compatibility only; it does not
build payloads or containers. Conversion failures exit nonzero and do not leave
partially generated task directories.

The generated task contains:

- `instruction.md` and `task.toml` for Harbor;
- `environment/Dockerfile` for the agent and verifier;
- `environment/docker-compose.yaml` and `mcp-bridge/`, containing only the
  selected service, shared helpers and referenced fixtures from the checkout;
- `tests/`, containing the selected grader, minimal grading API and verifier.

No fabricated oracle solution, agent-authored trace template or answer-bearing
reference solution is emitted. Keep generated tasks under ignored `outputs/`.

## Running and grading

On a host with Docker Compose, configure an MCP-capable Harbor agent and its
model credentials. After converting the task above, set `HARBOR_MODEL` to a
model supported by that agent and run:

```bash
harbor run --path outputs/claw-eval/T087_pinbench_market_news_brief \
  --agent claude-code --model "$HARBOR_MODEL" \
  --env docker --n-concurrent 1 --job-name market-brief

python scripts/export_rollouts.py --jobs-dir jobs/market-brief \
  --episodes last --output outputs/market-brief.jsonl
```

The generated composition expects Harbor's Docker backend and shared verifier
mode. Other backends and separate-verifier mode are not validated.

A fresh sidecar loads fixtures per trial and exposes `claweval-tools` over MCP.
The bridge records forwarded tool name, request, actual HTTP status and response.
The verifier fetches these records from the bridge's read-only `/dispatches`
endpoint and reads the last assistant answer from Harbor's
`/logs/agent/trajectory.json`. It never uses a workspace `trace.jsonl` as scoring
evidence. The selected task grader is copied unchanged. The vendored composite
score is `safety * (0.80 * completion + 0.20 * robustness)`.

`/logs/verifier/reward.json` contains numeric `reward`, `completion`, `robustness`
and `safety` fields; `scores.json` holds the grading details. Missing/malformed
evidence fails verification and removes any stale reward file. An HTTP error from
a tool retains its real status for grading; it is not silently changed into a
successful dispatch.

Important boundaries for interpreting results:

- The bridge records forwarded calls. MCP protocol errors rejected before
  dispatch are outside this record, and direct service calls are not recorded.
- Grading uses the final text and dispatches, not the full conversation or service
  audit. Efficiency metadata reconstructed from one final message is not the
  true dialogue turn count. Only the supported grader's weighted score is used.
- Error injection is disabled (`ERROR_RATE=0`). This is an adapted execution
  protocol; results are not automatically comparable to an official benchmark run.
- The sidecar has no authentication and its records live in process memory.
  Use a fresh isolated task network per trial; no host ports are published by the
  generated Compose file. A restarted sidecar does not preserve episode evidence.
- The shared container, accessible service endpoints and agent log mount are not
  hardened against an adversarial agent. This release does not claim tamper-proof
  evaluation or general-purpose sandbox security.

## Export recorded Harbor trials

```bash
python scripts/export_rollouts.py --jobs-dir /path/to/jobs --dry-run
python scripts/export_rollouts.py --jobs-dir /path/to/jobs \
  --episodes last --min-reward 0 --output outputs/rollouts.jsonl
```

The wrapper delegates conversion to Harbor and writes local JSONL only; it does
not upload datasets or invoke models. It requires `result.json` metadata and a
main-agent `agent/trajectory.json` from a Harbor-recognized ATIF-capable agent.
It reports unsupported agents and missing trajectories as skips, and treats
malformed metadata/ATIF or conversion errors as failures. No output is written
on failures unless `--allow-partial` is explicitly given. Existing output requires
`--overwrite`; successful output publication is atomic. Exit codes are 0 for a
successful export/dry-run, 1 for no rows or invalid output/input location, and 2
for refused partial export or invalid CLI arguments.

Rows contain `conversations`, `result`, task/run/agent/model metadata and, by
default, the instruction. Harbor's `--episodes all` means conversation prefixes
ending at each agent step, **not independent complete trajectories**; use `last`
for one final prefix per trial. Subagent export is deliberately disabled and
multimodal input is rejected by Harbor's text exporter. This JSONL is an
interchange format, not a claim that it directly matches every SFT/RL trainer.

`--min-reward` retains only finite numeric rewards above the threshold, including
zero at a threshold of zero. It drops exception-valued/missing/nonfinite results.
Without this flag, failed trials with available trajectories may remain in the
export. `--filter success|failure` uses Harbor's own trial-success rule; use the
numeric threshold when you need an explicit score cut. Review exported content
and rights before redistributing your own trajectories; the exporter is not a
content/redaction filter.

## Development validation and release checks

The complete development workflow has already been tested locally. This staged
release contains selected components, rather than all runtime files,
configurations and experiment materials from that tested development setup.
The following checks were repeated specifically on the extracted public code:

- 19 regression/integration tests passed with optional dependencies and the
  pinned upstream checkout: core utilities, real task conversion, valid Harbor
  task configurations, MCP schema/status recording, verifier failure handling,
  zero-reward export, partial-export refusal and overwrite protection.
- A scripted diagnostic launched the actual upstream RSS service and bridge on
  localhost, made two real MCP-over-HTTP tool calls, graded the recorded evidence
  and a diagnostic ATIF answer, and exported the result through Harbor to JSONL.
  This tested the transport/grading/export chain; it was not a model rollout or
  an evaluation of task-solving ability.
- Dependency consistency, Python syntax and shell syntax were checked.

This cleanup did not repeat the complete Docker/Compose and model-driven rollout
on the release-preparation host, where Docker was unavailable. That limits what
was rechecked for this extracted snapshot; it does not mean the complete
development workflow was untested. Reproducing that full workflow from the public
repository will require the additional files and configurations on the roadmap.

To rerun the tests (the bridge dependency is only needed for its optional tests):

```bash
python -m pip install -r adapters/claw_eval/runtime/mcp_bridge/requirements.txt
CLAW_EVAL_ROOT=/path/to/claw-eval python -m unittest discover -s tests -v
```

## ClawMark filesystem/email path

The separate archive directory `clawmark/` is now released selectively under
[`adapters/clawmark/`](../adapters/clawmark/README.md). It supports filesystem/email
stage setup and weighted verification through Harbor multi-step tasks. External
task data and assets remain in a user-supplied, pinned ClawMark checkout.

Its vendored runtime retains CC BY-NC 4.0 with attribution; it is not covered by
the project's MIT license. The ClawMark guide records the dependencies, converted
real tasks and scoring adaptations. The existing JSONL wrapper expects a
trial-level main-agent ATIF file and does not stitch multi-step ClawMark episodes.
