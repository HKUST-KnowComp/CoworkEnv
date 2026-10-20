<div align="center">

# CoworkEnv

### Synthesizing Training Environments for Long-Horizon Cowork Agents

**Build work environments. Collect interactions. Train cowork agents.**

[Quick Start](#quick-start) · [Documentation](#documentation) · [Roadmap](#release-and-roadmap) · [License](#license)

</div>

## Overview

This repository contains the implementation of **CoworkEnv: Synthesizing Training
Environments for Long-Horizon Cowork Agents**.

Cowork agents carry out multi-step work across files, tools and services.
Training them requires environments that connect a task's goal to an initial
world state, available actions and verifiable outcomes. CoworkEnv brings together
**task construction, environment execution, trajectory collection and training**
to support this workflow.

### Core capabilities

- **Construct training environments.** Turn seed specifications into task
  definitions, service fixtures and graders, with tool catalogs spanning 12
  service interfaces.
- **Inspect task quality.** Check known answer leakage, compare task diversity
  and summarize difficulty from recorded teacher rollouts.
- **Collect grounded interactions.** Use selected Claw-Eval and ClawMark adapters
  with Harbor to execute tasks, score outcomes and record trajectories.
- **Prepare agents for training.** Export text/tool trajectories, train and merge
  LoRA adapters, and prepare task manifests and GRPO/GDPO-style episode signals
  for RL workflows.

## Quick Start

Use **Python 3.12** and run commands from the repository root.

```bash
git clone https://github.com/HKUST-KnowComp/CoworkEnv.git
cd CoworkEnv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 1. Construct tasks

Provide a seed specification describing the task, initial service state and
completion checks:

```bash
python data_construction/d1_generator.py \
  --spec /path/to/seed_spec.json \
  --out outputs/constructed
```

The output contains a task definition, fixtures and a grader. See the
[task construction guide](docs/data_construction.md) for the specification
format, quality checks and runtime requirements.

### 2. Execute tasks and collect trajectories

Install the optional environment and export dependencies:

```bash
python -m pip install -r requirements-adapters.txt -r requirements-export.txt
```

Use the [Claw-Eval execution guide](docs/integrations.md) or
[ClawMark guide](adapters/clawmark/README.md) to adapt and run tasks from your
upstream checkout. Execution requires Docker and a configured Harbor agent.
The guides describe the supported task families and grading protocols.

Export recorded Harbor main-agent ATIF trials to local JSONL:

```bash
python scripts/export_rollouts.py \
  --jobs-dir /path/to/harbor-jobs --episodes last \
  --output outputs/rollouts.jsonl
```

### 3. Prepare training data

Convert selected text/tool trajectories and validate the resulting messages:

```bash
python -m training.sft.convert_data \
  --input outputs/rollouts.jsonl --output outputs/sft.jsonl
python -m training.sft.validate_data --data outputs/sft.jsonl
```

Continue with the [LoRA training and merging guide](training/README.md#train-a-lora-adapter).
For RL, see [task manifests](training/README.md#rl-task-manifests) and
[episode reward preprocessing](training/README.md#rl-episode-reward-preprocessing).
Model-training dependencies are installed separately from the environment tools.

## Documentation

- **[Task construction](docs/data_construction.md)** — seed specifications,
  fixtures, graders and quality checks.
- **[Claw-Eval integration](docs/integrations.md)** — task adaptation, execution,
  outcome scoring and trajectory export.
- **[ClawMark integration](adapters/clawmark/README.md)** — filesystem/email tasks
  with multiple stages and environment updates.
- **[SFT and RL components](training/README.md)** — data formats, LoRA training,
  weight merging and standalone RL utilities.

## Release and Roadmap

We are releasing CoworkEnv incrementally. The current repository includes the
components described above; RL support currently covers data and reward
preprocessing. The full distributed RL stack, task datasets and model weights
are not bundled. The complete development workflow has been tested locally;
component guides document the scope and checks of this public release.

Future releases may expand environment coverage, data construction and trajectory
tools, and training resources. Scope and timing depend on readiness.

## License

Project-authored code is released under the [MIT License](LICENSE).
Selected Claw-Eval components retain their upstream MIT notice. The vendored
ClawMark runtime retains **[CC BY-NC 4.0](third_party/clawmark.LICENSE)**, including
its noncommercial condition. See [NOTICE](NOTICE) for attribution and licensing
boundaries.
