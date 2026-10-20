# Selected SFT and RL components

This directory publishes selected training code from the existing development
workflow. It includes text/tool SFT conversion, validation, LoRA training and
merging, plus standalone RL task-manifest and episode-reward preprocessing.
It does not include the Slime rollout adapter, a policy optimizer, distributed
launchers, model weights or training datasets.

The complete development workflow has been tested locally. This extracted
release has its own checks described below; it is not the configuration needed
to reproduce every historical experiment.

## Installation

Run commands from the repository root. Data conversion, validation and RL
preprocessing use the standard library and `requirements.txt` only.

For training and merging, use a separate **Python 3.12** environment:

```bash
python3.12 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install -r requirements.txt -r requirements-sft.txt
python -m pip check
```

The optional stack pins PyTorch, torchvision, Transformers, PEFT, Accelerate and
datasets. It does not require Harbor, Slime, Megatron, DeepSpeed or an inference
server. Use a PyTorch installation compatible with your CUDA runtime for GPU
training. The entrypoint uses one process; distributed cluster recipes are not
part of this release. Select the intended device through `CUDA_VISIBLE_DEVICES`.

## Convert and validate your trajectories

```bash
python -m training.sft.convert_data \
  --input outputs/rollouts.jsonl --output outputs/sft.jsonl
python -m training.sft.validate_data --data outputs/sft.jsonl
```

Each input line can contain `messages`, Harbor-exported `conversations`, or a raw
ATIF object (`steps`, optionally nested under `trajectory`). Output contains
`messages` and, when supplied, `task_id`, `task_name`, `tools` and a numeric
`reward`. Rewards are metadata; the SFT trainer does not filter or weight by them.
Select your accepted trajectories before training.

- Raw ATIF conversion preserves reasoning, tool names, call IDs, arguments and
  text observations. Structured calls require exactly one matching result before
  the next conversation turn. Missing responses are rejected, never fabricated.
- Harbor 0.24's exporter renders tool calls and reasoning into text and tool
  observations as user messages. That representation is retained as exported;
  this converter does not recover structured call IDs or validate their pairing.
  Use raw ATIF when you need structured tool messages. These representations are
  different training formats; choose one consistently for an experiment.
- Existing system messages are retained; no generic system prompt is inserted.
  Tool-name rewriting is opt-in (`--strip-mcp-prefix`, raw ATIF only).
- Images/video blocks are rejected. Tool content is not truncated. Missing
  reward stays missing, zero stays zero, and invalid reward strings fail.
- Output is written atomically after all records convert successfully. Existing
  files require `--overwrite`; input and output must be distinct.

The validator checks message structure and reports counts and a dataset SHA-256.
It does not establish task correctness, data quality or train/test separation.
It only validates structured tool pairs that are present in the input.

## Train a LoRA adapter

The following paths refer to your own model and data; no example dataset or
checkpoint is downloaded by these commands.

```bash
python -m training.sft.train_lora \
  --model /path/to/base-model --model-kind qwen3-vl --local-files-only \
  --data outputs/sft.jsonl --output outputs/sft-run \
  --dtype bfloat16 --max-length 4096 \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 \
  --learning-rate 2e-4 --lora-rank 32 --lora-alpha 64
```

`qwen3-vl` loads `Qwen3VLForConditionalGeneration` with `AutoProcessor` for
**text/tool data only**. `causal-lm` loads `AutoModelForCausalLM` and
`AutoTokenizer`; supply a compatible chat template and LoRA target-module names.
The default target modules are the Qwen projection/MLP layers. Other
architectures are not automatically validated. Remote custom code is disabled.
For a Hugging Face model ID, omit `--local-files-only` and pin `--revision` to a
commit when reproducibility matters.

This retains the original selected script's **full-conversation causal-LM
objective**: user, assistant and tool text rendered by the model's chat template
can all contribute to the loss; only batch padding is masked. It is not
assistant-only SFT. Rendering of reasoning and tools is controlled by that
template. Inspect it before training. JSON-encoded tool arguments are converted
to objects for Transformers tool templates.

Records exceeding `--max-length` fail instead of cutting a tool interaction.
The entrypoint loads the tokenized dataset in memory. It uses cosine scheduling,
10% warmup, gradient checkpointing and epoch checkpoints by default. With very
small runs the first update can have zero learning rate due to warmup. Use
`--help` for exposed settings; no `device_map="auto"`, silent distributed
launcher or experiment-specific task exclusions are used.

The final adapter and processor/tokenizer are saved under `outputs/sft-run/final`.
`training_record.json` records the dataset hash, settings, package versions and
training metrics. A nonempty output directory is refused. Local output may
contain model paths in adapter metadata or run records; inspect artifacts
separately before publishing them. The entrypoint does not upload artifacts.

## Merge the adapter

```bash
python -m training.sft.merge_lora \
  --base /path/to/base-model --model-kind qwen3-vl --local-files-only \
  --adapter outputs/sft-run/final --output outputs/merged-model
```

The base must be the exact model/revision used for training. It is supplied
explicitly rather than recovered from a developer's path in adapter metadata.
Merging runs on CPU (float32 by default), checks for nonfinite merged weights,
and saves safetensors plus the processor/tokenizer. Ensure enough host memory
for the full model. This merger targets adapters from this entrypoint, which
does not add vocabulary or train the tokenizer. Existing nonempty output is
refused.

## RL task manifests

```bash
python -m training.rl.prepare_prompts \
  --tasks-root /path/to/claw-eval/tasks --task-ids YOUR_TASK_ID \
  --output outputs/rl-tasks.jsonl
```

Only the explicit selection is included. Each row has `task_id`, `task_path`
relative to `--tasks-root`, and a user `prompt` containing the task ID. That ID
is a handle for a custom rollout resolver, **not the natural-language task
instruction**. A trainer must resolve it to its trusted task/environment.
`--absolute-paths` is available for private deployment manifests.

This command checks directory containment and YAML structure. It does not run
tasks, infer a train/test split or claim compatibility with a particular Slime
version. Supply your own training split; selecting official evaluation tasks
does not create an unseen-task evaluation.

## RL episode reward preprocessing

`training.rl.group_rewards` exposes `process_group` and `process_groups` without
importing any trainer. Each group must explicitly contain the expected number
of samples from the same task/initial condition. Each sample carries a scalar
`reward`, or a reward mapping with an explicitly selected `reward_key`.

- **GRPO-style:** subtract the group mean, then divide by the larger of the
  population standard deviation and `std_floor` (default `0.05`). Variance at
  or below `epsilon` (default `1e-12`) gives zero advantages.
- **GDPO-style:** normalize each `reward_components` dimension within its group,
  combine the normalized dimensions with nonnegative weights summing to one,
  then normalize the combined signal over valid samples in the supplied batch.
  Default component weights are completion `0.8` and robustness `0.2`; callers
  can override `reward_weights`. An optional `safety_key` explicitly multiplies
  components by a value in `[0, 1]` **before** normalization. It is off by default.

Both produce one scalar advantage per trajectory. Neither is a temporal
estimator, GAE, state-group advantage or process-reward implementation.
`process_group` returns the pre-batch-normalization result; use `process_groups`
for the complete GDPO-style preprocessing.

Missing/nonfinite rewards, infrastructure failures, incomplete groups or missing
GDPO components reject the entire affected group. There is **no** fallback from
missing components to an aggregate reward. The batch result retains slot order
and provides `valid_mask`, `raw_rewards`, `advantages` and per-group reasons.
Invalid slots have placeholder zero advantages and null rewards; they must be
excluded from **all** trainer losses (including KL/entropy), not trained as
zero-reward failures. Zero-variance valid groups remain valid but supply zero
policy-gradient advantage. A zero safety multiplier does not imply zero
advantage after relative normalization.

For offline inspection, provide JSONL with one `{"samples": [...]}` group per
line:

```bash
python -m training.rl.group_rewards \
  --input /path/to/reward-groups.jsonl --output outputs/advantages.jsonl \
  --method grpo --group-size 8
```

For GDPO, provide `reward_components` with the two default dimensions or call
the Python API with explicit weights. The command treats all input lines as
one batch. During distributed training, the caller must define the common
group/batch scope, apply the validity mask and avoid normalizing the result a
second time. This module is not a drop-in Slime filter or an implementation of
the PPO policy loss. It does not implement resampling or retry policy.

## Release checks

```bash
python -m unittest discover -s tests -p 'test_training_data.py' -v
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_sft_runtime.py' -v
```

The data/reward tests cover tool pairing, zero/missing rewards, atomic output,
group rejection, normalization and slot alignment. Optional runtime tests use
local randomly initialized tiny Qwen3 and Qwen3-VL models on CPU. They execute
actual LoRA updates, save/reload adapters, merge weights and compare logits;
they also check padding labels and refusal to truncate. They require no model
download or API and are regression tests, not benchmark demonstrations.

Dependencies were installed and compatibility-checked in an isolated Python
3.12 environment. These release checks do not repeat the original full-scale
GPU training or certify equivalence to historical experiment results. Full RL
rollout, distributed training, process-reward/rubric calibration and the
remaining upstream patches/configurations are outside this selected release.
