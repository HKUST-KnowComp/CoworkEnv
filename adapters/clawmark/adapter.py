"""Adapt trusted filesystem/email ClawMark tasks to Harbor multi-step tasks.

Source task modules are imported during conversion, but stage seeding functions
are only executed inside the task container. Only literal stage notifications
are supported. This is an adapter, not the official ClawMark runner.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ADAPTER_DIR / 'template/tests'
RUNTIME_DIR = ADAPTER_DIR / 'runtime'
SUPPORTED_ENVIRONMENTS = {'filesystem', 'email'}


@dataclass
class ParsedTask:
    source_id: str
    domain_task: str
    environments: list[str]
    prompt: str
    stage_ids: list[str]
    seeding_stages: list[str]
    notifications: dict[str, str]
    task_py_src: str
    has_assets: bool


def contained_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Expected a relative path within the source checkout')
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Path escapes source checkout')
    return result


def _import_task_module(task_py: Path):
    name = '_clawmark_' + task_py.parent.parent.name + '_' + task_py.parent.name
    spec = importlib.util.spec_from_file_location(name, task_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _static_notification(tree: ast.Module, stage: str) -> str:
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == stage)
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if len(returns) != 1 or returns[0] not in fn.body:
        raise ValueError(f'{stage}: conditional/dynamic notifications are not supported')
    value = returns[0].value
    if isinstance(value, ast.Dict):
        for key, item in zip(value.keys, value.values):
            if isinstance(key, ast.Constant) and key.value in {'notification', 'message', 'prompt', 'text'}:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    return item.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    raise ValueError(f'{stage}: expected a literal notification; no seeding is run during conversion')


def parse_task(task_dir: Path) -> ParsedTask:
    src = (task_dir / 'task.py').read_text(encoding='utf-8')
    module = _import_task_module(task_dir / 'task.py')
    meta = module.METADATA
    rubric = module.RUBRIC
    source_id = meta.get('id', task_dir.parent.name + '_' + task_dir.name)
    if not isinstance(source_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', source_id):
        raise ValueError('Invalid task id')
    if not isinstance(rubric, dict) or not rubric:
        raise ValueError('RUBRIC must be a nonempty stage mapping')
    stages = list(rubric)
    for stage, entries in rubric.items():
        if not re.fullmatch(r'stage[0-9]+|final', stage):
            raise ValueError(f'Unsupported stage id: {stage}')
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'{stage}: empty or malformed rubric')
        weights = [float(e.get('weight', 1.0)) for e in entries]
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f'{stage}: weights must be finite/nonnegative with positive total')
        if any(not callable(e.get('checker')) for e in entries):
            raise ValueError(f'{stage}: checker must be callable')
    seeding = [s for s in stages if callable(getattr(module, s, None))]
    tree = ast.parse(src)
    notes = {s: _static_notification(tree, s) for s in seeding}
    prompt = getattr(module, 'PROMPT', '') or ''
    if not isinstance(prompt, str):
        raise ValueError('PROMPT must be text')
    return ParsedTask(source_id, f'{task_dir.parent.name}/{task_dir.name}',
                      list(meta.get('environments', [])), prompt.strip(), stages,
                      seeding, notes, src, (task_dir / 'assets').is_dir())


def offline_blockers(parsed: ParsedTask) -> list[str]:
    reasons = []
    unsupported = set(parsed.environments) - SUPPORTED_ENVIRONMENTS
    if unsupported:
        reasons.append('unsupported runtime environments: ' + ', '.join(sorted(unsupported)))
    if re.search(r'ctx\.(notion|google_sheets|sheets|calendar|snapshots)\b', parsed.task_py_src):
        reasons.append('task references an unsupported backend or historical snapshots')
    if not parsed.environments:
        reasons.append('no environments declared')
    return reasons


def _task_toml(parsed: ParsedTask, reward_strategy: str) -> str:
    steps = '\n'.join(f'[[steps]]\nname = {json.dumps(s)}\n[steps.agent]\ntimeout_sec = 600.0\n[steps.verifier]\ntimeout_sec = 120.0\n' for s in parsed.stage_ids)
    return f'''version = "1.0"
multi_step_reward_strategy = "{reward_strategy}"

[metadata]
author_name = "ClawMark / CoworkEnv adapter"
source_id = {json.dumps(parsed.source_id)}
source_task = {json.dumps(parsed.domain_task)}
environments = {json.dumps(parsed.environments)}
grading = "rule-weighted-per-stage"

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
network_mode = "public"

{steps}'''


def _dockerfile(parsed: ParsedTask) -> str:
    copy = 'COPY assets/ /workspace/\n' if parsed.has_assets else ''
    return f'''FROM python:3.12-slim
# Source managers in the released filesystem/email path use the standard library.
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
ENV TMP_WORKSPACE=/workspace
RUN mkdir -p /workspace/workspace
{copy}WORKDIR /workspace
CMD ["/bin/bash"]
'''


def _compose_yaml() -> str:
    return '''services:
  main:
    depends_on:
      greenmail:
        condition: service_started
    environment:
      EMAIL_SMTP_SERVER: greenmail
      EMAIL_SMTP_PORT: "3025"
      EMAIL_IMAP_SERVER: greenmail
      EMAIL_IMAP_PORT: "3143"
  greenmail:
    image: greenmail/standalone:2.1.8
    hostname: greenmail
    environment:
      GREENMAIL_OPTS: "-Dgreenmail.setup.test.all -Dgreenmail.hostname=0.0.0.0 -Dgreenmail.auth.disabled"
    expose:
      - "3025"
      - "3143"
'''


def _copy_assets(src: Path, dst: Path):
    if any(p.is_symlink() for p in [src, *src.rglob('*')]):
        raise ValueError('Task assets must not contain symlinks')
    shutil.copytree(src, dst)


class ClawMarkAdapter:
    NAME = 'clawmark'

    def __init__(self, out_dir: Path, *, reward_strategy: str = 'mean', oracle_root: Path | None = None):
        if reward_strategy not in {'mean', 'final'}:
            raise ValueError('reward_strategy must be mean or final')
        self.out_dir = Path(out_dir)
        self.reward_strategy = reward_strategy
        self.oracle_root = oracle_root

    def generate_task(self, task_dir: Path) -> Path:
        parsed = parse_task(task_dir)
        blockers = offline_blockers(parsed)
        if blockers:
            raise ValueError('; '.join(blockers))
        out = self.out_dir / parsed.source_id
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite {out}')
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.convert-', dir=self.out_dir) as tmp:
            stage_root = Path(tmp)
            env = stage_root / 'environment'
            env.mkdir()
            if parsed.has_assets:
                _copy_assets(task_dir / 'assets', env / 'assets')
            (env / 'Dockerfile').write_text(_dockerfile(parsed))
            if 'email' in parsed.environments:
                (env / 'docker-compose.yaml').write_text(_compose_yaml())
            (stage_root / 'task.toml').write_text(_task_toml(parsed, self.reward_strategy))
            # A copied task retains its upstream terms, even though supplied externally.
            shutil.copy2(RUNTIME_DIR / 'clawmark_vendor/LICENSE', stage_root / 'CLAWMARK_LICENSE')

            def payload(dest: Path, *, verifier: bool):
                dest.mkdir(parents=True)
                (dest / 'clawmark_task.py').write_text(parsed.task_py_src)
                shutil.copy2(RUNTIME_DIR / 'ctx_runtime.py', dest / 'ctx_runtime.py')
                shutil.copytree(RUNTIME_DIR / 'clawmark_vendor', dest / 'clawmark_vendor', ignore=shutil.ignore_patterns('__pycache__'))
                runner = 'test_runner.py' if verifier else 'stage_setup.py'
                shutil.copy2(TEMPLATE_DIR / runner, dest / runner)
                # Later stages can reference inject/<stage>/, not only assets/.
                for name in ('assets', 'inject'):
                    if (task_dir / name).is_dir():
                        _copy_assets(task_dir / name, dest / name)

            for stage in parsed.stage_ids:
                step = stage_root / 'steps' / stage
                payload(step / 'tests', verifier=True)
                text = '\n\n'.join(x for x in [parsed.prompt, parsed.notifications.get(stage, '')] if x)
                if not text:
                    text = f'Complete the task ({stage}). Review your workspace before final grading.'
                (step / 'instruction.md').write_text(text + '\n')
                (step / 'tests/test.sh').write_text(f'#!/bin/bash\nset -euo pipefail\ncd /tests\npython3 test_runner.py {stage}\n')
                (step / 'tests/test.sh').chmod(0o755)
                if stage in parsed.seeding_stages:
                    workdir = step / 'workdir'
                    seed = workdir / '.coworkenv-seed'
                    payload(seed, verifier=False)
                    (workdir / 'setup.sh').write_text(f'''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Remove seeding source (which also defines checkers) before the agent starts.
trap 'rm -rf -- .coworkenv-seed' EXIT
PYTHONDONTWRITEBYTECODE=1 python3 .coworkenv-seed/stage_setup.py {stage}
''')
                    (workdir / 'setup.sh').chmod(0o755)
                # Only an explicitly provided real oracle is emitted; no placeholder.
                if self.oracle_root is not None:
                    oracle = contained_path(self.oracle_root, f'{parsed.source_id}/{stage}.sh')
                    if oracle.is_file():
                        (step / 'solution').mkdir()
                        shutil.copy2(oracle, step / 'solution/solve.sh')
                        (step / 'solution/solve.sh').chmod(0o755)
            shutil.move(str(stage_root), out)
        return out
