#!/usr/bin/env python3
"""Grade bridge-recorded dispatches and a harness-owned ATIF final response.

Agent-authored trace.jsonl is never read. Missing/malformed evidence fails
verification with a nonzero exit status, not a zero task reward. Failed tool
calls retain their HTTP error status and are handled by the task grader.
This adapter supports the unchanged text/tool Pinbench grading implementation.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
VERIFIER_DIR = Path('/logs/verifier')
TRAJECTORY_PATH = Path('/logs/agent/trajectory.json')
LIVE_CONFIG_PATH = TESTS_DIR / 'live_config.json'
sys.path.insert(0, str(TESTS_DIR))
from claw_eval.graders.base import AbstractGrader
from claw_eval.models.message import Message
from claw_eval.models.scoring import compute_task_score
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import ToolDispatch, TraceMessage


def _load_grader_class():
    spec = importlib.util.spec_from_file_location('task_grader', TESTS_DIR / 'grader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, AbstractGrader) and obj is not AbstractGrader and obj.__module__ == module.__name__:
            return obj
    raise ValueError('No task grader subclass found')


def _final_assistant_text(path: Path) -> str:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('steps'), list) or not data['steps']:
        raise ValueError('Missing ATIF steps')
    if any(not isinstance(step, dict) or step.get('source') not in {'agent', 'user', 'system'} for step in data['steps']):
        raise ValueError('Invalid ATIF steps')
    for step in reversed(data['steps']):
        if step.get('source') != 'agent' or step.get('tool_calls'):
            continue
        message = step.get('message')
        if isinstance(message, str) and not message.startswith('Executed '):
            return message
        if isinstance(message, list):
            text = '\n'.join(x['text'] for x in message if isinstance(x, dict) and isinstance(x.get('text'), str))
            if text:
                return text
    return ''


def _load_dispatches(config: dict) -> list[ToolDispatch]:
    with urllib.request.urlopen(config['dispatch_url'], timeout=10) as response:
        data = json.load(response)
    if data.get('task_id') != config['task_id'] or not isinstance(data.get('dispatches'), list):
        raise ValueError('Bridge evidence does not match the task')
    allowed = set(config['tool_names'])
    dispatches = []
    for i, record in enumerate(data['dispatches']):
        if record.get('tool_name') not in allowed or not isinstance(record.get('response_status'), int):
            raise ValueError('Invalid bridge dispatch')
        dispatches.append(ToolDispatch(trace_id='live', tool_use_id=f'bridge_{i}', **record))
    return dispatches


def main() -> int:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = VERIFIER_DIR / 'reward.json'
    reward_path.unlink(missing_ok=True)
    (VERIFIER_DIR / 'error.json').unlink(missing_ok=True)
    try:
        config = json.loads(LIVE_CONFIG_PATH.read_text())
        dispatches = _load_dispatches(config)
        final_text = _final_assistant_text(TRAJECTORY_PATH)
        task = TaskDefinition(task_id=config['task_id'], task_name=config['task_id'], prompt={'text': ''})
        messages = [TraceMessage(trace_id='live', message=Message(role='assistant', content=final_text))]
        scores = _load_grader_class()().grade(messages, dispatches, task, audit_data={}, judge=None, media_events=[])
        reward = compute_task_score(scores)
        reward_path.write_text(json.dumps({'reward': reward, 'completion': scores.completion,
                                         'robustness': scores.robustness, 'safety': scores.safety}))
        (VERIFIER_DIR / 'scores.json').write_text(scores.model_dump_json(indent=2))
    except Exception as exc:
        (VERIFIER_DIR / 'error.json').write_text(json.dumps({'status': 'invalid_evidence', 'error': str(exc)}))
        print(f'[verifier] invalid evidence: {exc}', file=sys.stderr)
        return 1
    print(f'[verifier] reward={reward}; dispatches={len(dispatches)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
