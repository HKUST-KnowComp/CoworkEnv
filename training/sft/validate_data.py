"""Validate text/tool SFT data and compute input provenance.

Adapted from train/sft_preflight.py. System messages are optional; tool calls
must have exactly one matching response before the conversation continues.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from training.jsonl import read_records


def tool_arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError('Tool arguments must be a JSON object') from exc
    if not isinstance(value, dict):
        raise ValueError('Tool arguments must be an object or an encoded object')
    json.dumps(value, allow_nan=False)
    return value


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    seen, pending, roles = set(), set(), set()
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f'message {i} must be an object')
        role = message.get('role')
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise ValueError(f'message {i}: unsupported role')
        roles.add(role)
        if role == 'system' and i != 0:
            raise ValueError('Only a leading system message is supported')
        content = message.get('content', '')
        if not isinstance(content, str):
            raise ValueError('Only text content is supported; media must not be silently discarded')
        reasoning = message.get('reasoning_content', '')
        if not isinstance(reasoning, str):
            raise ValueError('reasoning_content must be text')
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or (calls and role != 'assistant'):
            raise ValueError('tool_calls must be a list on an assistant message')
        if role == 'tool':
            cid = message.get('tool_call_id')
            if not isinstance(cid, str) or cid not in pending:
                raise ValueError('Unmatched or duplicate tool result')
            pending.remove(cid)
        elif pending:
            raise ValueError('Missing tool results before the next conversation turn')
        if role == 'assistant' and not (content or reasoning or calls):
            raise ValueError('Empty assistant turn')
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError('Invalid tool call')
            cid, fn = call.get('id'), call.get('function')
            if not isinstance(cid, str) or not cid or cid in seen:
                raise ValueError('Tool call IDs must be nonempty and unique within a trajectory')
            if call.get('type', 'function') != 'function' or not isinstance(fn, dict):
                raise ValueError('Expected a function tool call')
            if not isinstance(fn.get('name'), str) or not fn['name']:
                raise ValueError('Missing tool name')
            tool_arguments(fn.get('arguments'))
            seen.add(cid)
            pending.add(cid)
    if pending:
        raise ValueError('Trajectory ends with missing tool results')
    if not {'user', 'assistant'} <= roles:
        raise ValueError('A training record needs both user and assistant turns')


def template_messages(messages):
    """Transform API argument strings to objects expected by HF tool templates."""
    validate_messages(messages)
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls', []):
            call['function']['arguments'] = tool_arguments(call['function']['arguments'])
    return result


def inspect_jsonl(path):
    path = Path(path)
    counts = {'record_count': 0, 'tool_call_count': 0, 'tool_result_count': 0}
    for index, record in enumerate(read_records(path), 1):
        try:
            validate_messages(record.get('messages'))
        except ValueError as exc:
            raise ValueError(f'record {index}: {exc}') from exc
        counts['record_count'] += 1
        counts['tool_call_count'] += sum(len(m.get('tool_calls', [])) for m in record['messages'])
        counts['tool_result_count'] += sum(m['role'] == 'tool' for m in record['messages'])
    if not counts['record_count']:
        raise ValueError('Dataset is empty')
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **counts}


def refuse_nonempty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('Output must be a new or empty directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    print(json.dumps(inspect_jsonl(parser.parse_args().data), indent=2))


if __name__ == '__main__':
    main()
