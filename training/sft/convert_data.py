"""Convert text ATIF or Harbor conversation JSONL into validated SFT messages.

Adapted from train/convert_sft_data.py. No fabricated tool responses, implicit
system prompt, truncation or missing-reward-to-zero conversion is performed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from training.jsonl import read_records, write_records
from training.sft.validate_data import tool_arguments, validate_messages


def text_content(value):
    if not isinstance(value, str):
        raise ValueError('Text-only conversion requires string content')
    # Retain the original project convention for encoded text observation blocks.
    try:
        obj = json.loads(value)
    except ValueError:
        return value
    if isinstance(obj, dict) and obj.get('type') == 'text' and isinstance(obj.get('text'), str):
        return obj['text']
    return value


def atif_to_messages(trajectory, *, strip_mcp_prefix=False):
    steps = trajectory.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('Missing ATIF steps')
    messages = []
    for step in steps:
        source = step.get('source')
        if source not in {'system', 'user', 'agent'}:
            raise ValueError('Unsupported ATIF step source')
        msg = {'role': 'assistant' if source == 'agent' else source,
               'content': text_content(step.get('message', ''))}
        if step.get('reasoning_content'):
            msg['reasoning_content'] = text_content(step['reasoning_content'])
        calls = step.get('tool_calls', [])
        if calls:
            if source != 'agent':
                raise ValueError('Only agent steps can issue tools')
            msg['tool_calls'] = []
            for call in calls:
                name = call.get('function_name', '')
                if strip_mcp_prefix and isinstance(name, str) and name.startswith('mcp__'):
                    name = name.split('__')[-1]
                msg['tool_calls'].append({
                    'id': call.get('tool_call_id'), 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(tool_arguments(call.get('arguments')), ensure_ascii=False)},
                })
        messages.append(msg)
        results = (step.get('observation') or {}).get('results', [])
        if results and not calls:
            raise ValueError('Observation results have no corresponding tool calls')
        for result in results:
            messages.append({'role': 'tool', 'tool_call_id': result.get('source_call_id'),
                             'content': text_content(result.get('content', ''))})
    return messages


def convert_record(record, *, strip_mcp_prefix=False):
    if 'messages' in record or 'conversations' in record:
        messages = copy.deepcopy(record.get('messages', record.get('conversations')))
        if strip_mcp_prefix:
            raise ValueError('--strip-mcp-prefix applies only to raw ATIF input')
    else:
        trajectory = record.get('trajectory', record)
        if not isinstance(trajectory, dict):
            raise ValueError('trajectory must be an object')
        messages = atif_to_messages(trajectory, strip_mcp_prefix=strip_mcp_prefix)
    validate_messages(messages)
    out = {'messages': messages}
    for key in ('task_name', 'task_id', 'tools'):
        if key in record:
            out[key] = record[key]
    reward = record.get('reward', record.get('result'))
    if reward is not None:
        if isinstance(reward, bool):
            raise ValueError('Reward must be numeric')
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError('Reward must be finite')
        out['reward'] = reward
    return out


def convert_file(source, output, *, overwrite=False, strip_mcp_prefix=False):
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError('Input and output must be different files')
    def records():
        for i, record in enumerate(read_records(source), 1):
            try:
                yield convert_record(record, strip_mcp_prefix=strip_mcp_prefix)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f'Cannot convert record {i}: {exc}') from exc
    return write_records(records(), output, overwrite=overwrite)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--strip-mcp-prefix', action='store_true')
    args = p.parse_args()
    print(f'Converted {convert_file(args.input, args.output, overwrite=args.overwrite, strip_mcp_prefix=args.strip_mcp_prefix)} records')


if __name__ == '__main__':
    main()
