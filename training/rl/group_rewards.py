"""GRPO/GDPO-style episode reward preprocessing, without a trainer dependency.

Adapted from claw_rl/group_filter_dim.py. Groups are supplied explicitly; this
module does not infer prompt groups, run rollouts or implement a policy loss.
Invalid members reject the entire group. Missing components never fall back to
aggregate rewards. Callers must exclude invalid slots from every training loss.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence
from training.jsonl import read_records, write_records


@dataclass(frozen=True)
class GroupDecision:
    keep: bool
    reason: str
    raw_rewards: tuple[float, ...]
    advantages: tuple[float, ...]


def finite_number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('Expected a finite reward')
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Expected a finite reward') from exc
    if not math.isfinite(number):
        raise ValueError('Expected a finite reward')
    return number


def reward_value(sample, reward_key=None):
    value = sample.get('reward')
    if isinstance(value, dict):
        if reward_key is None or reward_key not in value:
            raise ValueError('Dictionary rewards require an existing reward_key')
        value = value[reward_key]
    return finite_number(value)


def _normalize(values, *, epsilon, std_floor):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
    if not math.isfinite(variance):
        raise ValueError('Reward variance overflow')
    if variance <= epsilon:
        return [0.0] * len(values), 'zero_variance'
    scale = math.sqrt(variance)
    reason = 'accepted_floored' if scale < std_floor else 'accepted'
    return [(x - mean) / max(scale, std_floor) for x in values], reason


def _config(expected_size, epsilon, std_floor):
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 2:
        raise ValueError('expected_size must be an integer >= 2')
    if any(not math.isfinite(x) or x < 0 for x in (epsilon, std_floor)):
        raise ValueError('epsilon and std_floor must be finite and nonnegative')


def _validate_group(samples, expected_size, reward_key):
    if len(samples) != expected_size:
        raise ValueError('incomplete_group')
    rewards = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError('invalid_sample')
        if sample.get('valid_for_grpo', True) is False or sample.get('remove_sample', False) or sample.get('infra_error'):
            raise ValueError('infra_failure')
        rewards.append(reward_value(sample, reward_key))
    return rewards


def process_group(samples, *, method='grpo', expected_size=8, reward_key=None,
                  reward_weights=None, safety_key=None, epsilon=1e-12, std_floor=.05):
    _config(expected_size, epsilon, std_floor)
    if method not in {'grpo', 'gdpo'}:
        raise ValueError('method must be grpo or gdpo')
    weights = {'completion': .8, 'robustness': .2} if reward_weights is None else dict(reward_weights)
    if method == 'gdpo':
        if not weights or any(not isinstance(k, str) or not k for k in weights):
            raise ValueError('Reward component names are required')
        weights = {k: finite_number(v) for k, v in weights.items()}
        if any(v < 0 for v in weights.values()) or not math.isclose(math.fsum(weights.values()), 1.0):
            raise ValueError('Component weights must be nonnegative and sum to 1')
    try:
        raw = _validate_group(samples, expected_size, reward_key)
        if method == 'grpo':
            adv, reason = _normalize(raw, epsilon=epsilon, std_floor=std_floor)
        else:
            components = {key: [] for key in weights}
            for sample in samples:
                comp = sample.get('reward_components')
                if not isinstance(comp, Mapping):
                    raise ValueError('missing_components')
                safety = finite_number(comp.get(safety_key)) if safety_key is not None else 1.0
                if not 0 <= safety <= 1:
                    raise ValueError('invalid_safety_multiplier')
                for key in weights:
                    components[key].append(finite_number(comp.get(key)) * safety)
            normalized, reasons = {}, []
            for key, values in components.items():
                normalized[key], reason = _normalize(values, epsilon=epsilon, std_floor=std_floor)
                reasons.append(reason)
            adv = [math.fsum(weights[k] * normalized[k][i] for k in weights) for i in range(expected_size)]
            reason = ('zero_variance' if all(r == 'zero_variance' for r in reasons) else
                      'accepted_floored' if 'accepted_floored' in reasons else 'accepted')
        return GroupDecision(True, reason, tuple(raw), tuple(adv))
    except (ValueError, OverflowError) as exc:
        return GroupDecision(False, str(exc), (), ())


def process_groups(groups: Sequence[Sequence[Mapping]], **options):
    if not groups or any(not isinstance(g, (list, tuple)) or not g for g in groups):
        raise ValueError('Supply nonempty explicit sample groups')
    decisions = [process_group(group, **options) for group in groups]
    advantages, rewards, mask = [], [], []
    for group, decision in zip(groups, decisions):
        advantages.extend(decision.advantages if decision.keep else [0.0] * len(group))
        rewards.extend(decision.raw_rewards if decision.keep else [None] * len(group))
        mask.extend([decision.keep] * len(group))
    if options.get('method', 'grpo') == 'gdpo':
        indices = [i for i, valid in enumerate(mask) if valid]
        if indices:
            values, _ = _normalize([advantages[i] for i in indices], epsilon=options.get('epsilon', 1e-12), std_floor=0)
            for i, value in zip(indices, values):
                # Per-group normalization is zero-mean. Keep truly zero signals
                # at literal zero rather than introducing subtraction roundoff.
                advantages[i] = value if advantages[i] != 0 else 0.0
    return {'raw_rewards': rewards, 'advantages': advantages, 'valid_mask': mask,
            'groups': [{'keep': d.keep, 'reason': d.reason, 'size': len(g)} for g, d in zip(groups, decisions)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='JSONL: one {"samples": [...]} group per row')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--method', choices=['grpo', 'gdpo'], default='grpo')
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--std-floor', type=float, default=.05)
    p.add_argument('--reward-key')
    p.add_argument('--safety-key', help='Optional explicit component multiplier in [0,1]')
    args = p.parse_args()
    groups = [record['samples'] for record in read_records(args.input)]
    result = process_groups(groups, method=args.method, expected_size=args.group_size,
                            std_floor=args.std_floor, reward_key=args.reward_key, safety_key=args.safety_key)
    write_records([result], args.output)
    print('Accepted groups:', sum(g['keep'] for g in result['groups']), '/', len(groups))


if __name__ == '__main__':
    main()
